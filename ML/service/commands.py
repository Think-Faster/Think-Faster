"""Приём обратного потока — очередь RabbitMQ `tf.model.commands` (INTEGRATION §13.3, M11).

Публикует только BFF, модель читает. Прав `configure` у учётки `tf-model` нет, поэтому очередь
проверяется пассивно: если её нет, поток ждёт и пробует снова, сервис при этом считает прогнозы.

Подтверждение — только после того, как `Service.handle` записал состояние на том:
- ack — команда применена, либо её `command_id` уже видели, либо снимок настроек не новее текущего;
- reject без повтора — сломанное сообщение (`CommandError`, не JSON), уходит в `tf.dlq`;
- nack с повтором — временная ошибка (нет модели, диск); с паузой, чтобы не крутить её вхолостую.
  На пятой доставке — reject, и сообщение тоже в DLQ. Номер доставки — заголовок `x-delivery-count`
  у кворумной очереди, а у классической — свой счётчик по `command_id` в памяти процесса.
"""
import json
import logging
import threading
from urllib.parse import unquote, urlsplit

import svc as config
from core import CommandError

log = logging.getLogger('tf-model')
ACK, REJECT, RETRY = 'ack', 'reject', 'retry'
BACKOFF = (2, 5, 15, 30)                # секунд до повтора по номеру попытки


class Consumer:
    def __init__(self, service, retries: int = config.COMMAND_RETRIES):
        self.service, self.retries = service, retries
        self.attempts: dict[str, int] = {}

    def decide(self, body: bytes, headers: dict | None = None) -> str:
        """Что ответить брокеру на одно сообщение. Логика без pika — её проверяют тесты."""
        try:
            env = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning('команда не JSON — в DLQ')
            return REJECT
        cid = str(env.get('command_id')) if isinstance(env, dict) else None
        try:
            res = self.service.handle(env)
        except CommandError as e:
            log.warning('команда %s (%s) отклонена: %s — в DLQ', cid, (env or {}).get('kind'), e)
            self.attempts.pop(cid, None)
            return REJECT
        except Exception:
            n = max(self.attempts.get(cid, 0), int((headers or {}).get('x-delivery-count') or 0)) + 1
            self.attempts[cid] = n
            log.exception('команда %s: попытка %d из %d не удалась', cid, n, self.retries)
            if n >= self.retries:
                self.attempts.pop(cid, None)
                return REJECT
            return RETRY
        self.attempts.pop(cid, None)
        log.info('команда %s %s: %s', cid, env.get('kind'), res.get('status', 'ok') if isinstance(res, dict) else res)
        return ACK

    def delay(self, body: bytes) -> float:
        try:
            n = self.attempts.get(str(json.loads(body).get('command_id')), 1)
        except Exception:
            n = 1
        return BACKOFF[min(n, len(BACKOFF)) - 1]

    def on_message(self, conn, ch, method, props, body) -> None:
        verdict = self.decide(body, getattr(props, 'headers', None))
        tag = method.delivery_tag
        if verdict == ACK:
            ch.basic_ack(tag)
        elif verdict == REJECT:
            ch.basic_reject(tag, requeue=False)
        else:
            conn.call_later(self.delay(body), lambda: ch.basic_nack(tag, requeue=True))


def parameters(url: str = config.RABBIT_URL):
    """Адрес — из TF_RABBIT_URL без пароля; пароль учётки tf-model — из Vault secret/tf/rabbit."""
    import pika
    import tfkit
    params = pika.URLParameters(url)
    password = tfkit.secret('rabbit', 'model_password', 'TF_RABBIT_PASSWORD', required=config.ENV != 'dev')
    if password:
        params.credentials = pika.PlainCredentials(unquote(urlsplit(url).username or config.RABBIT_USER), password)
    params.heartbeat, params.blocked_connection_timeout = 60, 300
    return params


def run(service, stop: threading.Event, params=None) -> None:
    """Поток приёма команд: переподключение с паузой до 60 с, остановка — по `stop`."""
    import pika
    params = params or parameters()
    consumer, wait = Consumer(service), 2
    while not stop.is_set():
        conn = None
        try:
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            ch.queue_declare(config.QUEUE_COMMANDS, passive=True)
            ch.basic_qos(prefetch_count=1)          # по одной: порядок команд BFF сохраняется
            ch.basic_consume(config.QUEUE_COMMANDS,
                             lambda c, m, p, b: consumer.on_message(conn, c, m, p, b))
            log.info('RabbitMQ: читаю %s', config.QUEUE_COMMANDS)
            wait = 2
            while not stop.is_set():
                conn.process_data_events(time_limit=1)
        except pika.exceptions.AMQPError as e:
            log.warning('RabbitMQ: %s — снова через %d с', type(e).__name__, wait)
            stop.wait(wait)
            wait = min(wait * 2, 60)
        finally:
            if conn is not None and conn.is_open:
                try:
                    conn.close()
                except pika.exceptions.AMQPError:
                    pass


def start(service, stop: threading.Event) -> threading.Thread:
    t = threading.Thread(target=run, args=(service, stop), name='commands', daemon=True)
    t.start()
    return t
