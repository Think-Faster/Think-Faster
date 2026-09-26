"""Приём уведомлений из RabbitMQ (ML/INTEGRATION.md §13.7, think-infra/rabbitmq).

BFF публикует в exchange `tf.notifications` с ключом `email` или `telegram`, сообщение ложится в
очередь `tf.notify.email` или `tf.notify.telegram`:

    {"schema": 1, "notice_id": "uuid", "ticket_id": 1042, "kind": "fact",
     "subject": "Загазованность: объект 5122", "text": "...",
     "to": {"emails": ["..."], "chat_ids": [123]}, "request_id": "e81b07c4f2a9"}

Отправляет тот же код, что и ручки. Ответ брокеру — по правилам think-infra/rabbitmq/README.md:
- ack — отправлено. Кому-то могло не уйти по постоянной причине (адрес отвергнут, бот заблокирован):
  это `notify.sent` с числом отказов;
- reject без повтора — сообщение не разобрать или оно не ушло никому по постоянной причине:
  `tf.dlq` и `notify.failed`;
- nack с повтором — почта или Telegram недоступны, Telegram ограничил частоту; с паузой. На пятой
  доставке (`x-delivery-count` кворумной очереди) — reject и `notify.failed`.

Повтор не шлёт второй раз тем, кому уже ушло: отправленное по `notice_id` помнится сутки в Redis
(`notify:sent:<notice_id>:<канал>`). Лёг Redis — рассылка идёт без этой защиты: второе письмо лучше,
чем ни одного. Учётки брокера — `tf-notify-email` и `tf-notify-telegram`, каждая читает только свою
очередь; пароль — из Vault `secret/tf/rabbit` (`email_password`, `telegram_password`).
"""
import json
import logging
import threading
import time
import uuid
from urllib.parse import unquote, urlsplit

from email_validator import EmailNotValidError, validate_email

import mailer
import telegram_bot

log = logging.getLogger('notify')
ACK, REJECT, RETRY = 'ack', 'reject', 'retry'
BACKOFF = (5, 15, 30, 60)               # секунд до повтора по номеру попытки
QUEUES = {'email': 'tf.notify.email', 'telegram': 'tf.notify.telegram'}
USERS = {'email': 'tf-notify-email', 'telegram': 'tf-notify-telegram'}
DAY = 86_400


class BadNotice(ValueError):
    """Сообщение, которое никогда не отправится: сразу в DLQ."""


def parse(body: bytes, channel: str, settings) -> dict:
    try:
        n = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        raise BadNotice('не JSON') from None
    if not isinstance(n, dict):
        raise BadNotice('не объект')
    try:
        n['notice_id'] = str(uuid.UUID(str(n.get('notice_id'))))
    except ValueError:
        raise BadNotice('notice_id не uuid') from None
    subject, text = n.get('subject'), n.get('text')
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 255:
        raise BadNotice('тема пустая или длиннее 255')
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise BadNotice('текст пустой или длиннее 20000')
    n['subject'] = ' '.join(subject.split())        # перевод строки в теме ломает заголовки письма
    to = n.get('to') if isinstance(n.get('to'), dict) else {}
    if channel == 'email':
        raw = to.get('emails')
        if not isinstance(raw, list) or not raw:
            raise BadNotice('нет адресов почты')
        try:
            n['recipients'] = list(dict.fromkeys(
                validate_email(str(a), check_deliverability=False).normalized for a in raw))
        except EmailNotValidError:
            raise BadNotice('адрес почты с ошибкой') from None
    else:
        raw = to.get('chat_ids')
        if not isinstance(raw, list) or not raw or not all(
                isinstance(c, int) and not isinstance(c, bool) or isinstance(c, str) and c.strip() for c in raw):
            raise BadNotice('нет chat_id')
        n['recipients'] = list(dict.fromkeys(str(c).strip() for c in raw))
        if len(telegram_bot.render(n['subject'], text)) > telegram_bot.LIMIT:
            raise BadNotice(f'тема и текст длиннее {telegram_bot.LIMIT} символов — предел Telegram')
    ticket = n.get('ticket_id')
    n['ticket_id'] = None if ticket in (None, '') else str(ticket)
    return n


class Sent:
    """Кому уже ушло по `notice_id`: сутки в Redis, а без Redis — в памяти процесса."""

    def __init__(self, redis=None):
        self.r = redis
        self.mem: dict[str, tuple[float, set]] = {}

    @staticmethod
    def key(nid: str, channel: str) -> str:
        return f'notify:sent:{nid}:{channel}'

    def get(self, nid: str, channel: str) -> set[str]:
        k = self.key(nid, channel)
        if self.r is not None:
            try:
                got = self.r.smembers(k)
                return {x.decode() if isinstance(x, bytes) else x for x in got} | self._mem(k)
            except Exception as e:
                log.warning('Redis недоступен (%s): повтор без защиты от второй отправки', type(e).__name__)
        return self._mem(k)

    def _mem(self, k: str) -> set[str]:
        at, got = self.mem.get(k, (0.0, set()))
        return set(got) if time.time() - at < DAY else set()

    def add(self, nid: str, channel: str, recipients: list[str]) -> None:
        if not recipients:
            return
        k, now = self.key(nid, channel), time.time()
        self.mem[k] = (now, self._mem(k) | set(recipients))
        if len(self.mem) > 10_000:
            self.mem = {x: v for x, v in self.mem.items() if now - v[0] < DAY}
        if self.r is not None:
            try:
                p = self.r.pipeline()
                p.sadd(k, *recipients)
                p.expire(k, DAY)
                p.execute()
            except Exception as e:
                log.warning('Redis недоступен (%s): отправленное помню только в памяти', type(e).__name__)


class Consumer:
    """Одна очередь — один канал. Логика без pika: её проверяют тесты."""

    def __init__(self, channel: str, settings, audit, sent: Sent, retries: int | None = None,
                 send_mail=None, send_telegram=None):
        self.channel, self.settings, self.audit, self.sent = channel, settings, audit, sent
        self.retries = retries or settings.notify_retries
        self.send_mail = send_mail or mailer.send
        self.send_telegram = send_telegram or telegram_bot.send
        self.attempts: dict[str, int] = {}

    def decide(self, body: bytes, headers: dict | None = None) -> str:
        try:
            n = parse(body, self.channel, self.settings)
        except BadNotice as e:
            log.warning('%s: сообщение не разобрано (%s) — в DLQ', QUEUES[self.channel], e)
            self._event('notify.failed', 'error', None, reason=str(e))
            return REJECT
        nid = n['notice_id']
        done = self.sent.get(nid, self.channel)
        todo = [r for r in n['recipients'] if r not in done]
        if not todo:
            log.info('уведомление %s уже отправлено — повтор подтверждён без отправки', nid)
            self.attempts.pop(nid, None)
            return ACK
        attempt = max(self.attempts.get(nid, 0), int((headers or {}).get('x-delivery-count') or 0)) + 1
        self.attempts[nid] = attempt
        try:
            sent, failed, later = self._mail(n, todo) if self.channel == 'email' else self._telegram(n, todo)
        except Exception as e:                       # неожиданное — как временная ошибка
            log.exception('уведомление %s: сбой отправки', nid)
            sent, failed, later = [], {r: type(e).__name__ for r in todo}, list(todo)
        self.sent.add(nid, self.channel, sent)
        total_sent = len(done) + len(sent)
        if later:
            if attempt < self.retries:
                log.warning('уведомление %s: попытка %d из %d, не ушло %d — повтор', nid, attempt,
                            self.retries, len(later))
                return RETRY
            self.attempts.pop(nid, None)
            self._event('notify.failed', 'error', n, sent=total_sent, failed=len(later),
                        reason=_first(failed, later), attempt=attempt)
            return REJECT
        self.attempts.pop(nid, None)
        if total_sent:
            self._event('notify.sent', 'success', n, sent=total_sent, failed=len(failed), attempt=attempt)
            log.info('уведомление %s (%s): отправлено %d, отказ %d', nid, self.channel, total_sent, len(failed))
            return ACK
        self._event('notify.failed', 'error', n, sent=0, failed=len(failed), reason=_first(failed, todo),
                    attempt=attempt)
        return REJECT

    def _mail(self, n: dict, todo: list[str]):
        """Одно письмо на пачку до max_recipients адресатов (Gmail больше 100 не принимает)."""
        sent, failed, later = [], {}, []
        step = max(1, self.settings.max_recipients)
        for i in range(0, len(todo), step):
            chunk = todo[i:i + step]
            try:
                refused = self.send_mail(self.settings, chunk, n['subject'], n['text'])
            except mailer.MailError as e:
                if e.status == 422:                  # сервер отверг всех адресатов пачки — навсегда
                    failed.update(e.failed or {a: e.message for a in chunk})
                    continue
                rest = todo[i:]                      # сервер лёг или не принял логин — остальным тоже
                failed.update({a: e.message for a in rest})
                later += rest
                break
            failed.update(refused)
            sent += [a for a in chunk if a not in refused]
        return sent, failed, later

    def _telegram(self, n: dict, todo: list[str]):
        later: list[str] = []
        try:
            sent, failed = self.send_telegram(self.settings, todo, n['subject'], n['text'], later=later)
        except telegram_bot.TelegramError as e:      # никому не ушло: сеть, токен бота
            return [], dict(e.failed) or {c: e.message for c in todo}, list(todo)
        return sent, failed, later

    def _event(self, event: str, outcome: str, n: dict | None, **details) -> None:
        """§6.2: канал, число адресатов, тема, заявка, причина. Текста и адресов в журнале нет (§6.3)."""
        d = {'channel': self.channel, **details}
        if n is not None:
            d.update(notice_id=n['notice_id'], kind=n.get('kind'), subject=n['subject'],
                     recipients=len(n['recipients']))
        try:
            self.audit.event(event, outcome, actor_kind='service', request_id=(n or {}).get('request_id'),
                             object_type='ticket' if n and n.get('ticket_id') else None,
                             object_id=(n or {}).get('ticket_id'), details=d)
        except Exception:
            log.exception('аудит %s не записан', event)

    def delay(self, body: bytes) -> float:
        try:
            n = self.attempts.get(str(uuid.UUID(str(json.loads(body).get('notice_id')))), 1)
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


def _first(failed: dict, among: list) -> str:
    """Причина без адреса: у SMTP она бывает с адресом внутри, поэтому только код ответа."""
    for r in among:
        if r in failed:
            return str(failed[r]).split(' ', 1)[0] if str(failed[r])[:3].isdigit() else str(failed[r])[:200]
    return ''


def parameters(settings, channel: str):
    """Адрес — из TF_RABBIT_URL без пароля; учётка своего канала, пароль — из Vault."""
    import pika
    import config
    url = settings.rabbit()
    params = pika.URLParameters(url)
    params.credentials = pika.PlainCredentials(unquote(urlsplit(url).username or USERS[channel]),
                                               config.rabbit_password(settings, channel))
    params.heartbeat, params.blocked_connection_timeout = 60, 300
    return params


def run(consumer: Consumer, stop: threading.Event, params) -> None:
    """Поток одной очереди: переподключение с паузой до 60 с, остановка — по `stop`."""
    import pika
    queue, wait = QUEUES[consumer.channel], 2
    while not stop.is_set():
        conn = None
        try:
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            ch.queue_declare(queue, passive=True)       # прав configure нет: очередь заводит think-infra
            ch.basic_qos(prefetch_count=1)
            ch.basic_consume(queue, lambda c, m, p, b: consumer.on_message(conn, c, m, p, b))
            log.info('RabbitMQ: читаю %s', queue)
            wait = 2
            while not stop.is_set():
                conn.process_data_events(time_limit=1)
        except pika.exceptions.AMQPError as e:
            log.warning('RabbitMQ %s: %s — снова через %d с', queue, type(e).__name__, wait)
            stop.wait(wait)
            wait = min(wait * 2, 60)
        finally:
            if conn is not None and conn.is_open:
                try:
                    conn.close()
                except pika.exceptions.AMQPError:
                    pass


def ready(settings, channel: str) -> bool:
    """Есть чем отправлять. Нет — очередь не читается: сообщения ждут в ней (TTL сутки), а не уходят в DLQ."""
    if channel == 'email':
        return bool(settings.mail_from or settings.smtp_user)
    return bool(settings.telegram_bot_token.get_secret_value())


def start(settings, audit, stop: threading.Event) -> list[threading.Thread]:
    """Потоки очередей из NOTIFY_CHANNELS; при TF_RABBIT_URL=off (или dev без адреса) — ни одного.
    Пароль учётки не прочитался из Vault — SecretError сразу, сервис не стартует (§13.5)."""
    if settings.rabbit() is None:
        log.info('RabbitMQ выключен: уведомления только через ручки')
        return []
    channels = [c for c in settings.channels() if ready(settings, c)]
    for c in set(settings.channels()) - set(channels):
        log.warning('%s не читается: канал %s не настроен (нет секрета в Vault)', QUEUES[c], c)
    redis = None
    if settings.redis():
        import redis as redis_lib
        redis = redis_lib.Redis.from_url(settings.redis(), socket_timeout=2, socket_connect_timeout=2)
    sent, threads = Sent(redis), []
    params = {channel: parameters(settings, channel) for channel in channels}
    for channel, p in params.items():
        c = Consumer(channel, settings, audit, sent)
        t = threading.Thread(target=run, args=(c, stop, p), name=f'rabbit-{channel}', daemon=True)
        t.start()
        threads.append(t)
    return threads
