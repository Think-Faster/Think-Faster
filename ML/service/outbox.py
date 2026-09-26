"""Исходящий поток (INTEGRATION §2.3): одно сообщение на объект и час.

Формат зафиксирован в §2.3 — `tf.forecast.results`, ключ partition/ordering — object_id. Здесь:
сборка словаря сообщения (чистая) и два получателя — Kafka (прод) и файл ndjson (стенд/проверка).
Поля reasons/confidence/evidence только для тревог (§2.3); схема — целое поле, с ней интерфейсу
легче мигрировать.
"""
import json
from pathlib import Path

import svc as config


def build_message(object_id: int, hour_end: str, model_version: str,
                  types: list, horizon: int = 24, clock: str = 'live', kind: str = 'forecast') -> dict:
    """`clock: replay` — демонстрационное время (13.1): BFF по таким сообщениям заявок не создаёт."""
    return {'schema': 1, 'kind': kind, 'object_id': object_id, 'hour_end': hour_end,
            'horizon_hours': horizon, 'model_version': model_version, 'clock': clock,
            'types': {tp: {'score': 0.0, 'threshold': 0.0, 'alarm': False} for tp in types}}


def fill_type(msg: dict, tp: str, *, score, threshold, alarm, since_hours=None,
              reasons=None, confidence=None, evidence=None) -> None:
    blk = {'score': float(score), 'threshold': float(threshold), 'alarm': bool(alarm)}
    if alarm:
        if since_hours is not None:
            blk['since_hours'] = int(since_hours)
        if reasons:
            blk['reasons'] = reasons
        if confidence is not None:
            blk['confidence'] = float(confidence)
        if evidence:
            blk['evidence'] = evidence
    msg['types'][tp] = blk


def build_fact(object_id: int, hour_end: str, model_version: str, types: dict,
               clock: str = 'live') -> dict:
    """Канал «по факту» (M8, §2.3): только типы, у которых эпизод живой в этот час. У типа —
    `started_at`, `last_at`, `new` (объявление или обновление); в окне графика работ — `note` и
    `work_id` (§1.6: газ по коллектору в ППР приходит с пометкой, диспетчер закрывает причиной
    «известные работы на объекте»)."""
    return {'schema': 1, 'kind': 'fact', 'object_id': object_id, 'hour_end': hour_end,
            'model_version': model_version, 'clock': clock, 'types': types}


def attach_recommendation(msg: dict, rec: dict) -> None:
    """M13 (§2.3): у каждого тревожного типа — своя `recommendation`; если типов несколько, в
    сообщении ещё `object_recommendation` — меры всех типов по сроку и один `visit`."""
    parts = rec.get('parts')
    if parts is None:
        if rec.get('type') in msg['types']:
            msg['types'][rec['type']]['recommendation'] = rec
        return
    for r in parts:
        if r.get('type') in msg['types']:
            msg['types'][r['type']]['recommendation'] = r
    msg['object_recommendation'] = {k: v for k, v in rec.items() if k != 'parts'}


class Sink:
    def send(self, msg: dict) -> None: ...


class FileSink(Sink):
    """ndjson по одному сообщению в строке — для стенда и проверки формата."""

    def __init__(self, path: Path | None = None):
        self.path = path or (config.OUT_DIR / 'results.ndjson')
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, msg: dict) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(msg, ensure_ascii=False) + '\n')


class KafkaSink(Sink):
    """tf.forecast.results + tf.dlq (Н2). confluent-kafka опционален (см. KafkaReader).

    Доставка подтверждается брокером (acks=all, идемпотентный продюсер); что не доставилось за
    такт — в tf.dlq с причиной и в счётчик `failed` для наблюдения M10.
    """

    def __init__(self):
        try:
            from confluent_kafka import Producer
        except ImportError as e:
            raise RuntimeError('confluent-kafka не установлен (ML/service/requirements.txt)') from e
        self.producer = Producer(config.kafka_conf(**{'message.max.bytes': config.MAX_MSG_BYTES,
                                                      'acks': 'all', 'enable.idempotence': True}))
        self.failed = 0

    def _done(self, err, msg) -> None:
        if err is None or msg.topic() == config.TOPIC_DLQ:
            self.failed += err is not None
            return
        self.failed += 1
        self.producer.produce(config.TOPIC_DLQ, key=msg.key(), value=msg.value(),
                              headers={'reason': str(err), 'topic': msg.topic()})

    def send(self, msg: dict) -> None:
        key = str(msg['object_id']).encode()
        value = json.dumps(msg, ensure_ascii=False).encode()
        self.producer.produce(config.TOPIC_RESULTS, key=key, value=value, on_delivery=self._done)
        self.producer.poll(0)

    def dead(self, value: bytes, key: bytes | None, reason: str, topic: str) -> None:
        """Входное сообщение, которое не разобралось (приём M1), — в tf.dlq с причиной."""
        self.producer.produce(config.TOPIC_DLQ, key=key, value=value,
                              headers={'reason': reason[:500], 'topic': topic}, on_delivery=self._done)
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush(30)


class LastSink(Sink):
    """Последнее сообщение по каждому объекту — для GET /forecast (13.1) и после рестарта."""

    def __init__(self, inner: Sink, path: Path | None = None):
        self.inner, self.path = inner, path or config.LAST_RESULTS
        self.last: dict[int, dict] = {}
        if self.path.exists():
            self.last = {int(k): v for k, v in json.loads(self.path.read_text(encoding='utf-8')).items()}

    def send(self, msg: dict) -> None:
        self.inner.send(msg)
        if msg.get('kind', 'forecast') == 'forecast':     # факт не затирает прогноз объекта
            self.last[int(msg['object_id'])] = msg

    @property
    def failed(self) -> int:
        return getattr(self.inner, 'failed', 0)

    def flush(self) -> None:
        if hasattr(self.inner, 'flush'):
            self.inner.flush()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.last, ensure_ascii=False), encoding='utf-8')
        tmp.replace(self.path)