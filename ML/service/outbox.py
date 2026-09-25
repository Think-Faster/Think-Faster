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
                  types: list, horizon: int = 24) -> dict:
    return {'schema': 1, 'object_id': object_id, 'hour_end': hour_end,
            'horizon_hours': horizon, 'model_version': model_version,
            'types': {tp: {'score': 0.0, 'threshold': 0.0, 'alarm': False} for tp in types}}


def fill_type(msg: dict, tp: str, *, score, threshold, alarm, since_hours=None,
              reasons=None, confidence=None, evidence=None, recommendation=None,
              muted=None) -> None:
    blk = {'score': float(score), 'threshold': float(threshold), 'alarm': bool(alarm)}
    if muted:
        blk['muted'] = muted                      # M7: MUTED по графику работ (INTEGRATION §1.6)
    if alarm:
        if since_hours is not None:
            blk['since_hours'] = int(since_hours)
        if reasons:
            blk['reasons'] = reasons
        if confidence is not None:
            blk['confidence'] = float(confidence)
        if evidence:
            blk['evidence'] = evidence
        if recommendation:
            blk['recommendation'] = recommendation
    msg['types'][tp] = blk


def object_recommendation(msg: dict, rec: dict) -> None:
    """Составная рекомендация по объекту (§2.3), когда тревожат несколько типов разом."""
    if rec:
        msg['object_recommendation'] = rec


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
    """tf.forecast.results + tf.dlq. confluent-kafka опционален (см. KafkaReader)."""

    def __init__(self):
        try:
            from confluent_kafka import Producer
            import ingest
            ingest  # noqa: F401  (единый конфиг подключения в KafkaReader)
        except ImportError as e:
            raise RuntimeError('confluent-kafka не установлен (ML/service/requirements.txt)') from e
        conf = {'bootstrap.servers': config.KAFKA_BOOTSTRAP, 'message.max.bytes': config.MAX_MSG_BYTES}
        self.producer = Producer(conf)

    def send(self, msg: dict) -> None:
        key = str(msg['object_id']).encode()
        self.producer.produce(config.TOPIC_RESULTS, key=key, value=json.dumps(msg).encode())
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush()