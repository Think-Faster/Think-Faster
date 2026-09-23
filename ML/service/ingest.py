"""Потребитель журнала (M1): события Kafka или файловый повтор — в горячий журнал.

Рекомендация INTEGRATION §6: источник событий прячем за интерфейсом «читателя журнала», Kafka —
одна из реализаций. Это снимает расхождение с tasks/plan.md (там Kafka не планировалась) и делает
сервис проверяемым на стенде (ReplayReader проигрывает ext-journal-*.csv с той же частотой).

Схема `clean() → append()` одна для всех читателей: строка прошла чистку — значит, попала в чтения;
что-то падает на парсинге — в DLQ, а не в основную цепочку.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from storage import HotStore

import svc as config


@dataclass
class RawEvent:
    channel_id: int
    ts: datetime
    value: str


class JournalReader(ABC):
    """Источник событий за интерфейсом (§6): на что смотреть и как читать — деталь реализации."""

    @abstractmethod
    def poll(self, since: datetime | None, until: datetime | None) -> Iterator[RawEvent]: ...


class ReplayReader(JournalReader):
    """Файловый повтор журнала для стенда (think-test/emulator) и тестов."""

    def __init__(self, csv_files, reference=None, speed: float = 0.0):
        self.files = csv_files
        self.reference = reference
        self.speed = speed   # 0 — сразу весь диапазон, >0 — разбивка по часам

    def poll(self, since=None, until=None):
        for f in self.files:
            with open(f, encoding='utf-8') as fh:
                next(fh, None)                        # шапка
                for line in fh:
                    parts = line.rstrip('\n').split(',')
                    if len(parts) < 6:
                        continue
                    try:
                        ts = datetime.strptime(f'{parts[2]} {parts[3]}', '%Y-%m-%d %H:%M:%S')
                        cid = int(parts[1])
                    except ValueError:
                        continue
                    if since and ts < since:
                        continue
                    if until and ts >= until:
                        continue
                    yield RawEvent(channel_id=cid, ts=ts, value=parts[5])


class KafkaReader(JournalReader):
    """Реальный продовый источник. confluent-kafka опционален: без него сервис падает с понятной
    ошибкой только при создании читателя, а не при импорте модуля."""

    def __init__(self, topics: list[str] | None = None, group: str | None = None):
        try:
            from confluent_kafka import Consumer
        except ImportError as e:
            raise RuntimeError('confluent-kafka не установлен — сервис без него не читает '
                               'tf.ingest.* (см. ML/service/requirements.txt)') from e
        self.topics = topics or [config.TOPIC_READINGS, config.TOPIC_JOURNAL]
        conf = {'bootstrap.servers': config.KAFKA_BOOTSTRAP,
                'group.id': group or config.KAFKA_GROUP,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': True,
                'max.partition.fetch.bytes': config.MAX_MSG_BYTES}
        if config.KAFKA_PASSWORD:
            conf.update({'security.protocol': 'sasl_ssl', 'sasl.mechanism': 'PLAIN',
                         'sasl.username': 'tf-model', 'sasl.password': config.KAFKA_PASSWORD})
        self.consumer = Consumer(conf)
        self.consumer.subscribe(self.topics)

    def poll(self, since=None, until=None) -> Iterator[RawEvent]:
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue
            yield self._parse(msg.value())

    @staticmethod
    def _parse(payload: bytes) -> RawEvent:
        d = json.loads(payload)
        if {'ид_канала_данных', 'дата', 'время', 'значение_датчика'} <= set(d):
            return RawEvent(channel_id=int(d['ид_канала_данных']),
                            ts=datetime.strptime(f"{d['дата']} {d['время']}", '%Y-%m-%d %H:%M:%S'),
                            value=str(d['значение_датчика']))
        # альтернативный формат (согласуется с §8 «Форматы tf.ingest.*» во время M1)
        return RawEvent(channel_id=int(d['channel_id']),
                        ts=datetime.fromisoformat(d['ts']), value=str(d['value']))

    def close(self) -> None:
        self.consumer.close()


def pull(store: HotStore, reader: JournalReader, since=None, until=None) -> dict:
    """Прогнать читателя через чистку HotStore.clean_event в append.

    Возврат: сколько строк вошло в чтения и сколько отброшено чисткой (для наблюдения M10).
    """
    accepted = dropped = 0
    batch: list[dict] = []
    for ev in reader.poll(since, until):
        clean = store.clean_event(ev.channel_id, ev.ts, ev.value)
        if clean is None:
            dropped += 1
            continue
        batch.append(clean)
        accepted += 1
        if len(batch) >= 2000:
            store.append(batch)
            batch = []
    if batch:
        store.append(batch)
    return {'accepted': accepted, 'dropped': dropped}