"""Потребитель журнала (M1/П3/П9): события Kafka или файловый повтор — в горячий журнал.

Источник событий спрятан за интерфейсом «читателя журнала» (§6); Kafka — одна из реализаций, и
стенд проверяем ReplayReader'ом на тех же ext-journal-*.csv. Схема `clean() → append()` одна для
всех читателей; на парсинге/качке — строки в DLQ, не в основную цепочку.

Прод (П3): приём живёт в отдельном потоке и коммитит смещения САМ, явно, после каждой пачки —
auto.commit выключен, чтобы такт никогда не принёс «уже съеденное» после падения. Та же схема для
решений диспетчера — но они приходят не из Kafka, а из RabbitMQ `tf.model.commands` (commands.py,
INTEGRATION §13.3); отсутствие сообщений в паузе — не ошибка (poll с таймаутом).

Повтор (П9): чтение файлов — polars (ленивый периодический сдвиг), дата — параметр, а не
константа; массовая заливка — COPY-семейство поверх read_csv в HotStore.bulk_import.
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import polars as pl

from storage import HotStore

import svc as config


@dataclass
class RawEvent:
    channel_id: int
    ts: datetime
    value: str


class JournalReader(ABC):
    """Источник событий за интерфейсом (§6): что и как читать — деталь реализации."""

    @abstractmethod
    def poll(self, since: datetime | None, until: datetime | None) -> Iterator[RawEvent]: ...


class ReplayReader(JournalReader):
    """Файловый повтор журнала для стенда (think-test/emulator) и тестов (П9).

    Ленивая schema-инференция polars: кавычки, провалы и большие значения не ломают строку;
    фильтр по датам применяется до материализации, как bulk_import. Использование — побайтовый
    сдвиг не нужен: iterate поверх .iter_rows() идёт поупорядоченно после фильтра.
    """

    def __init__(self, csv_files: list[str], reference=None, speed: float = 0.0):
        self.files = csv_files
        self.reference = reference
        self.speed = speed                       # 0 — сразу, >0 — дробление по часам (не влияет)

    def poll(self, since=None, until=None) -> Iterator[RawEvent]:
        since, until = since or datetime(2019, 1, 1), until or datetime(2100, 1, 1)
        for f in self.files:
            df = pl.scan_csv(f, infer_schema_length=0,
                             dtypes={'дата': pl.Utf8, 'время': pl.Utf8,
                                     'ид_канала_данных': pl.Utf8, 'значение_датчика': pl.Utf8},
                             null_values=['', 'nan']).collect()
            rows = df.iter_rows(named=True)
            for d in rows:
                ts_s = f"{d['дата']} {d['время']}"
                try:
                    ts = datetime.strptime(ts_s, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
                if ts < since or ts >= until:
                    continue
                try:
                    cid = int(d['ид_канала_данных'])
                except (TypeError, ValueError):
                    continue
                yield RawEvent(channel_id=cid, ts=ts, value=str(d['значение_датчика']))


class KafkaReader(JournalReader):
    """Продовый источник. confluent-kafka опционален: падение только на создании читателя.

    Один инстанс = один group.id: приём журнала и решения диспетчера никогда не мешают друг другу
    (П4). Коммит явный: `commit()` вызывается извне после обработки пачки в pull().
    """

    def __init__(self, topics: list[str] | None = None, group: str | None = None):
        try:
            from confluent_kafka import Consumer
        except ImportError as e:
            raise RuntimeError('confluent-kafka не установлен — сервис без него не читает '
                               'tf.ingest.* (см. ML/service/requirements.txt)') from e
        self.topics = topics or [config.TOPIC_READINGS, config.TOPIC_JOURNAL, config.TOPIC_REFERENCE]
        conf = config.kafka_conf(**{'group.id': group or config.KAFKA_GROUP,
                                    'auto.offset.reset': 'earliest',
                                    'enable.auto.commit': False,     # П3: коммит после пачки
                                    'max.partition.fetch.bytes': config.MAX_MSG_BYTES})
        self.consumer = Consumer(conf)
        self.consumer.subscribe(self.topics)

    def poll_readings(self, timeout_ms: float = 0.05, commit: bool = False):
        """Следующее сообщение или None (нет данных за таймаут — не ошибка, П4)."""
        msg = self.consumer.poll(timeout_ms)
        if msg is None or msg.error():
            return None
        if commit:
            self.commit()
        return msg

    def poll(self, since=None, until=None) -> Iterator[RawEvent]:
        while True:
            msg = self.poll_readings()
            if msg is None:
                return
            yield self._parse(msg.value())

    @staticmethod
    def _parse(payload: bytes) -> RawEvent:
        d = json.loads(payload)
        if {'ид_канала_данных', 'дата', 'время', 'значение_датчика'} <= set(d):
            return RawEvent(channel_id=int(d['ид_канала_данных']),
                            ts=datetime.strptime(f"{d['дата']} {d['время']}", '%Y-%m-%d %H:%M:%S'),
                            value=str(d['значение_датчика']))
        return RawEvent(channel_id=int(d['channel_id']),
                        ts=datetime.fromisoformat(d['ts']), value=str(d['value']))

    def commit(self) -> None:
        self.consumer.commit(asynchronous=False)

    def close(self) -> None:
        self.consumer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def consume(store, reader: KafkaReader, stop, dead=None, batch_size: int = 2000, idle: float = 1.0,
            stats: dict | None = None) -> dict:
    """Прод (поток приёма в `--loop`): читать tf.ingest.* до `stop`, а не до первой паузы.

    Пачка пишется в горячий журнал на 2000 строк или на тишине в `idle` секунд, и только после
    записи коммитится смещение (П3). Сообщение, которое не разобралось, уходит в `dead` (tf.dlq) и
    тоже коммитится: повторять его бессмысленно. Строка, отброшенная чисткой (чужой канал, пустое
    значение), — только в счётчик `dropped` для наблюдения M10. `written_at` — когда последняя пачка
    легла в журнал; счётчики отдаёт `/status` (поле `ingest`).

    `tf.ingest.reference` (Н26) применяется сразу, мимо пачки: молчание каналов от воронки и правки
    справочника (`store.apply_reference`); счётчик — `reference`.
    """
    stats = stats if stats is not None else {}
    for k in ('accepted', 'dropped', 'dead', 'reference'):
        stats.setdefault(k, 0)
    batch: list[dict] = []
    pending = 0
    while not stop.is_set():
        msg = reader.consumer.poll(idle)
        if msg is not None and msg.error() is None:
            pending += 1
            try:
                if msg.topic() == config.TOPIC_REFERENCE:
                    store.apply_reference(json.loads(msg.value()))
                    stats['reference'] += 1
                    clean = False
                else:
                    ev = reader._parse(msg.value())
                    clean = store.clean_event(ev.channel_id, ev.ts, ev.value)
            except (ValueError, KeyError, TypeError) as e:     # JSONDecodeError и UnicodeDecodeError — тоже ValueError
                stats['dead'] += 1
                if dead is not None:
                    dead(msg.value(), msg.key(), f'{type(e).__name__}: {e}', msg.topic())
                clean = None
            else:
                if clean is None:
                    stats['dropped'] += 1
            if clean:
                batch.append(clean)
        if pending and (len(batch) >= batch_size or msg is None):
            if batch:
                store.append(batch)
                stats['accepted'] += len(batch)
                stats['written_at'] = config.now_msk().isoformat(timespec='seconds')
            reader.commit()
            batch, pending = [], 0
    if batch:
        store.append(batch)
        stats['accepted'] += len(batch)
        stats['written_at'] = config.now_msk().isoformat(timespec='seconds')
        reader.commit()
    return stats


def pull(store: HotStore, reader: JournalReader, on_batch=None, since=None, until=None) -> dict:
    """Прогнать читателя через чистку HotStore.clean_event в append.

    Возврат: сколько строк вошло в чтения и сколько отброшено чисткой (наблюдение M10). Для
    Kafka-читателя после каждой пачки — явный commit (П3), настройки с пачкой — on_batch.
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
            if isinstance(reader, KafkaReader):
                reader.commit()
            if on_batch:
                on_batch()
            batch = []
    if batch:
        store.append(batch)
        if isinstance(reader, KafkaReader):
            reader.commit()
    if on_batch:
        on_batch()
    return {'accepted': accepted, 'dropped': dropped}