"""Конфигурация сервиса tf-model (INTEGRATION §4, §8).

Сервис читает и пишет только в ML/work (по умолчанию) — git его не видит. Пути и периоды событий
берутся из pipeline/config.py, чтобы исследование и прод не разъехались.
"""
import os
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parent
ML = SERVICE.parent
sys.path.insert(0, str(ML / 'pipeline'))

import config as pipe  # noqa: E402  pipeline/config.py: TF_JOURNAL, TF_WORK, YEARS, TYPES, HORIZON

for _name in ('DICT', 'JOURNAL', 'YEARS', 'GAPS', 'WARMUP', 'DATA_END', 'TRAIN_END'):
    globals()[_name] = getattr(pipe, _name)

WORK = pipe.WORK
HOT_DB = WORK / 'service' / 'hot.duckdb'
OUT_DIR = WORK / 'service' / 'out'
EXPORT = WORK / 'export'
MANIFEST = EXPORT / 'manifest.json'
SETTINGS = ML / 'settings' / 'operating.json'
SETTINGS_SCHEMA = ML / 'settings' / 'operating.schema.json'
STATE_DIR = OUT_DIR
HISTORY = OUT_DIR / 'history.parquet'
DECISIONS_LOG = OUT_DIR / 'decisions.parquet'
RETRAIN_LOG = OUT_DIR / 'retrain.json'
OBS_LOG = OUT_DIR / 'observe.json'

# §8: дата и время в журнале без зоны, признаки завязаны на местное время — зону фиксируем явно
TZ = '+03:00'

# M2: глубина горячего журнала — самое длинное окно признаков 90 суток плюс запас (INTEGRATION §1.2)
HOT_RETENTION_DAYS = 100
GUARD_STYPE = 'Состояние охраны'     # строки охраны не удаляются из горячего журнала никогда

# Kafka (INTEGRATION §4): топики заданы в think-infra; пароль — из TF_KAFKA_MODEL_PASSWORD
KAFKA_BOOTSTRAP = os.environ.get('TF_KAFKA_BOOTSTRAP', 'localhost:9092')
KAFKA_PASSWORD = os.environ.get('TF_KAFKA_MODEL_PASSWORD', '')
KAFKA_GROUP = 'tf-model-ingest'
TOPIC_READINGS = 'tf.ingest.readings'
TOPIC_JOURNAL = 'tf.ingest.journal'
TOPIC_REFERENCE = 'tf.ingest.reference'
TOPIC_RESULTS = 'tf.forecast.results'
TOPIC_DLQ = 'tf.dlq'
TOPIC_DECISIONS = 'tf.dispatch.decisions'
TOPIC_SETTINGS = 'tf.dispatch.settings'
MAX_MSG_BYTES = 1_048_576           # 1 МБ по ТЗ

# §7: опоздание потока — не повод не считать; за сколько часов данные считаем «несвежими»
STALE_HOURS = {'default': 1, 'flood': 3}

TYPES = pipe.TYPES
HORIZON = pipe.HORIZON