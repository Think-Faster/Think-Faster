"""Конфигурация сервиса tf-model (INTEGRATION §4, §8).

Сервис читает и пишет только в ML/work (по умолчанию) — git его не видит. Пути, периоды и рабочие
настройки берутся из pipeline/config.py и pipeline/operating (settings/operating.json), чтобы
исследование и прод не разъехались: у сервиса нет своих копий pipeline-модулей и своего формата
настроек (INTEGRATION2 §10.1).
"""
import os
import sys
from pathlib import Path

SERVICE = Path(__file__).resolve().parent
ML = SERVICE.parent
sys.path.insert(0, str(ML / 'pipeline'))

import config as pipe  # noqa: E402  pipeline/config.py: TF_JOURNAL, TF_WORK, YEARS, TYPES, HORIZON

for _name in ('DICT', 'JOURNAL', 'YEARS', 'GAPS', 'WARMUP', 'DATA_END', 'TRAIN_END',
              'TYPES', 'HORIZON', 'SETTINGS'):
    globals()[_name] = getattr(pipe, _name)

WORK = pipe.WORK
HOT_DB = WORK / 'service' / 'hot.duckdb'
OUT_DIR = WORK / 'service' / 'out'
EXPORT = WORK / 'export'
MANIFEST = EXPORT / 'manifest.json'
FEATURES = WORK / 'features'
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
KAFKA_GROUP_DECISIONS = 'tf-model-decisions'   # П4: свой group у решений, не смешивать с приёмом
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

# M5: окно скользящего порога — те же 90 суток, что и retrop.rolling()/calib.py
THRESHOLD_WINDOW_DAYS = 90
# M6: склейка дребезга (П6) — как chatter.py / reject.GAP, разрыв короче не начинает новую тревогу
CHATTER_GAP_HOURS = 6
# M6: отклонение диспетчера после REJECT — молчание на N часов или до конца серии (ISA 18.2 «clear»)
REJECT_N_HOURS = 168

# П8: JWT на всех ручках; секрет и токены сервисов — из env, в коде только умолчания для стенда
TOKEN_SECRET = os.environ.get('TF_MODEL_TOKEN_SECRET', 'dev-secret')
TOKEN_ALGO = 'HS256'
TOKEN_TTL_HOURS = 12
API_HOST = os.environ.get('TF_MODEL_API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('TF_MODEL_API_PORT', '8000'))