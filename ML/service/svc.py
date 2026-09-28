"""Конфигурация сервиса tf-model (INTEGRATION §4, §8).

Сервис читает и пишет только в ML/work (по умолчанию) — git его не видит. Пути, периоды и рабочие
настройки берутся из pipeline/config.py и pipeline/operating (settings/operating.json), чтобы
исследование и прод не разъехались: у сервиса нет своих копий pipeline-модулей и своего формата
настроек (INTEGRATION2 §10.1).
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parent
ML = SERVICE.parent
sys.path.insert(0, str(ML / 'pipeline'))
# общий модуль контура (Vault, токены, аудит): в репозитории — docs/backend/tfkit, в образе — рядом
for _kit in (os.environ.get('TF_KIT'), ML.parent / 'docs' / 'backend' / 'tfkit', SERVICE / 'tfkit'):
    if _kit and (Path(_kit) / 'tfkit.py').exists():
        sys.path.insert(0, str(_kit))
        break

import config as pipe  # noqa: E402  pipeline/config.py: TF_JOURNAL, TF_WORK, YEARS, TYPES, HORIZON

for _name in ('DICT', 'JOURNAL', 'YEARS', 'GAPS', 'WARMUP', 'DATA_END', 'TRAIN_END',
              'TYPES', 'HORIZON', 'SETTINGS'):
    globals()[_name] = getattr(pipe, _name)

WORK = pipe.WORK
HOT_DB = WORK / 'service' / 'hot.duckdb'
OUT_DIR = WORK / 'service' / 'out'
# 13.6: выгрузка для контейнера — папка только для чтения (TF_MODEL_BUNDLE); на стенде — work/export
BUNDLE = Path(os.environ['TF_MODEL_BUNDLE']) if os.environ.get('TF_MODEL_BUNDLE') else None
EXPORT = BUNDLE or WORK / 'export'
MANIFEST = EXPORT / 'manifest.json'
FEATURES = WORK / 'features'
STATE_DIR = OUT_DIR
HISTORY = OUT_DIR / 'history.parquet'
DECISIONS_LOG = OUT_DIR / 'decisions.parquet'
RETRAIN_LOG = OUT_DIR / 'retrain.json'
OBS_LOG = OUT_DIR / 'observe.json'
RULES_STATE = OUT_DIR / 'rules.json'                   # отклонения, молчания, склейка — до ack (13.3)
SERVICE_STATE = OUT_DIR / 'state.json'                 # версии по типам, флаг переобучения, окна M7
# Таблицы главного диспетчера (13.3): рабочие настройки, график работ, игнорируемые периоды — на томе
# сервиса с версиями; первая версия — из выгрузки (settings/ рядом с моделями) или из ML/settings
SETTINGS_DIR = Path(os.environ.get('TF_MODEL_SETTINGS', WORK / 'service' / 'settings'))
SEED_SETTINGS = BUNDLE / 'settings' if BUNDLE and (BUNDLE / 'settings').exists() else ML / 'settings'
# §9.4: версии, которые сервис включает сам, пока главный диспетчер не выбрал другую (model.switch);
# отказ оборудования — v1 «тихая» (cat×5 0.75 + tcn×3 0.25), остальные типы — основная выгрузка
DEFAULT_VERSIONS = {'equipment': 1}

# §8: дата и время в журнале без зоны, признаки завязаны на местное время — зону фиксируем явно.
# Н3: в контейнере часы в UTC, поэтому «сейчас» считается от MSK, а не от зоны процесса. Москва без
# перехода на летнее время с 2014 года, фиксированного сдвига достаточно, tzdata не нужна.
TZ = '+03:00'
MSK = timezone(timedelta(hours=3), 'MSK')


def now_msk() -> datetime:
    """Местное время журнала без зоны — в том же виде, что даты в ext-journal-*.csv."""
    return datetime.now(MSK).replace(tzinfo=None)


ENV = os.environ.get('TF_ENV', 'prod')                 # dev: стенд без Vault (INTEGRATION §13.1)
# Н8: live — такт по настоящему времени; replay:<начало>:<скорость> — проигрыш тестового года,
# скорость — модельных секунд в реальную (3600 — модельный час за секунду)
CLOCK = os.environ.get('TF_MODEL_CLOCK', 'live')
CLOCK_STATE = OUT_DIR / 'clock.json'                   # Н1: последний посчитанный час
LAST_RESULTS = OUT_DIR / 'last.json'                   # /forecast: последнее сообщение по объекту
COMMANDS_SEEN = OUT_DIR / 'commands.json'              # 13.3: command_id уже применённых команд
AUDIT_SPOOL = OUT_DIR / 'audit.jsonl'                  # 13.4: события, пока Redis недоступен

# M2: глубина горячего журнала — самое длинное окно признаков 90 суток плюс запас (INTEGRATION §1.2)
HOT_RETENTION_DAYS = 100
GUARD_STYPE = 'Состояние охраны'     # строки охраны не удаляются из горячего журнала никогда

# Kafka (think-infra/kafka): SASL_PLAINTEXT внутри think-fast-net, учётка tf-model, группа — с
# префикса tf-model (acls.conf); пароль — из Vault secret/tf/kafka/model (INTEGRATION §13.5)
KAFKA_BOOTSTRAP = os.environ.get('TF_KAFKA_BOOTSTRAP', 'tf-kafka:9092')
KAFKA_USER = 'tf-model'
KAFKA_GROUP = 'tf-model-ingest'
TOPIC_READINGS = 'tf.ingest.readings'
TOPIC_JOURNAL = 'tf.ingest.journal'
TOPIC_REFERENCE = 'tf.ingest.reference'
TOPIC_RESULTS = 'tf.forecast.results'
TOPIC_DLQ = 'tf.dlq'
MAX_MSG_BYTES = 1_048_576           # 1 МБ по ТЗ

# RabbitMQ (think-infra/rabbitmq): модель только читает tf.model.commands (13.3), прав на
# объявление нет — очередь проверяется пассивно; пароль — из Vault secret/tf/rabbit/model
RABBIT_URL = os.environ.get('TF_RABBIT_URL', 'amqp://tf-rabbit:5672/tf')
RABBIT_USER = 'tf-model'
QUEUE_COMMANDS = 'tf.model.commands'
COMMAND_RETRIES = 5                 # после пятой доставки — reject, сообщение уходит в tf.dlq

REDIS_URL = os.environ.get('TF_REDIS_URL', 'redis://tf-redis:6379/0')     # поток аудита
AUTH_JWKS = os.environ.get('TF_AUTH_JWKS', 'http://tf-auth:8080/.well-known/jwks')
# Браузер (админ-панель) несёт токен пользователя в HttpOnly-куке think-auth, а не в заголовке — как у воронки.
AUTH_COOKIE = os.environ.get('TF_AUTH_COOKIE', 'access_token')


def kafka_conf(**extra) -> dict:
    """Подключение к Kafka для читателя и писателя: один конфиг на обоих."""
    import tfkit
    conf = {'bootstrap.servers': KAFKA_BOOTSTRAP}
    password = tfkit.secret('kafka/model', 'TF_KAFKA_MODEL_PASSWORD', 'TF_KAFKA_MODEL_PASSWORD',
                            required=ENV != 'dev')
    if password:
        conf.update({'security.protocol': 'SASL_PLAINTEXT', 'sasl.mechanism': 'PLAIN',
                     'sasl.username': KAFKA_USER, 'sasl.password': password})
    conf.update(extra)
    return conf

# §7: опоздание потока — не повод не считать; за сколько часов данные считаем «несвежими».
# Охрана (дверь, движение, люк) по всему парку ночью молчит до 13 ч (журнал 20.09.2025–08.01.2026):
# при 1 ч флаг горел в 24 % часов, при 12 ч — в 0,05 % (аналитика, раздел 64).
STALE_HOURS = {'default': 1, 'flood': 3, 'intrusion': 12}
# §7, раздел 55: молчание семейства датчиков объекта снижает `confidence` своего типа на долю поимок,
# которую это молчание стоит на проверке 2025 (дым: пожар −18 из 444, отказ датчика −27 из 403;
# газ: загазованность −18 из 72; температура у отказа датчика потерь не даёт — семейство только
# помечается). Остальные семейства в сообщении помечаются, уверенность не трогают.
SILENCE_COST = {'fire': {'smoke': 0.04}, 'gas': {'gas': 0.25}, 'sensor': {'smoke': 0.07, 'temp': 0.0}}
# §9.7: свидетели тревоги — последнее событие каналов семейства типа за это окно, не больше N штук
EVIDENCE_HOURS, EVIDENCE_MAX = 24, 5

# M5: окно скользящего порога — те же 90 суток, что и retrop.rolling()/calib.py
THRESHOLD_WINDOW_DAYS = 90
# M6: склейка дребезга (П6) — как chatter.py / reject.GAP, разрыв короче не начинает новую тревогу
CHATTER_GAP_HOURS = 6
# M6: отклонение диспетчера после REJECT — молчание на N часов или до конца серии (ISA 18.2 «clear»)
REJECT_N_HOURS = 168

# Н4: токены — RS256 от think-auth (13.2), проверка в tfkit.Verifier; своих секретов у ручек нет
API_HOST = os.environ.get('TF_MODEL_API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('TF_MODEL_API_PORT', '8000'))