"""Настройки сервиса: переменные окружения или файл .env рядом с кодом.

В контуре (TF_ENV=prod) секреты — только из Vault: `secret/tf/notify` (smtp_user, smtp_password,
telegram_bot_token, service_subs), `secret/tf/rabbit` (email_password, telegram_password),
`secret/tf/auth` (public_key). Значения секретов из окружения и .env берутся только при TF_ENV=dev
(ML/INTEGRATION.md §13.5).
"""
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(__file__).resolve().parent
for _kit in (os.environ.get('TF_KIT'), HERE.parent / 'tfkit', HERE / 'tfkit'):
    if _kit and (Path(_kit) / 'tfkit.py').exists():
        sys.path.insert(0, str(_kit))
        break
import tfkit  # noqa: E402

log = logging.getLogger('notify')
# TF_NOTIFY_ENV_FILE='' — без файла .env (тесты, контейнер)
ENV_FILE = os.environ.get('TF_NOTIFY_ENV_FILE', str(HERE / '.env')) or None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding='utf-8', extra='ignore')

    # Почта. По умолчанию — Gmail: порт 587 и STARTTLS
    smtp_host: str = 'smtp.gmail.com'
    smtp_port: int = 587
    smtp_security: Literal['starttls', 'ssl', 'none'] = 'starttls'
    smtp_user: str = ''
    smtp_password: SecretStr = SecretStr('')
    smtp_timeout: float = 15
    mail_from: str = ''  # пусто — письма уходят от SMTP_USER; Gmail всё равно подставит адрес учётки
    mail_from_name: str = 'Think Faster — оповещения'
    max_recipients: int = 50  # Gmail принимает не больше 100 адресатов в одном письме

    # Telegram
    telegram_bot_token: SecretStr = SecretStr('')
    telegram_api_url: str = 'https://api.telegram.org'
    telegram_timeout: float = 15

    # Проверка токена техучётки. Без ключа сервис принимает запросы без токена
    jwt_public_key: Path | None = None  # PEM: публичный ключ или сертификат X.509 сервиса аутентификации
    jwt_scope: str = 'notify.send'
    jwt_issuer: str = ''  # пусто — издатель не проверяется
    jwt_algorithms: list[str] = ['RS256', 'ES256']
    tf_auth_jwks: str = ''  # пусто вне dev — http://tf-auth:8080/.well-known/jwks
    service_subs: str = ''  # dev: sub техучёток через запятую, пока think-auth не кладёт scope (Н10)

    # Контур: dev — ручки без ключа открыты, секреты можно из .env; prod — только Vault и токен
    tf_env: Literal['dev', 'prod'] = 'prod'
    # очереди tf.notify.*: off — не читать; пусто — в dev off, в prod amqp://tf-rabbit:5672/tf
    tf_rabbit_url: str = ''
    notify_channels: str = 'email,telegram'  # какие очереди читает этот экземпляр
    notify_retries: int = 5  # как delivery-limit у очередей в think-infra
    rabbit_password: SecretStr = SecretStr('')  # только dev; в prod — Vault secret/tf/rabbit
    # кому уже ушло по notice_id — сутки в Redis; пусто — в dev в памяти, в prod redis://tf-redis:6379/0
    tf_redis_url: str = ''
    tf_audit_spool: Path | None = None  # файл досылки событий аудита, пока Redis лежит

    @field_validator('smtp_password')
    @classmethod
    def strip_gmail_spaces(cls, value: SecretStr, info: ValidationInfo) -> SecretStr:
        # Google показывает пароль приложения группами через пробел: «abcd efgh ijkl mnop»
        if 'gmail' in info.data.get('smtp_host', ''):
            return SecretStr(value.get_secret_value().replace(' ', ''))
        return value

    @field_validator('jwt_public_key', mode='before')
    @classmethod
    def relative_to_code(cls, value: str | Path | None) -> Path | None:
        if value in (None, ''):
            return None
        path = Path(value)
        return path if path.is_absolute() else HERE / path

    def rabbit(self) -> str | None:
        if self.tf_rabbit_url == 'off' or (not self.tf_rabbit_url and self.tf_env == 'dev'):
            return None
        return self.tf_rabbit_url or 'amqp://tf-rabbit:5672/tf'

    def redis(self) -> str | None:
        if self.tf_redis_url == 'off' or (not self.tf_redis_url and self.tf_env == 'dev'):
            return None
        return self.tf_redis_url or 'redis://tf-redis:6379/0'

    def channels(self) -> list[str]:
        return [c for c in (x.strip() for x in self.notify_channels.split(',')) if c in ('email', 'telegram')]


SECRETS = (('notify', 'smtp_user'), ('notify', 'smtp_password'), ('notify', 'telegram_bot_token'),
           ('notify', 'service_subs'))


def load(**over) -> Settings:
    """Настройки плюс секреты из Vault. Вне dev секрет, которого нет в Vault, пуст: канал выключен, в лог
    — путь и поле, не значение."""
    base = Settings(**over)
    os.environ.setdefault('TF_ENV', base.tf_env)        # TF_ENV из .env виден и tfkit
    vault = {}
    for path, field in SECRETS:
        value = tfkit.secret(path, field, required=False)
        if value:
            vault[field] = value
        elif base.tf_env != 'dev':
            vault[field] = ''
            if field != 'service_subs':
                log.warning('нет secret/tf/%s поля %s — канал без него не работает', path, field)
    return Settings(**{**over, **vault}) if vault else base


def rabbit_password(settings: Settings, channel: str) -> str | None:
    field = f'{channel}_password'
    value = tfkit.secret('rabbit', field, required=False)
    if value:
        return value
    if settings.tf_env == 'dev' and settings.rabbit_password.get_secret_value():
        return settings.rabbit_password.get_secret_value()
    raise tfkit.SecretError(f'нет secret/tf/rabbit поля {field}')


def public_key(settings: Settings) -> bytes | None:
    """Ключ проверки токенов: Vault secret/tf/auth, иначе файл JWT_PUBLIC_KEY (ключ не секрет)."""
    pem = tfkit.secret('auth', 'public_key', required=False)
    if pem:
        return pem.encode()
    return settings.jwt_public_key.read_bytes() if settings.jwt_public_key else None
