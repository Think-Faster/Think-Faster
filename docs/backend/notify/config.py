"""Настройки сервиса: переменные окружения или файл .env рядом с кодом."""
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

HERE = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=HERE / '.env', env_file_encoding='utf-8', extra='ignore')

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
