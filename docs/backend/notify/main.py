r"""Сервис уведомлений о происшествиях: письмо через SMTP и сообщение от бота Telegram.

Запуск из этой папки, первый раз:

    python -m venv venv
    venv\Scripts\activate              (Linux и macOS: source venv/bin/activate)
    pip install -r requirements.txt
    uvicorn main:app --port 8010

В следующие разы — только activate и uvicorn. Описание запросов с кнопкой
«Try it out» открывается на http://127.0.0.1:8010/docs.

Настройки — в файле .env рядом с этим файлом, git его не видит:

    SMTP_USER=адрес@gmail.com
    SMTP_PASSWORD=пароль приложения Google, 16 букв
    TELEGRAM_BOT_TOKEN=токен от @BotFather
    JWT_PUBLIC_KEY=публичный ключ или сертификат сервиса аутентификации, PEM

Без JWT_PUBLIC_KEY сервис принимает запросы без токена — так удобно проверять
руками. С ключом нужен токен техучётки со scope notify.send в заголовке
Authorization: Bearer. Остальные настройки — в config.py.
"""
import logging
from pathlib import Path
from typing import Literal

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

import mailer
import telegram_bot
from config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)  # httpx пишет в лог адрес запроса, а в нём токен бота
log = logging.getLogger('notify')


def load_public_key(path: Path):
    pem = path.read_bytes()
    if b'BEGIN CERTIFICATE' in pem:
        return x509.load_pem_x509_certificate(pem).public_key()
    return serialization.load_pem_public_key(pem)


settings = Settings()
public_key = load_public_key(settings.jwt_public_key) if settings.jwt_public_key else None

app = FastAPI(title='Think Faster — уведомления', version='0.1.0')
bearer = HTTPBearer(auto_error=False)


class Notice(BaseModel):
    subject: str = Field(min_length=1, max_length=255, examples=['Тревога: газовый датчик, ДУ объект Альфа'])
    text: str = Field(min_length=1, max_length=20000, examples=['Канал 196771 «Газовая охрана». Заявка 1042.'])

    @field_validator('subject')
    @classmethod
    def one_line(cls, value: str) -> str:
        # перевод строки в теме письма ломает заголовки
        return ' '.join(value.split())


class MailNotice(Notice):
    to: list[EmailStr] = Field(min_length=1, examples=[['dispatcher@example.com']])


class TelegramNotice(Notice):
    # chat_id человека или группы (у групп он отрицательный), либо @имя канала
    to: list[int | str] = Field(min_length=1, examples=[[123456789]])


class Result(BaseModel):
    channel: Literal['mail', 'telegram']
    sent: list[str]
    failed: dict[str, str]


class Failure(Result):
    detail: str


def check_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict | None:
    if public_key is None:
        return None
    if credentials is None:
        raise HTTPException(401, 'нужен токен техучётки: Authorization: Bearer <токен>',
                            headers={'WWW-Authenticate': 'Bearer'})
    try:
        claims = jwt.decode(credentials.credentials, public_key, algorithms=settings.jwt_algorithms,
                            issuer=settings.jwt_issuer or None, options={'require': ['exp', 'sub']})
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, 'токен истёк', headers={'WWW-Authenticate': 'Bearer'}) from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, 'токен не прошёл проверку', headers={'WWW-Authenticate': 'Bearer'}) from None
    if settings.jwt_scope not in str(claims.get('scope', '')).split():
        raise HTTPException(403, f'у учётки нет права {settings.jwt_scope}')
    return claims


def recipients(to: list) -> list:
    unique = list(dict.fromkeys(to))
    if len(unique) > settings.max_recipients:
        raise HTTPException(422, f'адресатов {len(unique)}, больше {settings.max_recipients} за раз нельзя')
    return unique


def who(caller: dict | None) -> str:
    if caller is None:
        return 'без токена'
    return caller.get('login') or caller['sub']


def failure(channel: str, error: mailer.MailError | telegram_bot.TelegramError) -> JSONResponse:
    body = Failure(channel=channel, sent=[], failed=error.failed, detail=error.message)
    return JSONResponse(status_code=error.status, content=body.model_dump())


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'mail': bool(settings.mail_from or settings.smtp_user),
        'telegram': bool(settings.telegram_bot_token.get_secret_value()),
        'token_check': public_key is not None,
    }


@app.post('/mail', response_model=Result, responses={422: {'model': Failure}, 502: {'model': Failure},
                                                      503: {'model': Failure}})
def send_mail(notice: MailNotice, caller: dict | None = Depends(check_token)):
    """Одно письмо всем адресатам. `sent` — кого принял почтовый сервер."""
    to = recipients([str(address) for address in notice.to])
    if not (settings.mail_from or settings.smtp_user):
        return failure('mail', mailer.MailError(503, 'не задан SMTP_USER'))
    try:
        refused = mailer.send(settings, to, notice.subject, notice.text)
    except mailer.MailError as e:
        log.warning('почта от %s, тема «%s»: %s', who(caller), notice.subject, e.message)
        return failure('mail', e)
    sent = [address for address in to if address not in refused]
    log.info('почта от %s, тема «%s»: принято %d, отказ %d', who(caller), notice.subject, len(sent), len(refused))
    return Result(channel='mail', sent=sent, failed=refused)


@app.post('/telegram', response_model=Result, responses={422: {'model': Failure}, 502: {'model': Failure},
                                                          503: {'model': Failure}})
def send_telegram(notice: TelegramNotice, caller: dict | None = Depends(check_token)):
    """Сообщение от бота в каждый чат: тема жирным, под ней текст."""
    to = recipients(notice.to)
    if len(telegram_bot.render(notice.subject, notice.text)) > telegram_bot.LIMIT:
        raise HTTPException(422, f'тема и текст длиннее {telegram_bot.LIMIT} символов — предел Telegram')
    try:
        sent, failed = telegram_bot.send(settings, to, notice.subject, notice.text)
    except telegram_bot.TelegramError as e:
        log.warning('telegram от %s, тема «%s»: %s', who(caller), notice.subject, e.message)
        return failure('telegram', e)
    log.info('telegram от %s, тема «%s»: доставлено %d, отказ %d', who(caller), notice.subject, len(sent), len(failed))
    if not sent:
        return failure('telegram', telegram_bot.TelegramError(422, 'ни в один чат не доставлено', failed))
    return Result(channel='telegram', sent=sent, failed=failed)


@app.get('/telegram/chats')
def telegram_chats(caller: dict | None = Depends(check_token)):
    """Кто писал боту за последние сутки — отсюда берут chat_id получателей."""
    try:
        return telegram_bot.recent_chats(settings)
    except telegram_bot.TelegramError as e:
        return JSONResponse(status_code=e.status, content={'detail': e.message})
