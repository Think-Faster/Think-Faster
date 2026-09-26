r"""Сервис уведомлений о происшествиях: письмо через SMTP и сообщение от бота Telegram.

Запуск из этой папки, первый раз:

    python -m venv venv
    venv\Scripts\activate              (Linux и macOS: source venv/bin/activate)
    pip install -r requirements.txt
    uvicorn main:app --port 8010

В следующие разы — только activate и uvicorn. Описание запросов с кнопкой
«Try it out» открывается на http://127.0.0.1:8010/docs.

Настройки — в файле .env рядом с этим файлом, git его не видит:

    TF_ENV=dev
    SMTP_USER=адрес@gmail.com
    SMTP_PASSWORD=пароль приложения Google, 16 букв
    TELEGRAM_BOT_TOKEN=токен от @BotFather
    JWT_PUBLIC_KEY=публичный ключ или сертификат сервиса аутентификации, PEM

Без ключа проверки токенов сервис принимает запросы без токена только при TF_ENV=dev — так удобно
проверять руками. Иначе нужен токен техучётки со scope notify.send в заголовке Authorization: Bearer;
ключ — из Vault secret/tf/auth, файла JWT_PUBLIC_KEY или с JWKS аутентификации. Остальные
настройки — в config.py.

В контуре (ML/INTEGRATION.md §13.7) сервис ещё читает очереди RabbitMQ tf.notify.email и
tf.notify.telegram (rabbit.py), секреты берёт только из Vault, а отправки и отказы пишет в журнал
аудита: notify.sent, notify.failed, token.refused, access.denied — без текста и адресов.
"""
import logging
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator

import config
import mailer
import rabbit
import telegram_bot
import tfkit

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)  # httpx пишет в лог адрес запроса, а в нём токен бота
logging.getLogger('pika').setLevel(logging.WARNING)
log = logging.getLogger('notify')


def make_verifier(settings: config.Settings) -> tfkit.Verifier | None:
    """Ключ — из Vault или файла, иначе с JWKS аутентификации. Без ключа ручки открыты только в dev.
    Поля токена принимаются в обоих вариантах, техучётка без scope — по списку service_subs (§13.2)."""
    pem = config.public_key(settings)
    jwks = settings.tf_auth_jwks or None
    if pem is None and jwks is None:
        if settings.tf_env == 'dev':
            log.warning('dev: ключа проверки токенов нет — ручки открыты без токена')
            return None
        jwks = 'http://tf-auth:8080/.well-known/jwks'
    return tfkit.Verifier(public_key=pem, jwks_url=None if pem else jwks,
                          issuers=(settings.jwt_issuer,) if settings.jwt_issuer else ('auth-service', 'tf-auth'),
                          service_subs=[s.strip() for s in settings.service_subs.split(',') if s.strip()],
                          algorithms=settings.jwt_algorithms)


settings = config.load()
verifier = make_verifier(settings)
audit = tfkit.Audit('notify', redis_url=settings.redis() or '', spool=settings.tf_audit_spool)
consumers: list[threading.Thread] = []


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = threading.Event()
    consumers[:] = rabbit.start(settings, audit, stop)
    yield
    stop.set()


app = FastAPI(title='Think Faster — уведомления', version='0.2.0', lifespan=lifespan)
tfkit.request_log(app, audit, verifier)                  # строка на запрос (права-и-аудит §6.1)
bearer = HTTPBearer(auto_error=False)


class Notice(BaseModel):
    subject: str = Field(min_length=1, max_length=255, examples=['Тревога: газовый датчик, ДУ объект Альфа'])
    text: str = Field(min_length=1, max_length=20000, examples=['Канал 196771 «Газовая охрана». Заявка 1042.'])
    ticket_id: int | None = Field(default=None, examples=[1042])  # заявка — для журнала аудита

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


def refuse(event: str, status: int, reason: str, request: Request, claims: dict | None = None, jti=None):
    """Отказ в журнал и ответ. 401 — исполнитель неизвестен, 403 — его `sub`. Из токена — только `jti`."""
    claims = claims or {}
    try:
        audit.event(event, 'denied', actor_kind=verifier.kind(claims) if claims else 'anonymous',
                    actor_id=claims.get('sub'), request_id=request.headers.get('x-request-id'),
                    ip=request.client.host if request.client else None, object_type='route',
                    object_id=request.url.path, details={'reason': reason, 'jti': claims.get('jti', jti)})
    except Exception:
        log.exception('аудит %s не записан', event)
    raise HTTPException(status, reason, headers={'WWW-Authenticate': 'Bearer'} if status == 401 else None)


def check_token(request: Request,
                credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict | None:
    if verifier is None:
        return None
    if credentials is None:
        refuse('token.refused', 401, 'нужен токен техучётки: Authorization: Bearer <токен>', request)
    try:
        claims = verifier.verify(credentials.credentials)
    except tfkit.TokenError as e:
        if not e.audit:           # токен протух или ключа нет у нас самих — не событие вызывающего
            raise HTTPException(e.status, e.reason,
                                headers={'WWW-Authenticate': 'Bearer'} if e.status == 401 else None) from None
        refuse('token.refused', e.status, e.reason, request, jti=e.jti)
    if not verifier.has_scope(claims, settings.jwt_scope):
        refuse('access.denied', 403, f'у учётки нет права {settings.jwt_scope}', request, claims)
    return claims


def journal(event: str, channel: str, notice: Notice, request: Request, caller: dict | None, **details):
    """notify.sent / notify.failed: канал, число адресатов, тема, заявка, причина. Текста и адресов нет."""
    try:
        audit.event(event, 'success' if event == 'notify.sent' else 'error',
                    actor_kind=verifier.kind(caller) if caller else 'anonymous',
                    actor_id=(caller or {}).get('sub'), request_id=request.headers.get('x-request-id'),
                    object_type='ticket' if notice.ticket_id is not None else None, object_id=notice.ticket_id,
                    details={'channel': channel, 'via': 'http', 'subject': notice.subject,
                             'recipients': len(notice.to), **details})
    except Exception:
        log.exception('аудит %s не записан', event)


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
        'env': settings.tf_env,
        'mail': bool(settings.mail_from or settings.smtp_user),
        'telegram': bool(settings.telegram_bot_token.get_secret_value()),
        'token_check': verifier is not None,
        'queues': [t.name.removeprefix('rabbit-') for t in consumers if t.is_alive()],
    }


@app.post('/mail', response_model=Result, responses={422: {'model': Failure}, 502: {'model': Failure},
                                                      503: {'model': Failure}})
def send_mail(notice: MailNotice, request: Request, caller: dict | None = Depends(check_token)):
    """Одно письмо всем адресатам. `sent` — кого принял почтовый сервер."""
    to = recipients([str(address) for address in notice.to])
    if not (settings.mail_from or settings.smtp_user):
        journal('notify.failed', 'email', notice, request, caller, sent=0, reason='не задан SMTP_USER')
        return failure('mail', mailer.MailError(503, 'не задан SMTP_USER'))
    try:
        refused = mailer.send(settings, to, notice.subject, notice.text)
    except mailer.MailError as e:
        log.warning('почта от %s, тема «%s»: %s', who(caller), notice.subject, e.message)
        journal('notify.failed', 'email', notice, request, caller, sent=0, reason=f'{e.status}')
        return failure('mail', e)
    sent = [address for address in to if address not in refused]
    log.info('почта от %s, тема «%s»: принято %d, отказ %d', who(caller), notice.subject, len(sent), len(refused))
    journal('notify.sent', 'email', notice, request, caller, sent=len(sent), failed=len(refused))
    return Result(channel='mail', sent=sent, failed=refused)


@app.post('/telegram', response_model=Result, responses={422: {'model': Failure}, 502: {'model': Failure},
                                                          503: {'model': Failure}})
def send_telegram(notice: TelegramNotice, request: Request, caller: dict | None = Depends(check_token)):
    """Сообщение от бота в каждый чат: тема жирным, под ней текст."""
    to = recipients(notice.to)
    if len(telegram_bot.render(notice.subject, notice.text)) > telegram_bot.LIMIT:
        raise HTTPException(422, f'тема и текст длиннее {telegram_bot.LIMIT} символов — предел Telegram')
    try:
        sent, failed = telegram_bot.send(settings, to, notice.subject, notice.text)
    except telegram_bot.TelegramError as e:
        log.warning('telegram от %s, тема «%s»: %s', who(caller), notice.subject, e.message)
        journal('notify.failed', 'telegram', notice, request, caller, sent=0, reason=f'{e.status}')
        return failure('telegram', e)
    log.info('telegram от %s, тема «%s»: доставлено %d, отказ %d', who(caller), notice.subject, len(sent), len(failed))
    if not sent:
        journal('notify.failed', 'telegram', notice, request, caller, sent=0, failed=len(failed),
                reason='ни в один чат не доставлено')
        return failure('telegram', telegram_bot.TelegramError(422, 'ни в один чат не доставлено', failed))
    journal('notify.sent', 'telegram', notice, request, caller, sent=len(sent), failed=len(failed))
    return Result(channel='telegram', sent=sent, failed=failed)


@app.get('/telegram/chats')
def telegram_chats(caller: dict | None = Depends(check_token)):
    """Кто писал боту за последние сутки — отсюда берут chat_id получателей."""
    try:
        return telegram_bot.recent_chats(settings)
    except telegram_bot.TelegramError as e:
        return JSONResponse(status_code=e.status, content={'detail': e.message})
