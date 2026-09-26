"""HTTP-ручки сервиса (INTEGRATION §13.1, токены — §13.2).

Каждая ручка доступна и с префиксом `/api/ml`, и без него (Н5). Токен — RS256 от think-auth,
проверка в `tfkit.Verifier`: подпись, `exp`, `aud = api`, издатель, тип `access`. Своих секретов у
ручек нет, токены нигде не хранятся.

| ручка | кто | право |
|---|---|---|
| `/health` | nginx, Docker | без токена |
| `/status`, `/estimate` | админ-панель через BFF или напрямую | техучётка с `ml.read` либо токен пользователя |
| `/forecast`, `/history` | BFF | только техучётка с `ml.read` |

Права пользователя сервис не знает (права-и-аудит §1): их проверяет BFF у себя и ходит сюда своей
техучёткой. Пользовательский токен принимается только на `/status` и `/estimate`; на остальных —
403 и `access.denied`. Неверный токен — 401 и `token.refused`; протухший — 401 без события (§6.2:
фронт обновляет токен каждые 10 минут, это не событие). В событии из токена только `jti`.
При `TF_ENV=dev` без ключа и без TF_AUTH_JWKS ручки открыты (стенд); в `prod` ключ, которого нет в
Vault, берётся с JWKS think-auth, а пока его не получить — 503.
"""
import logging
import os
import threading
from datetime import timedelta

import svc as config

log = logging.getLogger('tf-model')
SCOPE = 'ml.read'


def make_verifier():
    """Ключ — из TF_AUTH_PUBLIC_KEY (открытый, не секрет; в Vault think-infra его нет), иначе с JWKS
    think-auth; техучётки без `scope` — по списку `sub` из Vault (secret/tf/app/tf-model
    TF_MODEL_SERVICE_SUBS), пока think-auth не кладёт `scope`."""
    import tfkit
    pem = os.environ.get('TF_AUTH_PUBLIC_KEY') or None
    subs = tfkit.secret('app/tf-model', 'TF_MODEL_SERVICE_SUBS', 'TF_MODEL_SERVICE_SUBS', required=False) or ''
    if pem is None and config.ENV == 'dev' and not os.environ.get('TF_AUTH_JWKS'):
        log.warning('dev: ключа проверки токенов нет — ручки открыты')
        return None
    return tfkit.Verifier(public_key=pem, jwks_url=None if pem else config.AUTH_JWKS,
                          service_subs=[s.strip() for s in subs.split(',') if s.strip()])


def create_app(service, verifier=None):
    from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request
    import tfkit
    from core import to_msk

    def refuse(event: str, status: int, reason: str, request: Request, claims: dict | None = None, jti=None):
        """401 — токена нет или он не прошёл проверку (исполнитель неизвестен); 403 — токен верный,
        права нет (исполнитель — его `sub`). Из токена в событие идёт только `jti`."""
        claims = claims or {}
        try:
            service.audit.event(event, 'denied',
                                actor_kind=verifier.kind(claims) if claims else 'anonymous',
                                actor_id=claims.get('sub'), request_id=request.headers.get('x-request-id'),
                                ip=request.client.host if request.client else None,
                                object_type='route', object_id=request.url.path,
                                details={'reason': reason, 'jti': claims.get('jti', jti)})
        except Exception:
            log.exception('аудит %s не записан', event)
        raise HTTPException(status, reason)

    def guard(users: bool):
        def check(request: Request, authorization: str | None = Header(None)) -> dict:
            if verifier is None:
                return {'sub': 'dev'}
            if not authorization or not authorization.lower().startswith('bearer '):
                refuse('token.refused', 401, 'нет токена', request)
            try:
                claims = verifier.verify(authorization.split(None, 1)[1].strip())
            except tfkit.TokenError as e:
                if not e.audit:               # ключа нет (наш отказ) или токен протух (§6.2): без аудита
                    raise HTTPException(e.status, e.reason) from None
                refuse('token.refused', e.status, e.reason, request, jti=e.jti)
            if verifier.kind(claims) == 'service':
                if not verifier.has_scope(claims, SCOPE):
                    refuse('access.denied', 403, f'нужно право {SCOPE}', request, claims)
            elif not users:
                refuse('access.denied', 403, 'ручка только для техучётки', request, claims)
            return claims
        return check

    anyone, services = guard(users=True), guard(users=False)
    r = APIRouter()

    @r.get('/health')
    def health():
        return service.health()

    @r.get('/status')
    def status(_=Depends(anyone)):
        return service.status()

    @r.get('/estimate')
    def estimate(type: str = Query(...), share: float = Query(...), _=Depends(anyone)):
        try:
            return service.estimate(type, share)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @r.get('/forecast')
    def forecast(object_id: int = Query(...), _=Depends(services)):
        msg = service.forecast(object_id)
        if msg is None:
            raise HTTPException(404, f'по объекту {object_id} расчёта ещё нет')
        return msg

    @r.get('/history')
    def history(object_id: int = Query(...), type: str = Query(...), from_: str | None = Query(None, alias='from'),
                to: str | None = Query(None), _=Depends(services)):
        if type not in config.TYPES:
            raise HTTPException(400, f'тип {type!r} не из {config.TYPES}')
        try:
            b = to_msk(to) if to else (service.last_now or config.now_msk())
            a = to_msk(from_) if from_ else b - timedelta(days=7)
        except ValueError:
            raise HTTPException(400, 'from и to — время ISO 8601') from None
        if not a < b or b - a > timedelta(days=config.HOT_RETENTION_DAYS):
            raise HTTPException(400, f'период — от from до to, не длиннее {config.HOT_RETENTION_DAYS} суток')
        return {'object_id': object_id, 'type': type, 'hours': service.forecast_history(object_id, type, a, b)}

    app = FastAPI(title='tf-model', docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(r)
    app.include_router(r, prefix='/api/ml')
    if isinstance(getattr(service, 'audit', None), tfkit.Audit):
        tfkit.request_log(app, service.audit, verifier)          # строка на запрос (права-и-аудит §6.1)
    return app


def start_api(service, verifier=None, host: str = config.API_HOST, port: int = config.API_PORT):
    """uvicorn в потоке процесса сервиса (13.1: четвёртый поток)."""
    import uvicorn
    app = create_app(service, verifier)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='warning'))
    t = threading.Thread(target=server.run, name='api', daemon=True)
    t.start()
    return server
