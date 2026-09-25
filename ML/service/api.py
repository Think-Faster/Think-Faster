"""HTTP-ручки сервиса (M11/M12/П8): JWT на ВСЕХ ручках, 401 без токена.

FastAPI: маршрут /api/ml обслуживает nginx по суффиксу (INTEGRATION §4). Модуль импортируется без
fastapi, app = None — стенд работает без него. Токен — Bearer, HMAC-SHA256 с секретом из
окружения TF_MODEL_TOKEN_SECRET (MБ дев-умолчание только для стенда); подпись фиксирует header и
payload, exp — не дальше TTL. Единственный глобальный конёк живых ссылок — State: П7 меняет
predictor/reload сеттинги, и ручки сразу это видят.

П8: ручек без авторизации нет вообще — ни /health, ни /status; nginx/лоад-балансер знает токен
службы (заголовок Authorization), клиенты получают его через think-infra.
"""
import base64
import hashlib
import hmac
import json
import threading
import time

import svc as config
from settings import OperatingSettings

app = None
_HAVE_FASTAPI = False


class ApiError(Exception):
    """Ошибка ручки без зависимости от fastapi (стенд/тесты без него). Ручка переводит в HTTP."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    _HAVE_FASTAPI = True
    app = FastAPI(title='tf-model')
except ImportError:
    pass


# ----- JWT (П8) ---------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _b64d(s: str) -> bytes:
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def sign(data: bytes, secret: str = config.TOKEN_SECRET) -> bytes:
    return hmac.new(secret.encode(), data, hashlib.sha256).digest()


def make_token(payload: dict, secret: str = config.TOKEN_SECRET,
               ttl_hours: int = config.TOKEN_TTL_HOURS) -> str:
    """Выпуск токена: header и payload подписываются вместе (JWS). Для стенда/тестов."""
    header = _b64(json.dumps({'alg': 'HS256', 'typ': 'JWT'}).encode())
    claims = dict(payload)
    claims['exp'] = int(time.time()) + ttl_hours * 3600
    body = _b64(json.dumps(claims, sort_keys=True).encode())
    sig = _b64(sign(f'{header}.{body}'.encode(), secret))
    return f'{header}.{body}.{sig}'


def verify_token(token: str, secret: str = config.TOKEN_SECRET) -> dict | None:
    if isinstance(token, bytes):
        token = token.decode()
    parts = token.split('.')
    if len(parts) != 3:
        return None
    header_s, body_s, sig_s = parts
    try:
        expected = hmac.new(secret.encode(), f'{header_s}.{body_s}'.encode(),
                            hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(sig_s), expected):
            return None
        claims = json.loads(_b64d(body_s))
    except (ValueError, TypeError):
        return None
    if claims.get('exp', 0) < time.time():
        return None
    return claims


def _require_auth(authorization: str | None) -> dict:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(401, 'нет Bearer-токена')
    claims = verify_token(authorization[7:].strip())
    if claims is None:
        raise HTTPException(401, 'токен не прошёл проверку (подпись/exp)')
    return claims


# ----- состояние и привязка -------------------------------------------------
class State:
    """Живые ссылки, которые ручки читают в момент запроса (не копии на bind-момент)."""

    def __init__(self, store=None, predictor=None, history=None, rules=None, settings=None):
        self.store = store
        self.predictor = predictor
        self.history = history
        self.rules = rules
        self.settings = settings
        self._settings_mtime = None
        self.reload_settings()

    def apply_settings(self, _=None):
        """on_batch из потока приёма: подхватить новую операционку, если админ переписал файл."""
        self.reload_settings()

    def reload_settings(self) -> bool:
        p = config.SETTINGS
        if self._settings_mtime is None:
            self._settings_mtime = p.stat().st_mtime_ns if p.exists() else None
            return False
        if p.exists() and (mt := p.stat().st_mtime_ns) != self._settings_mtime:
            self._settings_mtime = mt
            self.settings = OperatingSettings.load()
            return True
        return False


def estimate_view(st: State, type: str, share: float) -> dict:
    """Ядро /api/ml/estimate (M12, §9.3): доля часа под тревогой → тревог в сутки по парку.

    Порог — (1-share)-квантиль истории оценок парка текущей версии, отсюда число объекто-часов
    выше него за сутки истории; та же формула, что calib.py называет «обещано проверкой»
    в переносе порога (transfer). Без fastapi — тестируется на стенде.
    """
    if st.settings is None or st.history is None:
        raise ApiError(503, 'сервис не инициализирован (нет настроек/истории)')
    try:
        st.settings.check(share)
    except Exception as e:
        raise ApiError(422, str(e))
    try:
        import settings as smod
        hh, pp = st.history.history(type)
        return smod.estimate(pp, share, config.THRESHOLD_WINDOW_DAYS)
    except (KeyError, IndexError):
        raise ApiError(422, f'типа {type} нет в истории оценок')


def ensure_app():
    if app is None:
        raise RuntimeError('fastapi не установлен — ручки недоступны, стенд без них')


def _state() -> State:
    if _live is None:
        raise HTTPException(503, 'сервис не инициализирован')
    return _live


_live: State | None = None


def bind(state: State) -> None:
    global _live
    _live = state


if app is not None:

    def require_auth(authorization: str | None = Header(default=None)) -> dict:
        return _require_auth(authorization)

    @app.get('/health')
    def health(auth: dict = Depends(require_auth)):
        st = _state()
        return {'ok': True, 'settings_version': st.settings.version
                if st.settings else None}

    @app.get('/api/ml/estimate')
    def estimate(type: str, share: float, auth: dict = Depends(require_auth)):
        ensure_app()
        try:
            return estimate_view(_state(), type, share)
        except ApiError as e:
            raise HTTPException(e.status_code, e.detail)

    @app.post('/api/ml/settings')
    def update_settings(updates: dict, by: str, reason: str, auth: dict = Depends(require_auth)):
        ensure_app()
        st = _state()
        if st.settings is None:
            raise HTTPException(503, 'нет настроек')
        try:
            new = st.settings.rebase(updates, by, reason)
            new.save()
            st.settings = new
        except Exception as e:
            raise HTTPException(422, str(e))
        return {'version': new.version}

    @app.get('/api/ml/status')
    def status(auth: dict = Depends(require_auth)):
        ensure_app()
        st = _state()
        return {'settings_version': st.settings.version if st.settings else None,
                'thresholds': st.history.thresholds if st.history else None}


def start_api(state: State, host: str | None = None, port: int | None = None) -> threading.Thread:
    """П8: uvicorn из процесса сервиса (поток-демон); bind в момент старта."""
    import uvicorn
    ensure_app()
    bind(state)
    t = threading.Thread(target=lambda: uvicorn.run(
        app, host=host or config.API_HOST, port=port or config.API_PORT,
        log_level='warning'), daemon=True)
    t.start()
    return t