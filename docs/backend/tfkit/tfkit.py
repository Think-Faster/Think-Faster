"""Общее для наших сервисов контура: секреты из Vault, проверка токенов think-auth, события аудита.

Один файл без своих зависимостей, кроме PyJWT и cryptography для токенов и redis для аудита
(INTEGRATION §13.2, §13.4, §13.5). Подключается из сервиса так:

    sys.path.insert(0, str(<корень репозитория> / 'docs' / 'backend' / 'tfkit'))
    import tfkit

Режим контура — переменная TF_ENV: `prod` (по умолчанию) или `dev`. В dev секрет, которого нет в
Vault, берётся из переменной окружения. В prod — только из Vault, иначе SecretError с путём (без значения).
"""
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger('tfkit')


def env() -> str:
    return os.environ.get('TF_ENV', 'prod')


def is_dev() -> bool:
    return env() == 'dev'


# ----- секреты ---------------------------------------------------------------------------------
class SecretError(RuntimeError):
    """Секрет не прочитан. В тексте — только путь и поле, значения никогда."""


_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _vault_token() -> str | None:
    tok = os.environ.get('VAULT_TOKEN')
    if tok:
        return tok.strip()
    path = os.environ.get('VAULT_TOKEN_FILE')
    if path and Path(path).exists():
        return Path(path).read_text(encoding='utf-8').strip()
    return None


VAULT_RETRY = 5.0          # пауза между попытками, пока Vault запечатан или не поднялся


def vault_read(path: str, timeout: float = 5.0) -> dict:
    """Все поля `secret/tf/<path>` (KV v2). Кэш на время жизни процесса: секреты читаются при старте.

    Vault Гриши после каждого перезапуска запечатан и отвечает 503, пока его не распечатают ключами
    (think-infra/hashicorp/README.md). Чтобы сервис не падал по кругу, `TF_VAULT_WAIT` секунд он ждёт
    распечатывания (или подъёма Vault) и пишет об этом в лог; 403 и 404 — сразу ошибка."""
    with _lock:
        if path in _cache:
            return _cache[path]
    addr, tok = os.environ.get('VAULT_ADDR'), _vault_token()
    if not addr or not tok:
        raise SecretError(f'Vault не настроен (VAULT_ADDR и VAULT_TOKEN), нужен secret/tf/{path}')
    req = urllib.request.Request(f'{addr.rstrip("/")}/v1/secret/data/tf/{path}',
                                 headers={'X-Vault-Token': tok})
    deadline = time.monotonic() + float(os.environ.get('TF_VAULT_WAIT') or 0)
    said = None
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())['data']['data']
            break
        except urllib.error.HTTPError as e:
            problem, again = f'Vault ответил {e.code} на secret/tf/{path}', e.code in (502, 503, 504)
        except (urllib.error.URLError, OSError) as e:
            problem, again = (f'Vault недоступен для secret/tf/{path}: {type(e).__name__}', True)
        except (KeyError, ValueError) as e:
            raise SecretError(f'ответ Vault не KV v2 для secret/tf/{path}: {type(e).__name__}') from None
        if not again or time.monotonic() >= deadline:
            raise SecretError(problem)
        if said is None or time.monotonic() - said >= 60:
            log.warning('%s; жду распечатывания Vault', problem)
            said = time.monotonic()
        time.sleep(VAULT_RETRY)
    with _lock:
        _cache[path] = data
    return data


def secret(path: str, field: str, env_var: str | None = None, required: bool = True) -> str | None:
    """Поле секрета. Сначала Vault; в dev при неудаче — переменная окружения `env_var`."""
    try:
        value = vault_read(path).get(field)
        if value:
            return value
        problem = f'в secret/tf/{path} нет поля {field}'
    except SecretError as e:
        problem = str(e)
    if env_var and is_dev() and os.environ.get(env_var):
        # только в dev (локальный стенд без Vault); в логе имя переменной, не значение
        log.warning('%s; беру из переменной %s', problem, env_var)
        return os.environ[env_var]
    if required:
        raise SecretError(problem)
    return None


# ----- токены ----------------------------------------------------------------------------------
class TokenError(Exception):
    """`audit` — писать ли отказ в журнал действий. Протухший токен — штатная работа фронта (обновление
    раз в 10 минут), в журнал действий не идёт (права-и-аудит §6.2): остаётся строкой 401 в журнале
    запросов. Отказ по нашей вине (нет ключа, 503) — тоже не событие вызывающего."""

    def __init__(self, status: int, reason: str, jti: str | None = None, sub: str | None = None,
                 audit: bool | None = None):
        super().__init__(reason)
        self.status, self.reason, self.jti, self.sub = status, reason, jti, sub
        self.audit = status < 500 if audit is None else audit


class Verifier:
    """Проверка RS256-токенов think-auth (INTEGRATION §13.2).

    Поля в концепте и в think-auth пока расходятся (§12.2, Н10), поэтому принимаются оба варианта:
    `typ` или `token_type`, издатель `auth-service` или `tf-auth`. Техучётка — по `scope`, а пока
    think-auth его не кладёт, по `sub` из списка `service_subs`.
    """

    def __init__(self, public_key: str | bytes | None = None, jwks_url: str | None = None,
                 audience: str | None = 'api', issuers=('auth-service', 'tf-auth'),
                 service_subs=(), key_ttl: float = 3600.0, algorithms=('RS256',)):
        self._pem = public_key.encode() if isinstance(public_key, str) else public_key
        self.jwks_url = jwks_url
        self.audience = audience
        self.issuers = tuple(issuers)
        self.service_subs = set(service_subs)
        self.key_ttl = key_ttl
        self.algorithms = list(algorithms)
        self._key, self._key_at = None, 0.0

    def _load_key(self):
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
        pem = self._pem
        if pem is None:
            if not self.jwks_url:
                raise TokenError(503, 'нет публичного ключа аутентификации')
            try:
                with urllib.request.urlopen(self.jwks_url, timeout=5) as r:
                    pem = r.read()
            except (urllib.error.URLError, OSError):
                raise TokenError(503, 'аутентификация не отдала публичный ключ') from None
        if b'BEGIN CERTIFICATE' in pem:
            return x509.load_pem_x509_certificate(pem).public_key()
        return serialization.load_pem_public_key(pem)

    def key(self):
        if self._key is None or (self._pem is None and time.time() - self._key_at > self.key_ttl):
            self._key, self._key_at = self._load_key(), time.time()
        return self._key

    def verify(self, token: str, scope: str | None = None) -> dict:
        """Claims проверенного токена доступа; иначе TokenError с кодом 401 или 403."""
        import jwt
        try:
            claims = jwt.decode(token, self.key(), algorithms=self.algorithms, audience=self.audience,
                                options={'require': ['exp', 'sub'], 'verify_aud': self.audience is not None})
        except jwt.ExpiredSignatureError:
            raise TokenError(401, 'срок токена истёк', audit=False) from None
        except jwt.PyJWTError as e:
            jti = None
            try:
                jti = jwt.decode(token, options={'verify_signature': False}).get('jti')
            except jwt.PyJWTError:
                pass
            raise TokenError(401, f'токен не прошёл проверку: {type(e).__name__}', jti) from None
        jti, sub = claims.get('jti'), claims.get('sub')
        if self.issuers and claims.get('iss') not in self.issuers:
            raise TokenError(401, 'чужой издатель', jti, sub)
        if (claims.get('typ') or claims.get('token_type')) != 'access':
            raise TokenError(401, 'это не токен доступа', jti, sub)
        if scope is not None and not self.has_scope(claims, scope):
            raise TokenError(403, f'нужно право {scope}', jti, sub)
        return claims

    def has_scope(self, claims: dict, scope: str) -> bool:
        if scope in str(claims.get('scope', '')).split():
            return True
        return 'scope' not in claims and claims.get('sub') in self.service_subs

    def kind(self, claims: dict) -> str:
        """Техучётка — по `scope`, а без него по списку `service_subs` (как в has_scope)."""
        if claims.get('kind') in ('user', 'service'):
            return claims['kind']
        return 'service' if 'scope' in claims or claims.get('sub') in self.service_subs else 'user'


# ----- аудит -----------------------------------------------------------------------------------
# Строка — audit.events из docs/common/права-и-аудит.md §6.4; транспорт — вариант Б (§6.5):
# XADD в поток Redis `audit`. Лёг Redis — событие дописывается в файл и досылается позже.
FORBIDDEN = {'password', 'token', 'authorization', 'cookie', 'text', 'body'}


class Audit:
    def __init__(self, service: str, redis_url: str | None = None, spool: Path | None = None,
                 stream: str = 'audit', maxlen: int = 1_000_000):
        self.service, self.stream, self.maxlen = service, stream, maxlen
        self.redis_url = redis_url if redis_url is not None else os.environ.get('TF_REDIS_URL')
        self.spool = spool
        self._r = None
        self._lock = threading.RLock()       # такт, команды и ручки пишут из разных потоков

    def _redis(self):
        if self._r is None and self.redis_url:
            import redis
            self._r = redis.Redis.from_url(self.redis_url, socket_timeout=2, socket_connect_timeout=2)
        return self._r

    @staticmethod
    def _clean(details: dict) -> dict:
        # §6.3: пароли, токены, куки, тела и тексты сообщений не пишутся никогда
        bad = [k for k in details if k.lower() in FORBIDDEN]
        if bad:
            raise ValueError(f'в аудит нельзя писать поля {bad}')
        return details

    def event(self, event_type: str, outcome: str = 'success', *, actor_kind: str = 'service',
              actor_id: str | None = None, actor_login: str | None = None,
              request_id: str | None = None, ip: str | None = None, object_type: str | None = None,
              object_id: str | None = None, area_id: int | None = None, details: dict | None = None,
              occurred_at: datetime | None = None) -> dict:
        assert outcome in ('success', 'denied', 'error')
        assert actor_kind in ('user', 'service', 'anonymous')
        row = {'event_id': str(uuid.uuid4()),
               'occurred_at': (occurred_at or datetime.now(timezone.utc)).isoformat(),
               'service': self.service, 'event_type': event_type, 'outcome': outcome,
               'actor_kind': actor_kind, 'actor_id': actor_id, 'actor_login': actor_login,
               'request_id': request_id, 'ip': ip, 'object_type': object_type,
               'object_id': None if object_id is None else str(object_id), 'area_id': area_id,
               'details': self._clean(details or {})}
        self.send(row)
        return row

    def send(self, row: dict) -> bool:
        payload = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            return self._send(payload)

    def _send(self, payload: str) -> bool:
        try:
            r = self._redis()
            if r is not None:
                self._flush(r)
                r.xadd(self.stream, {'event': payload}, maxlen=self.maxlen, approximate=True)
                return True
        except Exception as e:                                  # аудит не роняет сервис
            log.warning('поток аудита недоступен: %s', type(e).__name__)
            self._r = None
        if self.spool is not None:
            self.spool.parent.mkdir(parents=True, exist_ok=True)
            with open(self.spool, 'a', encoding='utf-8') as f:
                f.write(payload + '\n')
        else:
            log.info('аудит без транспорта: %s', payload)
        return False

    def flush(self) -> int:
        """Дослать то, что копилось в файле, пока Redis лежал: перед каждым событием и по такту сервиса.
        Оборвётся посреди — часть строк уйдёт второй раз; сервис аудита отбрасывает повтор по `event_id`."""
        with self._lock:
            if self.spool is None or not self.spool.exists():
                return 0
            try:
                r = self._redis()
                return self._flush(r) if r is not None else 0
            except Exception as e:
                log.warning('поток аудита недоступен: %s', type(e).__name__)
                self._r = None
                return 0

    def _flush(self, r) -> int:
        if self.spool is None or not self.spool.exists():
            return 0
        lines = self.spool.read_text(encoding='utf-8').splitlines()
        for line in lines:
            r.xadd(self.stream, {'event': line}, maxlen=self.maxlen, approximate=True)
        self.spool.unlink()
        return len(lines)


def host() -> str:
    return socket.gethostname()
