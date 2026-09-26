"""Сервис аудита: поток Redis → PostgreSQL `audit` (права-и-аудит §6, ML/INTEGRATION.md §13.4).

Сервисы кладут события в поток Redis (`XADD audit`, `tfkit.Audit`) и работают дальше (§6.5, вариант
Б). Этот сервис читает поток группой потребителей `audit-writer`, пишет в базу пачками и только
после записи подтверждает (XACK). Лёг PostgreSQL — события ждут в потоке и доедут; упал сам сервис —
неподтверждённое перечитывается при старте, а зависшее у другого экземпляра забирается (XAUTOCLAIM).
Повтор той же записи отбрасывается по `event_id`. Запись, которую нельзя положить в таблицу, уходит
в поток `audit:dead` с причиной и не держит остальные.

Поток `audit:requests` — журнал запросов (§6.1, строка на запрос) — пишется тем же порядком в
`audit.requests`.

Чтение: `GET /events` — только техучётке с правом `audit.read` (BFF проверяет право пользователя у
себя: историю отклонённых и заглушённых прогнозов видит только главный диспетчер, §13.4).
`GET /health` — без токена.

    python audit.py            запись и ручки
    python audit.py --once     одна пачка из потока и выход (проверка стенда)

Окружение: TF_REDIS_URL (пароль Redis — из Vault `secret/tf/redis`), TF_AUDIT_DB (строка подключения
без пароля, пользователь audit_user), пароль — Vault `secret/tf/postgres/audit` поле
`TF_PG_AUDIT_USER_PASSWORD` (в dev — переменная с тем же именем), TF_AUTH_PUBLIC_KEY или TF_AUTH_JWKS,
техучётки без `scope` — Vault `secret/tf/app/tf-audit` поле `TF_AUDIT_SERVICE_SUBS`.
"""
import argparse
import ipaddress
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _kit in (os.environ.get('TF_KIT'), HERE.parent / 'tfkit', HERE / 'tfkit'):
    if _kit and (Path(_kit) / 'tfkit.py').exists():
        sys.path.insert(0, str(_kit))
        break
import tfkit  # noqa: E402

log = logging.getLogger('tf-audit')

STREAM_EVENTS, STREAM_REQUESTS, STREAM_DEAD = 'audit', 'audit:requests', 'audit:dead'
GROUP = 'audit-writer'
MSK = timezone(timedelta(hours=3), 'MSK')
EARLIEST = datetime(2026, 1, 1, tzinfo=timezone.utc)    # раньше партиций нет (schema.sql), в любой зоне базы
EVENT_TYPE = re.compile(r'^[a-z][a-z_]*(\.[a-z_]+)+$')
MAX_DETAILS = 16_384                                     # байт JSON: «было → стало», не тела запросов
NS = uuid.UUID('6b1f5c1e-7a0e-4f7e-9d43-3c2a1b0e5d11')  # event_id записи без него — из id в потоке

EVENT_COLS = ('event_id', 'occurred_at', 'service', 'event_type', 'outcome', 'actor_kind', 'actor_id',
              'actor_login', 'request_id', 'ip', 'object_type', 'object_id', 'area_id', 'details')
REQUEST_COLS = ('occurred_at', 'service', 'method', 'route', 'status', 'duration_ms', 'actor_kind',
                'actor_id', 'request_id', 'ip')


class BadEvent(ValueError):
    """Запись из потока нельзя положить в таблицу: уходит в audit:dead."""


# ----- разбор ----------------------------------------------------------------------------------
def _when(v) -> datetime:
    try:
        t = datetime.fromisoformat(str(v))
    except ValueError:
        raise BadEvent(f'occurred_at не время: {str(v)[:40]!r}') from None
    if t.tzinfo is None:                     # время без зоны в контуре — московское (INTEGRATION §8)
        t = t.replace(tzinfo=MSK)
    if not EARLIEST <= t <= datetime.now(timezone.utc) + timedelta(days=1):
        raise BadEvent(f'occurred_at вне журнала: {t.isoformat()}')
    return t


def _uuid(v, details: dict, name: str):
    """sub из токена — uuid по концепту; think-auth может выдать другой вид (Н10), тогда — в details."""
    if v in (None, ''):
        return None
    try:
        return uuid.UUID(str(v))
    except ValueError:
        details[name] = str(v)[:200]
        return None


def _ip(v):
    if v in (None, ''):
        return None
    try:
        return str(ipaddress.ip_address(str(v).split('%')[0]))
    except ValueError:
        return None


def _strip(d, dropped: list, path=''):
    """§6.3: паролей, токенов, кук, тел и текстов сообщений в журнале нет никогда — даже если
    источник их прислал. Сервис аудита — последний рубеж: такие поля вырезаются на любой глубине."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if str(k).lower() in tfkit.FORBIDDEN:
                dropped.append(path + str(k))
            else:
                out[k] = _strip(v, dropped, f'{path}{k}.')
        return out
    if isinstance(d, list):
        return [_strip(v, dropped, path) for v in d]
    return d


def _text(v, n: int = 200):
    return None if v in (None, '') else str(v)[:n]


def parse_event(raw: str, entry_id: str) -> dict:
    try:
        e = json.loads(raw)
    except (TypeError, ValueError):
        raise BadEvent('не JSON') from None
    if not isinstance(e, dict):
        raise BadEvent('не объект')
    details = e.get('details') or {}
    if not isinstance(details, dict):
        raise BadEvent('details не объект')
    dropped: list[str] = []
    details = _strip(details, dropped)
    if dropped:
        details['_dropped'] = dropped
        log.warning('из события %s вырезаны поля %s', e.get('event_type'), dropped)
    service, event_type = _text(e.get('service'), 40), str(e.get('event_type') or '')
    if not service:
        raise BadEvent('нет service')
    if not EVENT_TYPE.match(event_type):
        raise BadEvent(f'event_type не вида «сущность.действие»: {event_type[:40]!r}')
    if e.get('outcome') not in ('success', 'denied', 'error'):
        raise BadEvent(f'outcome: {str(e.get("outcome"))[:20]!r}')
    if e.get('actor_kind') not in ('user', 'service', 'anonymous'):
        raise BadEvent(f'actor_kind: {str(e.get("actor_kind"))[:20]!r}')
    area = e.get('area_id')
    if area is not None:
        try:
            area = int(area)
        except (TypeError, ValueError):
            details['area_raw'], area = str(area)[:40], None
    try:
        eid = uuid.UUID(str(e['event_id']))
    except (KeyError, ValueError):
        eid = uuid.uuid5(NS, entry_id)       # повтор той же записи потока даст тот же ключ
    row = {'event_id': eid, 'occurred_at': _when(e.get('occurred_at')), 'service': service,
           'event_type': event_type, 'outcome': e['outcome'], 'actor_kind': e['actor_kind'],
           'actor_id': _uuid(e.get('actor_id'), details, 'actor_sub'), 'actor_login': _text(e.get('actor_login')),
           'request_id': _text(e.get('request_id'), 100), 'ip': _ip(e.get('ip')),
           'object_type': _text(e.get('object_type'), 60), 'object_id': _text(e.get('object_id')),
           'area_id': area, 'details': details}
    if len(json.dumps(details, ensure_ascii=False, default=str).encode()) > MAX_DETAILS:
        raise BadEvent(f'details больше {MAX_DETAILS} байт')
    return row


def parse_request(raw: str, entry_id: str) -> dict:
    try:
        e = json.loads(raw)
        row = {'occurred_at': _when(e.get('occurred_at')), 'service': _text(e['service'], 40),
               'method': str(e['method']).upper()[:10], 'route': _text(e['route'], 300),
               'status': int(e['status']), 'duration_ms': int(e['duration_ms']),
               'actor_kind': e.get('actor_kind') or 'anonymous', 'actor_id': _uuid(e.get('actor_id'), {}, 'x'),
               'request_id': _text(e.get('request_id'), 100), 'ip': _ip(e.get('ip'))}
    except BadEvent:
        raise
    except (TypeError, ValueError, KeyError, AttributeError) as x:
        raise BadEvent(f'строка запроса: {type(x).__name__} {x}') from None
    if '?' in (row['route'] or ''):
        raise BadEvent('в route строка запроса: нужен шаблон маршрута (§6.4)')
    if not row['service'] or not row['route']:
        raise BadEvent('нет service или route')
    return row


# ----- запись ----------------------------------------------------------------------------------
def _sql(table: str, cols: tuple) -> str:
    vals = ', '.join('%s::inet' if c == 'ip' else '%s' for c in cols)
    tail = ' on conflict do nothing' if table == 'events' else ''
    return f'insert into audit.{table} ({", ".join(cols)}) values ({vals}){tail}'


class Writer:
    """Поток Redis → таблица; подтверждение только после commit."""

    STREAMS = {STREAM_EVENTS: ('events', EVENT_COLS, parse_event),
               STREAM_REQUESTS: ('requests', REQUEST_COLS, parse_request)}

    def __init__(self, redis, connect, consumer: str | None = None, batch: int = 500,
                 block_ms: int = 5000, claim_idle_ms: int = 60_000):
        self.r, self.connect = redis, connect
        self.consumer = consumer or socket.gethostname()
        self.batch, self.block_ms, self.claim_idle_ms = batch, block_ms, claim_idle_ms
        self.db = None
        self.stats = {'written': 0, 'duplicates': 0, 'dead': 0, 'lost': 0, 'last_write': None}
        self._claimed_at = 0.0
        self._parts_at = 0.0

    # --- подготовка
    def ensure_groups(self) -> None:
        for s in self.STREAMS:
            try:
                # id '0': при первом создании группы забрать и то, что копилось в потоке до неё
                self.r.xgroup_create(s, GROUP, id='0', mkstream=True)
            except Exception as e:
                if 'BUSYGROUP' not in str(e):
                    raise

    def _conn(self):
        if self.db is None or getattr(self.db, 'closed', False):
            self.db = self.connect()
        if time.time() - self._parts_at > 86_400:          # новый месяц — новая партиция
            with self.db.cursor() as cur:
                cur.execute('select audit.ensure_partitions()')
            self.db.commit()
            self._parts_at = time.time()
        return self.db

    # --- одна пачка
    def handle(self, stream: str, entries: list) -> int:
        table, cols, parse = self.STREAMS[stream]
        rows, ack, dead = [], [], []
        for entry_id, fields in entries:
            if fields is None:            # запись вытеснена из потока (MAXLEN), пока ждала: только в счётчик
                self.stats['lost'] += 1
                ack.append(entry_id)
                continue
            raw = fields.get('event')
            try:
                rows.append((entry_id, parse(raw, entry_id)))
            except BadEvent as e:
                dead.append((entry_id, raw, str(e)))
        refused = self._insert(table, cols, rows) if rows else []
        dead += refused
        for entry_id, raw, why in dead:
            self.r.xadd(STREAM_DEAD, {'stream': stream, 'id': entry_id, 'event': str(raw)[:65_536],
                                      'reason': why[:500]}, maxlen=100_000, approximate=True)
            log.warning('в %s: %s %s — %s', STREAM_DEAD, stream, entry_id, why)
        ids = ack + [i for i, _ in rows] + [i for i, *_ in dead]
        ids = list(dict.fromkeys(ids))
        if ids:
            self.r.xack(stream, GROUP, *ids)
        self.stats['dead'] += len(dead)
        return len(rows) - len(refused)

    def _insert(self, table: str, cols: tuple, rows: list) -> list:
        """Пачка одной транзакцией. Если база отвергла строку по данным — по одной, каждая в своей точке
        сохранения: плохая уходит в audit:dead, остальные пишутся. Сбой соединения — наружу: ничего не
        подтверждено, пачка перечитается."""
        import psycopg
        from psycopg.types.json import Jsonb
        db = self._conn()
        vals = [tuple(Jsonb(r[c]) if c == 'details' else r[c] for c in cols) for _, r in rows]
        sql = _sql(table, cols)
        try:
            with db.cursor() as cur:
                cur.executemany(sql, vals)
                n = cur.rowcount
            db.commit()
            self._done(len(rows), n)
            return []
        except psycopg.OperationalError:
            self._drop_conn()
            raise
        except psycopg.Error:
            db.rollback()
        dead, n = [], 0
        for (entry_id, r), v in zip(rows, vals):
            try:
                with db.transaction(), db.cursor() as cur:
                    cur.execute(sql, v)
                    n += max(cur.rowcount, 0)
            except psycopg.OperationalError:
                self._drop_conn()
                raise
            except psycopg.Error as e:
                dead.append((entry_id, json.dumps({c: r[c] for c in cols}, ensure_ascii=False, default=str),
                             f'база: {type(e).__name__}: {str(e).splitlines()[0][:300]}'))
        db.commit()
        self._done(len(rows) - len(dead), n)
        return dead

    def _done(self, sent: int, written: int) -> None:
        written = sent if written is None or written < 0 else written
        self.stats['written'] += written
        self.stats['duplicates'] += max(sent - written, 0)
        self.stats['last_write'] = datetime.now(MSK).isoformat(timespec='seconds')

    def _drop_conn(self) -> None:
        try:
            if self.db is not None:
                self.db.close()
        finally:
            self.db = None

    # --- цикл
    def _pending(self, stream: str) -> int:
        """Своё неподтверждённое — после сбоя базы или перезапуска с тем же именем потребителя."""
        n = 0
        while True:
            resp = self.r.xreadgroup(GROUP, self.consumer, {stream: '0'}, count=self.batch)
            entries = resp[0][1] if resp else []
            if not entries:
                return n
            n += self.handle(stream, entries)

    def _claim(self, stream: str) -> int:
        """Зависшее у упавшего экземпляра дольше claim_idle_ms — забрать себе."""
        res = self.r.xautoclaim(stream, GROUP, self.consumer, min_idle_time=self.claim_idle_ms,
                                start_id='0-0', count=self.batch)
        entries = res[1] if res else []
        return self.handle(stream, entries) if entries else 0

    def step(self) -> int:
        n = 0
        if time.time() - self._claimed_at > self.claim_idle_ms / 1000:
            for s in self.STREAMS:
                n += self._pending(s) + self._claim(s)
            self._claimed_at = time.time()
        resp = self.r.xreadgroup(GROUP, self.consumer, {s: '>' for s in self.STREAMS},
                                 count=self.batch, block=self.block_ms)
        for stream, entries in resp or []:
            n += self.handle(stream, entries)
        return n

    def run(self, stop: threading.Event, heartbeat: Path | None = None) -> None:
        wait = 1.0
        while not stop.is_set():
            try:
                self.ensure_groups()
                while not stop.is_set():
                    self.step()
                    if heartbeat is not None:
                        heartbeat.touch()
                    wait = 1.0
            except Exception as e:                    # база или Redis легли: ждать, ничего не терять
                log.warning('запись аудита стоит: %s: %s — повтор через %.0f с', type(e).__name__,
                            str(e).splitlines()[0][:200] if str(e) else '', wait)
                self._claimed_at = 0.0                # после сбоя сначала своё неподтверждённое
                self._drop_conn()                     # транзакция могла остаться прерванной
                stop.wait(wait)
                wait = min(wait * 2, 30.0)


# ----- чтение ----------------------------------------------------------------------------------
READ_SCOPE = 'audit.read'
FILTERS = {'service': 'service = %s', 'object_type': 'object_type = %s', 'object_id': 'object_id = %s',
           'actor_login': 'actor_login = %s', 'request_id': 'request_id = %s', 'outcome': 'outcome = %s'}


def query_events(db, event_type: list[str] | None = None, since: datetime | None = None,
                 until: datetime | None = None, before_id: int | None = None, limit: int = 200,
                 **eq) -> list[dict]:
    """События по фильтрам, новые сверху. `event_type` — точные имена или префиксы с точкой на конце
    (`forecast.` — все события прогнозов). Страницы — по `before_id`."""
    where, args = [], []
    if event_type:
        parts = []
        for t in event_type:
            if t.endswith('.'):
                parts.append('event_type like %s')
                args.append(t.replace('%', '').replace('_', r'\_') + '%')
            else:
                parts.append('event_type = %s')
                args.append(t)
        where.append('(' + ' or '.join(parts) + ')')
    for k, v in eq.items():
        if v is not None:
            where.append(FILTERS[k])
            args.append(v)
    if since is not None:
        where.append('occurred_at >= %s')
        args.append(since)
    if until is not None:
        where.append('occurred_at < %s')
        args.append(until)
    if before_id is not None:
        where.append('id < %s')
        args.append(before_id)
    sql = ('select id, event_id, occurred_at, received_at, service, event_type, outcome, actor_kind, actor_id, '
           'actor_login, request_id, host(ip) as ip, object_type, object_id, area_id, details from audit.events'
           + (' where ' + ' and '.join(where) if where else '') + ' order by occurred_at desc, id desc limit %s')
    args.append(max(1, min(int(limit), 1000)))
    with db.cursor() as cur:
        cur.execute(sql, args)
        names = [d[0] for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    db.rollback()                                     # только чтение: не держать транзакцию
    for r in rows:
        for k in ('event_id', 'actor_id'):
            r[k] = None if r[k] is None else str(r[k])
        for k in ('occurred_at', 'received_at'):
            r[k] = r[k].isoformat()
    return rows


def make_app(connect, verifier, writer: Writer | None = None, audit: 'tfkit.Audit | None' = None):
    from fastapi import FastAPI, Header, HTTPException, Query, Request
    app = FastAPI(title='tf-audit', docs_url=None, redoc_url=None)
    if isinstance(audit, tfkit.Audit):
        tfkit.request_log(app, audit, verifier)          # чтение журнала — тоже строка журнала запросов
    local = threading.local()

    def db():
        c = getattr(local, 'db', None)
        if c is None or getattr(c, 'closed', False):
            c = local.db = connect()
        return c

    def refuse(event: str, status: int, reason: str, request: Request, claims: dict | None = None, jti=None):
        """Как в ML/service/api.py: 401 — исполнитель неизвестен, 403 — его `sub`; из токена только `jti`."""
        claims = claims or {}
        if audit is not None:
            try:
                audit.event(event, 'denied', actor_kind=verifier.kind(claims) if claims else 'anonymous',
                            actor_id=claims.get('sub'), request_id=request.headers.get('x-request-id'),
                            ip=request.client.host if request.client else None, object_type='route',
                            object_id=request.url.path, details={'reason': reason, 'jti': claims.get('jti', jti)})
            except Exception:
                log.exception('аудит %s не записан', event)
        raise HTTPException(status, reason)

    def check(request: Request, authorization: str | None):
        if verifier is None:                          # dev-стенд без ключа
            return
        if not authorization or not authorization.lower().startswith('bearer '):
            refuse('token.refused', 401, 'нет токена', request)
        try:
            claims = verifier.verify(authorization.split(None, 1)[1].strip())
        except tfkit.TokenError as e:
            if not e.audit:                           # ключа нет или токен протух (§6.2): без события
                raise HTTPException(e.status, e.reason) from None
            refuse('token.refused', e.status, e.reason, request, jti=e.jti)
        if verifier.kind(claims) != 'service':
            refuse('access.denied', 403, 'журнал отдаётся только техучётке BFF', request, claims)
        if not verifier.has_scope(claims, READ_SCOPE):
            refuse('access.denied', 403, f'нужно право {READ_SCOPE}', request, claims)

    @app.get('/health')
    def health():
        return {'ok': True, **(writer.stats if writer is not None else {})}

    @app.get('/events')
    def events(request: Request, event_type: list[str] = Query(default=[]), service: str | None = None,
               object_type: str | None = None, object_id: str | None = None, actor_login: str | None = None,
               request_id: str | None = None, outcome: str | None = None, since: datetime | None = None,
               until: datetime | None = None, before_id: int | None = None, limit: int = 200,
               authorization: str | None = Header(default=None)):
        check(request, authorization)
        try:
            return query_events(db(), event_type or None, since, until, before_id, limit, service=service,
                                object_type=object_type, object_id=object_id, actor_login=actor_login,
                                request_id=request_id, outcome=outcome)
        except Exception as e:
            local.db = None
            log.warning('чтение журнала: %s', type(e).__name__)
            raise HTTPException(503, 'журнал аудита недоступен') from None

    return app


# ----- запуск ----------------------------------------------------------------------------------
def connector():
    import psycopg
    dsn = os.environ.get('TF_AUDIT_DB', 'host=tf-postgres dbname=tf user=audit_user')
    pw = tfkit.secret('postgres/audit', 'TF_PG_AUDIT_USER_PASSWORD', 'TF_PG_AUDIT_USER_PASSWORD',
                      required=not tfkit.is_dev())

    def connect():
        return psycopg.connect(dsn, password=pw, connect_timeout=5, application_name='tf-audit')
    return connect


def make_verifier():
    pem = os.environ.get('TF_AUTH_PUBLIC_KEY') or None           # открытый ключ, в Vault его нет
    subs = tfkit.secret('app/tf-audit', 'TF_AUDIT_SERVICE_SUBS', 'TF_AUDIT_SERVICE_SUBS', required=False) or ''
    jwks = os.environ.get('TF_AUTH_JWKS')
    if pem is None and not jwks:
        if tfkit.is_dev():
            log.warning('dev: ключа проверки токенов нет — чтение журнала открыто')
            return None
        jwks = 'http://tf-auth:8080/.well-known/jwks'
    return tfkit.Verifier(public_key=pem, jwks_url=None if pem else jwks,
                          service_subs=[s.strip() for s in subs.split(',') if s.strip()])


def main() -> None:
    import redis
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--once', action='store_true', help='одна пачка из потока и выход')
    ap.add_argument('--port', type=int, default=int(os.environ.get('TF_AUDIT_PORT', '8000')))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    r = redis.Redis.from_url(tfkit.redis_url(os.environ.get('TF_REDIS_URL', 'redis://tf-redis:6379/0')),
                             decode_responses=True,
                             socket_timeout=10, socket_connect_timeout=5)
    connect = connector()
    w = Writer(r, connect)
    if args.once:
        w.ensure_groups()
        w._claimed_at = 0.0
        print(json.dumps({'written': w.step(), **w.stats}, ensure_ascii=False))
        return
    import signal
    import uvicorn
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    t = threading.Thread(target=w.run, args=(stop, Path('/tmp/tf-audit.alive')), name='writer', daemon=True)
    t.start()
    # события самого аудита (отказы на чтении) идут тем же потоком
    app = make_app(connect, make_verifier(), w, tfkit.Audit('audit'))
    server = uvicorn.Server(uvicorn.Config(app, host='0.0.0.0', port=args.port, log_level='warning'))
    threading.Thread(target=lambda: (stop.wait(), setattr(server, 'should_exit', True)), daemon=True).start()
    server.run()
    stop.set()
    t.join(timeout=10)


if __name__ == '__main__':
    main()
