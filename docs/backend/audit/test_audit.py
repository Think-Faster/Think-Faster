"""Сервис аудита на поддельных Redis и базе; с живыми — если заданы TF_AUDIT_TEST_REDIS и TF_AUDIT_TEST_DB
(база с прогнанным schema.sql, пользователь audit_writer; пароль — в строке подключения стенда)."""
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import audit
import tfkit


# ----- поддельный Redis: поток, группа, неподтверждённое, автозахват ---------------------------------
class FakeRedis:
    def __init__(self):
        self.streams: dict[str, list] = {}
        self.groups: dict[tuple, dict] = {}       # (поток, группа) → {'last': n, 'pel': {id: [consumer, t]}}
        self.n = 0

    def xadd(self, stream, fields, maxlen=None, approximate=True):
        self.n += 1
        eid = f'{self.n}-0'
        self.streams.setdefault(stream, []).append((eid, {k: v for k, v in fields.items()}))
        return eid

    def xgroup_create(self, stream, group, id='$', mkstream=False):
        if (stream, group) in self.groups:
            raise Exception('BUSYGROUP Consumer Group name already exists')
        self.streams.setdefault(stream, [])
        self.groups[(stream, group)] = {'last': 0 if id == '0' else self.n, 'pel': {}}

    def _num(self, eid):
        return int(eid.split('-')[0])

    def xreadgroup(self, group, consumer, streams, count=None, block=None):
        out = []
        for s, pos in streams.items():
            g = self.groups[(s, group)]
            if pos == '>':
                new = [e for e in self.streams[s] if self._num(e[0]) > g['last']][:count]
                for eid, _ in new:
                    g['pel'][eid] = [consumer, time.time()]
                    g['last'] = self._num(eid)
                entries = new
            else:
                mine = [eid for eid, (c, _) in g['pel'].items() if c == consumer and self._num(eid) > self._num(pos)]
                entries = [(eid, self._fields(s, eid)) for eid in sorted(mine, key=self._num)][:count]
            if entries:
                out.append((s, entries))
        return out

    def _fields(self, s, eid):
        return next((f for e, f in self.streams[s] if e == eid), None)   # вытеснено MAXLEN — None

    def xack(self, stream, group, *ids):
        pel = self.groups[(stream, group)]['pel']
        return sum(pel.pop(i, None) is not None for i in ids)

    def xautoclaim(self, stream, group, consumer, min_idle_time, start_id='0-0', count=100):
        pel, now, got = self.groups[(stream, group)]['pel'], time.time(), []
        for eid, (c, t) in sorted(pel.items(), key=lambda x: self._num(x[0])):
            if (now - t) * 1000 >= min_idle_time and len(got) < count:
                pel[eid] = [consumer, now]
                got.append((eid, self._fields(stream, eid)))
        return ['0-0', got, []]

    def pending(self, stream='audit'):
        return len(self.groups[(stream, audit.GROUP)]['pel'])


# ----- поддельная база: уникальность event_id, отказ по данным, обрыв соединения ---------------------
class FakeDB:
    def __init__(self):
        self.tables = {'events': [], 'requests': []}
        self.staged, self.closed, self.down, self.parts = [], False, False, 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        if self.down:
            raise psycopg.OperationalError('server closed the connection unexpectedly')
        for table, row in self.staged:
            self.tables[table].append(row)
        self.staged = []

    def rollback(self):
        self.staged = []

    def transaction(self):
        db = self

        class Savepoint:
            def __enter__(self):
                self.mark = len(db.staged)

            def __exit__(self, t, e, tb):
                if t is not None:
                    del db.staged[self.mark:]
                return False
        return Savepoint()

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, db):
        self.db, self.rowcount = db, -1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, args=()):
        if self.db.down:
            raise psycopg.OperationalError('server closed the connection unexpectedly')
        if 'ensure_partitions' in sql:
            self.db.parts += 1
            return
        table = sql.split('audit.')[1].split()[0]
        cols = sql.split('(')[1].split(')')[0].split(', ')
        row = dict(zip(cols, args))
        if row.get('object_id') == 'bad':
            raise psycopg.errors.CheckViolation('new row violates check constraint')
        if table == 'events':
            keys = {r['event_id'] for t, r in self.db.staged if t == 'events'} | \
                   {r['event_id'] for r in self.db.tables['events']}
            if row['event_id'] in keys:
                self.rowcount = 0                 # on conflict do nothing
                return
        self.db.staged.append((table, row))
        self.rowcount = 1

    def executemany(self, sql, seq):
        n = 0
        for args in seq:
            self.execute(sql, args)
            n += self.rowcount
        self.rowcount = n


@pytest.fixture
def env():
    r, db = FakeRedis(), FakeDB()
    kit = tfkit.Audit('bff')
    kit._r = r                                    # tfkit пишет прямо в поддельный поток
    w = audit.Writer(r, lambda: db, consumer='a1', claim_idle_ms=0)
    w.ensure_groups()
    return kit, r, db, w


def rows(db):
    return db.tables['events']


# ----- разбор ----------------------------------------------------------------------------------
def test_event_from_tfkit_lands_as_is(env):
    kit, r, db, w = env
    e = kit.event('ticket.decided', actor_kind='user', actor_id=str(uuid.uuid4()), actor_login='ivanov',
                  request_id='r1', ip='10.0.0.5', object_type='ticket', object_id=48213, area_id=12,
                  details={'decision': 'ложная', 'reason_code': 'R07'})
    assert w.step() == 1
    got = rows(db)[0]
    assert got['event_id'] == uuid.UUID(e['event_id']) and got['object_id'] == '48213'
    assert got['ip'] == '10.0.0.5' and got['area_id'] == 12 and got['details'].obj['reason_code'] == 'R07'
    assert r.pending() == 0 and w.stats['written'] == 1


def test_parse_normalises_and_strips():
    raw = json.dumps({'occurred_at': '2026-09-26T05:00:00', 'service': 'ml', 'event_type': 'forecast.muted',
                      'outcome': 'success', 'actor_kind': 'service', 'actor_id': 'tf-model', 'ip': 'не адрес',
                      'area_id': 'цех-3',
                      'details': {'why': 'ППР', 'Password': 'x', 'nested': [{'token': 'eyJ', 'ok': 1}]}})
    row = audit.parse_event(raw, '7-0')
    assert row['occurred_at'].utcoffset() == timedelta(hours=3)        # время без зоны — московское
    assert row['event_id'] == audit.parse_event(raw, '7-0')['event_id']  # повтор записи — тот же ключ
    assert row['actor_id'] is None and row['details']['actor_sub'] == 'tf-model'
    assert row['ip'] is None and row['area_id'] is None and row['details']['area_raw'] == 'цех-3'
    assert 'Password' not in row['details'] and row['details']['nested'] == [{'ok': 1}]
    assert sorted(row['details']['_dropped']) == ['Password', 'nested.token']


@pytest.mark.parametrize('over,why', [
    ({'event_type': 'Ticket Decided'}, 'event_type'),
    ({'outcome': 'ok'}, 'outcome'),
    ({'actor_kind': 'root'}, 'actor_kind'),
    ({'occurred_at': '2025-12-31T23:00:00+00:00'}, 'вне журнала'),
    ({'occurred_at': (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()}, 'вне журнала'),
    ({'occurred_at': 'вчера'}, 'не время'),
    ({'service': ''}, 'service'),
    ({'details': {'diff': 'x' * 20_000}}, 'details больше'),
])
def test_parse_refuses(over, why):
    e = {'occurred_at': '2026-09-26T05:00:00+03:00', 'service': 'bff', 'event_type': 'ticket.taken',
         'outcome': 'success', 'actor_kind': 'user'}
    e.update(over)
    with pytest.raises(audit.BadEvent, match=why):
        audit.parse_event(json.dumps(e), '1-0')


def test_request_row_needs_route_template():
    base = {'occurred_at': '2026-09-26T05:00:00Z', 'service': 'bff', 'method': 'get', 'status': 200,
            'duration_ms': 12, 'actor_kind': 'user', 'ip': '10.0.0.1'}
    row = audit.parse_request(json.dumps({**base, 'route': '/tickets/{id}/take'}), '1-0')
    assert row['method'] == 'GET' and row['route'] == '/tickets/{id}/take'
    with pytest.raises(audit.BadEvent, match='шаблон'):
        audit.parse_request(json.dumps({**base, 'route': '/tickets?login=ivanov'}), '2-0')
    with pytest.raises(audit.BadEvent):
        audit.parse_request(json.dumps({**base, 'route': '/x', 'status': 'ok'}), '3-0')


# ----- запись ----------------------------------------------------------------------------------
def test_duplicate_from_spool_is_written_once(env):
    kit, r, db, w = env
    e = kit.event('notify.sent', details={'channel': 'email', 'recipients': 3})
    r.xadd('audit', {'event': json.dumps(e)})     # досылка из файла tfkit после обрыва
    assert w.step() == 2
    assert len(rows(db)) == 1 and w.stats['duplicates'] == 1 and r.pending() == 0


def test_bad_entries_go_to_dead_stream_and_do_not_block(env):
    kit, r, db, w = env
    r.xadd('audit', {'event': '{не json'})
    kit.event('ticket.taken', actor_kind='user')
    kit.event('ticket.taken', actor_kind='user', object_type='ticket', object_id='bad')  # база отвергла
    kit.event('ticket.commented', actor_kind='user', details={'comment': 'камеру посмотрел'})
    w.step()
    assert [x['event_type'] for x in rows(db)] == ['ticket.taken', 'ticket.commented']
    dead = r.streams['audit:dead']
    assert [d[1]['reason'].split(':')[0] for d in dead] == ['не JSON', 'база']
    assert r.pending() == 0 and w.stats['dead'] == 2


def test_db_down_nothing_acked_then_delivered(env):
    kit, r, db, w = env
    kit.event('threshold.changed', actor_kind='user', details={'sensor': 17, 'was': 60, 'now': 65})
    db.down = True
    with pytest.raises(psycopg.OperationalError):
        w.step()
    assert r.pending() == 1 and rows(db) == []    # в потоке, не подтверждено
    db.down = False
    w._claimed_at = 0.0                           # как в run(): после сбоя — сначала своё
    w.step()
    assert len(rows(db)) == 1 and r.pending() == 0


def test_stale_entry_of_dead_consumer_is_claimed(env):
    kit, r, db, w = env
    kit.event('markup.exported', actor_kind='user', details={'rows': 1200})
    other = audit.Writer(r, lambda: FakeDB(), consumer='a2')
    r.xreadgroup(audit.GROUP, 'a2', {'audit': '>'}, count=10)   # взял и упал
    assert r.pending() == 1
    w.step()
    assert len(rows(db)) == 1 and r.pending() == 0
    assert other.stats['written'] == 0


def test_trimmed_entry_counted_and_acked(env):
    kit, r, db, w = env
    kit.event('ticket.taken', actor_kind='user')
    r.xreadgroup(audit.GROUP, 'a2', {'audit': '>'}, count=10)
    r.streams['audit'].clear()                    # вытеснено MAXLEN, пока висело
    w.step()
    assert w.stats['lost'] == 1 and r.pending() == 0


def test_requests_stream_and_partitions(env):
    kit, r, db, w = env
    r.xadd('audit:requests', {'event': json.dumps({
        'occurred_at': '2026-09-26T05:00:00+03:00', 'service': 'bff', 'method': 'POST',
        'route': '/tickets/{id}/decide', 'status': 200, 'duration_ms': 41, 'actor_kind': 'user'})})
    w.step()
    assert db.tables['requests'][0]['route'] == '/tickets/{id}/decide'
    assert db.parts == 1                          # при первом подключении — партиции на месяцы вперёд


# ----- чтение ----------------------------------------------------------------------------------
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def token(**over):
    c = {'sub': 'bff-svc', 'jti': 'j1', 'iss': 'auth-service', 'aud': 'api', 'token_type': 'access',
         'scope': 'audit.read', 'exp': int(time.time()) + 600}
    c.update(over)
    return jwt.encode({k: v for k, v in c.items() if v is not None}, KEY, algorithm='RS256')


class ReadDB(FakeDB):
    def __init__(self):
        super().__init__()
        self.asked = []

    def cursor(self):
        db = self

        class C(FakeCursor):
            description = [('id',), ('event_id',), ('occurred_at',), ('received_at',), ('event_type',),
                           ('actor_id',)]

            def execute(self, sql, args=()):
                db.asked.append((sql, list(args)))

            def fetchall(self):
                t = datetime(2026, 9, 26, 5, tzinfo=timezone.utc)
                return [(7, uuid.UUID(int=7), t, t, 'forecast.muted', None)]
        return C(db)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    kit, db = tfkit.Audit('audit'), ReadDB()
    kit.events = []
    kit.send = lambda row: kit.events.append(row) or True
    app = audit.make_app(lambda: db, tfkit.Verifier(PEM), audit=kit)
    return TestClient(app), db, kit


def test_read_filters_and_prefix(client):
    c, db, _ = client
    r = c.get('/events', headers={'Authorization': 'Bearer ' + token()},
              params={'event_type': ['forecast.', 'ticket.decided'], 'object_id': '5122',
                      'since': '2026-09-01T00:00:00+03:00', 'limit': 5000})
    assert r.status_code == 200 and r.json()[0]['event_type'] == 'forecast.muted'
    sql, args = db.asked[-1]
    assert '(event_type like %s or event_type = %s)' in sql and 'object_id = %s' in sql
    assert args[0] == 'forecast.%' and args[1] == 'ticket.decided' and args[-1] == 1000


@pytest.mark.parametrize('tok,code,event', [
    (None, 401, 'token.refused'),
    (token(exp=int(time.time()) - 5), 401, None),                 # протухший — без события (§6.2)
    (token(scope=None, sub='u1'), 403, 'access.denied'),          # пользователь — только через BFF
    (token(scope='ml.write'), 403, 'access.denied'),
])
def test_read_refusals(client, tok, code, event):
    c, db, kit = client
    r = c.get('/events', headers={'Authorization': 'Bearer ' + tok} if tok else {})
    assert r.status_code == code and db.asked == []
    assert [e['event_type'] for e in kit.events] == ([event] if event else [])
    assert 'eyJ' not in repr(kit.events)


# ----- живой стенд ------------------------------------------------------------------------------
LIVE = os.environ.get('TF_AUDIT_TEST_REDIS') and os.environ.get('TF_AUDIT_TEST_DB')


@pytest.mark.skipif(not LIVE, reason='нет TF_AUDIT_TEST_REDIS и TF_AUDIT_TEST_DB')
def test_live_roundtrip():
    import redis
    r = redis.Redis.from_url(os.environ['TF_AUDIT_TEST_REDIS'], decode_responses=True)
    connect = lambda: psycopg.connect(os.environ['TF_AUDIT_TEST_DB'])   # noqa: E731
    w = audit.Writer(r, connect, consumer='test', block_ms=100, claim_idle_ms=0)
    w.ensure_groups()
    kit = tfkit.Audit('audit-test', redis_url=os.environ['TF_AUDIT_TEST_REDIS'])
    e = kit.event('ticket.taken', actor_kind='user', object_type='ticket', object_id='live-1')
    kit.send(e)                                   # повтор
    for _ in range(5):
        w.step()
    with connect() as db:
        got = audit.query_events(db, ['ticket.taken'], object_id='live-1', service='audit-test')
        assert [x['event_id'] for x in got].count(e['event_id']) == 1
        with pytest.raises(psycopg.Error):
            db.execute('delete from audit.events where object_id = %s', ['live-1'])
