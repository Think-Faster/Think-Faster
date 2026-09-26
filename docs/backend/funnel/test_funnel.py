"""Воронка без сети: запись в Kafka и эмулятор подменены, аудит пишет в список."""
import json
import os
import sys
import time
import types
import urllib.error

for _v in ('VAULT_ADDR', 'VAULT_TOKEN', 'VAULT_TOKEN_FILE', 'TF_REDIS_URL', 'TF_AUTH_PUBLIC_KEY', 'TF_AUTH_JWKS',
           'TF_FUNNEL_SERVICE_SUBS', 'TF_KAFKA_FUNNEL_PASSWORD', 'TF_FUNNEL_PULL'):
    os.environ.pop(_v, None)
os.environ['TF_ENV'] = 'dev'

import jwt  # noqa: E402
import pytest  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

import funnel  # noqa: E402
import tfkit  # noqa: E402


def ev(channel=196771, value='0.4', alarm=False, **over):
    e = {'ид_события': 1, 'ид_канала_данных': channel, 'дата': '2026-09-26', 'время': '03:09:27',
         'тревожное': alarm, 'значение_датчика': value}
    e.update(over)
    return e


class Sink:
    def __init__(self):
        self.sent, self.down = [], False

    def send(self, messages):
        if self.down:
            raise funnel.Unavailable('Kafka не подтвердила 1 сообщений')
        self.sent.extend(messages)


def recorder():
    kit = tfkit.Audit('funnel', redis_url='')
    kit.events = []
    kit.send = lambda row: kit.events.append(row) or True
    return kit


@pytest.fixture
def f():
    return funnel.Funnel(Sink(), recorder(), funnel.Channels(3600))


# ----- разбор ----------------------------------------------------------------------------------
def test_normalize_keeps_journal_fields_only():
    raw = ev(value=12, alarm='True', курсор=5, название_объекта='Объект 1', сбой='stuck')
    assert funnel.normalize(raw) == {'ид_события': 1, 'ид_канала_данных': 196771, 'дата': '2026-09-26',
                                     'время': '03:09:27', 'тревожное': True, 'значение_датчика': '12'}


@pytest.mark.parametrize('raw, reason', [
    ('строка', 'не объект'),
    ({'дата': '2026-09-26'}, 'нет ид_канала'),
    (ev(channel='abc'), 'не число'),
    (ev(channel=True), 'не число'),
    (ev(channel=-3), 'не число'),
    (ev(дата='26.09.2026'), 'не в формате'),
    (ev(время=None), 'не в формате'),
    ({k: v for k, v in ev().items() if k != 'время'}, 'нет даты'),
    (ev(alarm='может быть'), 'тревожное'),
    (ev(value=None), 'нет значения'),
    (ev(value=False), 'нет значения'),
    (ev(value='  '), 'пустое'),
    (ev(value='x' * 256), 'длиннее'),
    (ev(ид_события='x'), 'ид_события'),
])
def test_normalize_rejects(raw, reason):
    with pytest.raises(ValueError, match=reason):
        funnel.normalize(raw)


@pytest.mark.parametrize('value, alarm, topic', [
    ('0.4', False, funnel.TOPIC_READINGS),
    ('-12', False, funnel.TOPIC_READINGS),
    ('1e-3', False, funnel.TOPIC_READINGS),
    ('0.4', True, funnel.TOPIC_JOURNAL),          # тревога — в ленту BFF, даже числом
    ('Норма', False, funnel.TOPIC_JOURNAL),
    ('##.##.2026 ##:##', False, funnel.TOPIC_JOURNAL),
    ('nan', False, funnel.TOPIC_JOURNAL),
])
def test_topic(value, alarm, topic):
    assert funnel.topic_of(funnel.normalize(ev(value=value, alarm=alarm))) == topic


def test_packet_shapes():
    assert funnel.events_of({'events': [1]}) == [1]
    assert funnel.events_of([1, 2]) == [1, 2]
    assert funnel.events_of({'курсор': 9, 'событий': 1, 'события': [3]}) == [3]
    for bad in ({'x': 1}, 'текст', None, {'events': {}}):
        with pytest.raises(ValueError, match='список'):
            funnel.events_of(bad)
    with pytest.raises(ValueError, match='больше'):
        funnel.events_of([{}] * (funnel.MAX_EVENTS + 1))


# ----- приём -----------------------------------------------------------------------------------
def test_take_partial_routes_by_channel_and_audits_without_values(f):
    out = f.take([ev(1, '0.4'), ev(2, 'Норма'), ev('x', 'секрет'), ev(3, 'секрет', дата='вчера')], via='http',
                 actor={'sub': 'bus-1', 'kind': 'service'}, request_id='r1')
    assert out['accepted'] == 2
    assert [r['index'] for r in out['rejected']] == [2, 3]
    assert [(t, k) for t, k, _ in f.sink.sent] == [(funnel.TOPIC_READINGS, '1'), (funnel.TOPIC_JOURNAL, '2')]
    [row] = f.audit.events
    assert (row['event_type'], row['outcome'], row['actor_id'], row['request_id']) == \
        ('telemetry.rejected', 'denied', 'bus-1', 'r1')
    details = json.loads(row['details']) if isinstance(row['details'], str) else row['details']
    assert details['events'] == 4 and details['rejected'] == 2 and len(details['reasons']) == 2
    assert 'секрет' not in json.dumps(row, ensure_ascii=False)
    assert f.stats['accepted'] == 2 and f.stats['rejected'] == 2 and f.stats['packets'] == 1


def test_take_kafka_down_nothing_marked(f):
    f.sink.down = True
    with pytest.raises(funnel.Unavailable):
        f.take([ev(1)], via='http')
    assert f.channels.last == {} and f.stats['accepted'] == 0 and f.stats['unavailable'] == 1


def test_audit_failure_does_not_lose_packet(f):
    f.audit.send = lambda row: (_ for _ in ()).throw(RuntimeError('redis'))
    assert f.take([ev(1), 'мусор'], via='http')['accepted'] == 1


# ----- молчание --------------------------------------------------------------------------------
def test_channel_goes_silent_and_back(f):
    for t in (0, 10, 20):
        f.channels.seen(1, t)
    f.channels.seen(2, 20)                        # слышали один раз — обычного интервала нет, не судим
    assert f.sweep(now=20 + 3599) == []
    assert f.sweep(now=20 + 3601) == [1]
    assert f.sweep(now=20 + 7200) == []           # уже помечен
    topic, key, body = f.sink.sent[-1]
    assert (topic, key, body['status'], body['kind']) == (funnel.TOPIC_REFERENCE, 'channel-status:1', 'silent',
                                                          'channel.status')
    assert f.channels.snapshot()['silent_channels'][0]['channel'] == 1
    f.take([ev(1)], via='http')
    assert f.sink.sent[-1][1] == 'channel-status:1' and f.sink.sent[-1][2]['status'] == 'ok'
    assert f.channels.silent == {} and f.channels.gap[1] == 10     # молчание не раздуло интервал


def test_rare_channel_judged_by_its_own_interval():
    ch = funnel.Channels(3600)
    for t in (0, 7200, 14400):                    # раз в два часа
        ch.seen(5, t)
    assert ch.sweep(14400 + 4 * 3600)[0] == []
    assert ch.sweep(14400 + 4 * 7200 + 1)[0] == [5]


def test_bus_silence(f):
    f.take([ev(1)], via='http')
    t = f.channels.source_last
    assert f.channels.sweep(t + 3599)[1] is False
    assert f.channels.sweep(t + 3601)[1] is True
    assert f.channels.sweep(t + 7200)[1] is False  # один раз на переход
    assert f.channels.packet(t + 7300) is True and f.channels.source_silent is False


def test_status_write_failure_is_not_fatal(f):
    for t in (0, 10, 20):
        f.channels.seen(1, t)
    f.sink.down = True
    assert f.sweep(now=10_000) == [1]


# ----- HTTP ------------------------------------------------------------------------------------
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def token(key=KEY, **over):
    c = {'sub': 'bus-1', 'jti': 'j1', 'iss': 'auth-service', 'aud': 'api', 'token_type': 'access',
         'scope': 'telemetry.push', 'exp': int(time.time()) + 600}
    c.update(over)
    return jwt.encode({k: v for k, v in c.items() if v is not None}, key, algorithm='RS256')


def bearer(**over):
    return {'Authorization': f'Bearer {token(**over)}', 'X-Request-Id': 'rq'}


@pytest.fixture
def http(f):
    from fastapi.testclient import TestClient
    verifier = tfkit.Verifier(PEM, service_subs=['think-bus'])
    return TestClient(funnel.make_app(f, verifier, f.audit)), f


def test_http_accepts_packet(http):
    c, f = http
    r = c.post('/events', json={'events': [ev(1), ev(2, 'Норма')]}, headers=bearer())
    assert r.status_code == 202 and r.json() == {'accepted': 2, 'rejected': []}
    assert len(f.sink.sent) == 2 and f.audit.events == []
    r = c.post('/api/funnel/events', json=[ev(3)], headers=bearer())
    assert r.status_code == 202 and r.json()['accepted'] == 1


def test_http_service_sub_without_scope(http):
    c, _ = http
    assert c.post('/events', json=[ev()], headers=bearer(sub='think-bus', scope=None)).status_code == 202


def test_http_no_token(http):
    c, f = http
    r = c.post('/events', json=[ev()])
    assert r.status_code == 401 and r.headers['www-authenticate'] == 'Bearer'
    [row] = f.audit.events
    assert (row['event_type'], row['actor_kind'], row['object_id']) == ('token.refused', 'anonymous', '/events')
    assert f.sink.sent == []


def test_http_foreign_token(http):
    c, f = http
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r = c.post('/events', json=[ev()], headers={'Authorization': f'Bearer {token(key=other, jti="чужой")}'})
    assert r.status_code == 401
    [row] = f.audit.events
    assert row['event_type'] == 'token.refused' and 'чужой' in json.dumps(row, ensure_ascii=False)
    assert f.sink.sent == []


def test_http_expired_token_not_audited(http):
    c, f = http
    r = c.post('/events', json=[ev()], headers=bearer(exp=int(time.time()) - 60))
    assert r.status_code == 401 and f.audit.events == []


def test_http_user_without_push_right(http):
    c, f = http
    r = c.post('/events', json=[ev()], headers=bearer(sub='ivanov', scope='forecast.read', kind='user'))
    assert r.status_code == 403
    [row] = f.audit.events
    assert (row['event_type'], row['actor_kind'], row['actor_id']) == ('access.denied', 'user', 'ivanov')


def test_http_all_bad_is_422(http):
    c, f = http
    r = c.post('/events', json=[ev(channel='x')], headers=bearer())
    assert r.status_code == 422 and r.json()['rejected'][0]['index'] == 0
    assert f.audit.events[0]['event_type'] == 'telemetry.rejected'


def test_http_not_a_packet(http):
    c, f = http
    r = c.post('/events', json={'x': 1}, headers=bearer())
    assert r.status_code == 422
    assert f.audit.events[0]['event_type'] == 'telemetry.rejected'


def test_http_kafka_down_503(http):
    c, f = http
    f.sink.down = True
    r = c.post('/events', json=[ev()], headers=bearer())
    assert r.status_code == 503 and r.headers['retry-after'] == '5'


def test_http_health_open_status_closed(http):
    c, _ = http
    assert c.get('/health').json()['ok'] is True
    assert c.get('/api/funnel/status').status_code == 401
    s = c.get('/status', headers=bearer()).json()
    assert s['channels'] == 0 and s['source_silent'] is False


def test_http_dev_without_key(f):
    from fastapi.testclient import TestClient
    c = TestClient(funnel.make_app(f, None, f.audit))
    assert c.post('/events', json=[ev()]).status_code == 202


# ----- стенд: забор у эмулятора ----------------------------------------------------------------
class Stop:
    """Не спит: считает паузы и останавливает цикл после `n`."""

    def __init__(self, n):
        self.n, self.waits = n, []

    def is_set(self):
        return len(self.waits) >= self.n

    def wait(self, s):
        self.waits.append(s)
        return self.is_set()


def page(*cursors):
    return {'курсор': cursors[-1] if cursors else 0, 'событий': len(cursors),
            'события': [ev(c % 3 + 1, курсор=c) for c in cursors]}


def test_pull_follows_cursor_and_skips_pause_on_full_page(f):
    asked, answers = [], iter([page(1, 2), page(3), page(), page()])

    def get(url):
        asked.append(url)
        return next(answers)
    stop = Stop(3)
    funnel.pull(f, 'http://emu/', stop, interval=1, limit=2, get=get)
    assert asked[:3] == ['http://emu/events?cursor=0&limit=2', 'http://emu/events?cursor=2&limit=2',
                         'http://emu/events?cursor=3&limit=2']
    assert stop.waits[0] == 1 and f.stats['accepted'] == 3


def test_pull_restarted_emulator_read_from_start(f):
    asked = []

    def get(url):
        asked.append(url)
        if '/health' in url:
            return {'ok': True, 'курсор': 4}
        return page(100) if len(asked) == 1 else page()
    funnel.pull(f, 'http://emu', Stop(4), interval=15, get=get)
    assert 'http://emu/health' in asked
    assert asked[-1].startswith('http://emu/events?cursor=0&')


def test_pull_survives_emulator_down_with_backoff(f):
    def get(url):
        raise urllib.error.URLError('connection refused')
    stop = Stop(5)
    funnel.pull(f, 'http://emu', stop, interval=1, get=get)
    assert stop.waits == [2, 4, 8, 16, 32]


def test_pull_kafka_down_keeps_cursor(f):
    asked = []

    def get(url):
        asked.append(url)
        return page(1)
    f.sink.down = True
    funnel.pull(f, 'http://emu', Stop(2), interval=1, get=get)
    assert asked == ['http://emu/events?cursor=0&limit=5000'] * 2


# ----- запись ----------------------------------------------------------------------------------
def test_file_sink(tmp_path):
    s = funnel.FileSink(tmp_path / 'out' / 'f.jsonl')
    s.send([(funnel.TOPIC_READINGS, '1', {'значение_датчика': '0.4'})])
    assert json.loads((tmp_path / 'out' / 'f.jsonl').read_text(encoding='utf-8')) == \
        {'topic': funnel.TOPIC_READINGS, 'key': '1', 'value': {'значение_датчика': '0.4'}}


class Producer:
    made = []

    def __init__(self, conf):
        self.conf, self.msgs, self.err, self.left, self.full = conf, [], None, 0, False
        Producer.made.append(self)

    def produce(self, topic, key, value, on_delivery):
        if self.full:
            raise BufferError
        self.msgs.append((topic, key, value))
        on_delivery(self.err, None)

    def poll(self, t):
        return 0

    def flush(self, timeout):
        return self.left


@pytest.fixture
def kafka(monkeypatch):
    monkeypatch.setitem(sys.modules, 'confluent_kafka', types.SimpleNamespace(Producer=Producer))
    Producer.made.clear()
    return funnel.KafkaSink({'bootstrap.servers': 'k:9092'}), Producer.made


def test_kafka_sink(kafka):
    s, made = kafka
    s.send([(funnel.TOPIC_JOURNAL, '7', {'значение_датчика': 'Норма'})])
    p = made[0]
    assert p.conf['acks'] == 'all' and p.conf['enable.idempotence'] is True
    assert p.msgs == [(funnel.TOPIC_JOURNAL, b'7', '{"значение_датчика": "Норма"}'.encode())]


@pytest.mark.parametrize('break_it', ['err', 'left', 'full'])
def test_kafka_sink_unconfirmed(kafka, break_it):
    s, made = kafka
    p = made[0]
    p.err, p.left, p.full = ('ошибка' if break_it == 'err' else None), (1 if break_it == 'left' else 0), \
        break_it == 'full'
    with pytest.raises(funnel.Unavailable):
        s.send([(funnel.TOPIC_READINGS, '1', {})])


# ----- запуск ----------------------------------------------------------------------------------
def test_make_sink_off(monkeypatch, tmp_path):
    monkeypatch.setenv('TF_KAFKA_BOOTSTRAP', 'off')
    monkeypatch.setenv('TF_FUNNEL_OUT', str(tmp_path / 'x.jsonl'))
    assert isinstance(funnel.make_sink(), funnel.FileSink)


def test_make_sink_prod_needs_vault(monkeypatch):
    monkeypatch.setenv('TF_ENV', 'prod')
    monkeypatch.setenv('TF_KAFKA_BOOTSTRAP', 'k:9092')
    monkeypatch.setenv('TF_KAFKA_FUNNEL_PASSWORD', 'не-берётся-в-prod')
    with pytest.raises(tfkit.SecretError):
        funnel.make_sink()


def test_make_sink_dev_password(monkeypatch, kafka):
    monkeypatch.setenv('TF_KAFKA_BOOTSTRAP', 'k:9092')
    monkeypatch.setenv('TF_KAFKA_FUNNEL_PASSWORD', 'dev')
    funnel.make_sink()
    conf = Producer.made[-1].conf
    assert (conf['sasl.username'], conf['security.protocol']) == ('tf-funnel', 'SASL_PLAINTEXT')


def test_make_verifier(monkeypatch):
    assert funnel.make_verifier() is None                      # dev без ключа — открыто
    monkeypatch.setenv('TF_ENV', 'prod')
    v = funnel.make_verifier()
    assert v.jwks_url == 'http://tf-auth:8080/.well-known/jwks'
    monkeypatch.setenv('TF_ENV', 'dev')
    monkeypatch.setenv('TF_AUTH_PUBLIC_KEY', PEM.decode())
    monkeypatch.setenv('TF_FUNNEL_SERVICE_SUBS', 'think-bus, emu')
    v = funnel.make_verifier()
    assert v.jwks_url is None and v.service_subs == {'think-bus', 'emu'}
