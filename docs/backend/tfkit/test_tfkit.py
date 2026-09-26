import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import tfkit

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.public_key().public_bytes(serialization.Encoding.PEM,
                                    serialization.PublicFormat.SubjectPublicKeyInfo)


def token(**over):
    claims = {'sub': 'u1', 'jti': 'j1', 'iss': 'auth-service', 'aud': 'api',
              'token_type': 'access', 'exp': int(time.time()) + 600}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, KEY, algorithm='RS256')


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ('TF_ENV', 'VAULT_ADDR', 'VAULT_TOKEN', 'VAULT_TOKEN_FILE', 'VAULT_ROLE_ID', 'VAULT_SECRET_ID',
              'TF_REDIS_URL', 'TF_REDIS_PASSWORD', 'X_PASS', 'TF_VAULT_WAIT'):
        monkeypatch.delenv(v, raising=False)
    tfkit._cache.clear()
    tfkit._login.clear()


@pytest.fixture
def vault(monkeypatch):
    class Seen(list):
        good = {'t0'}      # действующие токены; тест может «просрочить» токен AppRole

    seen, issued = Seen(), []
    good = seen.good

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append((self.path, None))
            if self.path != '/v1/auth/approle/login' or body != {'role_id': 'r1', 'secret_id': 's1'}:
                self.send_response(400)
                self.end_headers()
                return
            tok = f'a{len(issued)}'
            issued.append(tok)
            good.add(tok)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'auth': {'client_token': tok}}).encode())

        def do_GET(self):
            seen.append((self.path, self.headers.get('X-Vault-Token')))
            data = {'/tf/kafka': {'model_password': 'p1'}, '/tf/redis': {'TF_REDIS_PASSWORD': 'r/p@1'}}
            key = next((k for k in data if self.path.endswith(k)), None)
            if self.headers.get('X-Vault-Token') not in good or key is None:
                self.send_response(403 if key else 404)
                self.end_headers()
                return
            body = json.dumps({'data': {'data': data[key], 'metadata': {}}}).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv('VAULT_ADDR', f'http://127.0.0.1:{srv.server_port}')
    monkeypatch.setenv('VAULT_TOKEN', 't0')
    yield seen
    srv.shutdown()


def test_secret_from_vault(vault):
    assert tfkit.secret('kafka', 'model_password') == 'p1'
    assert vault == [('/v1/secret/data/tf/kafka', 't0')]
    tfkit.secret('kafka', 'model_password')
    assert len(vault) == 1                                     # кэш процесса


def test_secret_missing_field_in_prod_names_path_only(vault, monkeypatch):
    monkeypatch.setenv('X_PASS', 'из-окружения')
    with pytest.raises(tfkit.SecretError) as e:
        tfkit.secret('kafka', 'other', 'X_PASS')
    assert 'secret/tf/kafka' in str(e.value) and 'p1' not in str(e.value)


def test_secret_dev_falls_back_to_env(vault, monkeypatch):
    monkeypatch.setenv('TF_ENV', 'dev')
    monkeypatch.setenv('X_PASS', 'из-окружения')
    assert tfkit.secret('rabbit', 'model_password', 'X_PASS') == 'из-окружения'


def test_secret_without_vault_env_only_in_dev(monkeypatch):
    monkeypatch.setenv('X_PASS', 'стенд')
    with pytest.raises(tfkit.SecretError):                     # prod: все секреты только из Vault
        tfkit.secret('rabbit', 'model_password', 'X_PASS')
    monkeypatch.setenv('TF_ENV', 'dev')
    assert tfkit.secret('rabbit', 'model_password', 'X_PASS') == 'стенд'
    with pytest.raises(tfkit.SecretError):
        tfkit.secret('rabbit', 'model_password')
    assert tfkit.secret('rabbit', 'model_password', required=False) is None


def test_vault_token_file(vault, monkeypatch, tmp_path):
    monkeypatch.delenv('VAULT_TOKEN')
    (tmp_path / 'tok').write_text('t0\n', encoding='utf-8')
    monkeypatch.setenv('VAULT_TOKEN_FILE', str(tmp_path / 'tok'))
    assert tfkit.secret('kafka', 'model_password') == 'p1'


def test_sealed_vault_waits_then_reads(monkeypatch):
    calls = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            if len(calls) < 3:                                 # запечатан: 503, пока не распечатают
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({'data': {'data': {'x': 'v'}}}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv('VAULT_ADDR', f'http://127.0.0.1:{srv.server_port}')
    monkeypatch.setenv('VAULT_TOKEN', 't0')
    monkeypatch.setattr(tfkit, 'VAULT_RETRY', 0.01)
    with pytest.raises(tfkit.SecretError, match='503'):       # без TF_VAULT_WAIT — сразу ошибка
        tfkit.secret('model', 'x')
    monkeypatch.setenv('TF_VAULT_WAIT', '5')
    assert tfkit.secret('model', 'x') == 'v' and len(calls) == 3
    srv.shutdown()


def test_vault_approle_login(vault, monkeypatch):
    # как docs/vault-entrypoint.sh think-infra: роль и секрет → токен, дальше чтение по токену
    monkeypatch.delenv('VAULT_TOKEN')
    monkeypatch.setenv('VAULT_ROLE_ID', 'r1')
    monkeypatch.setenv('VAULT_SECRET_ID', 's1')
    assert tfkit.secret('kafka', 'model_password') == 'p1'
    assert tfkit.secret('redis', 'TF_REDIS_PASSWORD') == 'r/p@1'
    assert vault == [('/v1/auth/approle/login', None), ('/v1/secret/data/tf/kafka', 'a0'),
                     ('/v1/secret/data/tf/redis', 'a0')]          # вход один раз на процесс


def test_vault_approle_relogin_on_expired_token(vault, monkeypatch):
    monkeypatch.delenv('VAULT_TOKEN')
    monkeypatch.setenv('VAULT_ROLE_ID', 'r1')
    monkeypatch.setenv('VAULT_SECRET_ID', 's1')
    tfkit.secret('kafka', 'model_password')
    vault.good.discard('a0')                                   # токен истёк
    tfkit._cache.clear()
    assert tfkit.secret('kafka', 'model_password') == 'p1'
    assert [p for p, _ in vault].count('/v1/auth/approle/login') == 2


def test_vault_approle_wrong_secret_id(vault, monkeypatch):
    monkeypatch.delenv('VAULT_TOKEN')
    monkeypatch.setenv('VAULT_ROLE_ID', 'r1')
    monkeypatch.setenv('VAULT_SECRET_ID', 'чужой')
    with pytest.raises(tfkit.SecretError, match='400 на вход AppRole') as e:
        tfkit.secret('kafka', 'model_password')
    assert 'чужой' not in str(e.value)


def test_redis_url_gets_password(vault, monkeypatch):
    assert tfkit.redis_url('redis://tf-redis:6379/0') == 'redis://:r%2Fp%401@tf-redis:6379/0'
    assert tfkit.redis_url('redis://:своё@tf-redis:6379') == 'redis://:своё@tf-redis:6379'
    assert tfkit.redis_url('') == '' and tfkit.redis_url(None) is None


def test_redis_url_without_password_stays(monkeypatch):
    assert tfkit.redis_url('redis://tf-redis:6379/0') == 'redis://tf-redis:6379/0'   # стенд без Vault
    monkeypatch.setenv('TF_ENV', 'dev')
    monkeypatch.setenv('TF_REDIS_PASSWORD', 'dev')
    assert tfkit.redis_url('redis://localhost:6379') == 'redis://:dev@localhost:6379'


def test_vault_forbidden_is_not_waited(vault, monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 't-other')
    monkeypatch.setenv('TF_VAULT_WAIT', '30')
    t = time.monotonic()
    with pytest.raises(tfkit.SecretError, match='403'):
        tfkit.secret('kafka', 'model_password')
    assert time.monotonic() - t < 2


def test_verify_think_auth_token():
    v = tfkit.Verifier(PEM)
    c = v.verify(token())
    assert c['sub'] == 'u1' and v.kind(c) == 'user'
    assert v.verify(token(token_type=None, typ='access', iss='tf-auth'))['sub'] == 'u1'


@pytest.mark.parametrize('over,why', [
    ({'token_type': 'refresh'}, 'не токен доступа'),
    ({'iss': 'evil'}, 'издатель'),
    ({'aud': 'other'}, 'не прошёл'),
    ({'exp': int(time.time()) - 5}, 'срок'),
])
def test_verify_refuses(over, why):
    with pytest.raises(tfkit.TokenError) as e:
        tfkit.Verifier(PEM).verify(token(**over))
    assert e.value.status == 401 and why in e.value.reason
    assert e.value.audit is (why != 'срок')        # протухший — не событие журнала (§6.2)


def test_verify_foreign_key_keeps_jti_for_audit():
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad = jwt.encode({'sub': 'u1', 'jti': 'j9', 'exp': int(time.time()) + 60}, other, algorithm='RS256')
    with pytest.raises(tfkit.TokenError) as e:
        tfkit.Verifier(PEM).verify(bad)
    assert e.value.status == 401 and e.value.jti == 'j9'


def test_scope_and_service_subs():
    v = tfkit.Verifier(PEM, service_subs=['svc-bff'])
    assert v.verify(token(scope='ml.read ml.write', kind='service'), 'ml.read')
    with pytest.raises(tfkit.TokenError) as e:
        v.verify(token(scope='ml.write'), 'ml.read')
    assert e.value.status == 403
    assert v.verify(token(sub='svc-bff'), 'ml.read')           # think-auth пока без scope
    assert v.kind(v.verify(token(sub='svc-bff'))) == 'service' and v.kind(v.verify(token())) == 'user'
    with pytest.raises(tfkit.TokenError):
        v.verify(token(sub='u1'), 'ml.read')


def test_key_from_jwks_raw_pem():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(PEM)

        def log_message(self, *a):
            pass

    srv = HTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        v = tfkit.Verifier(jwks_url=f'http://127.0.0.1:{srv.server_port}/.well-known/jwks')
        assert v.verify(token())['jti'] == 'j1'
    finally:
        srv.shutdown()
    with pytest.raises(tfkit.TokenError) as e:
        tfkit.Verifier(jwks_url='http://127.0.0.1:9/.well-known/jwks').verify(token())
    assert e.value.status == 503


class FakeRedis:
    def __init__(self, fail=False):
        self.rows, self.fail = [], fail

    def xadd(self, stream, fields, **kw):
        if self.fail:
            raise ConnectionError('down')
        self.rows.append((stream, json.loads(fields['event'])))


def test_audit_event_shape():
    a = tfkit.Audit('tf-model', redis_url='')
    a._r = FakeRedis()
    row = a.event('forecast.muted', actor_kind='user', actor_id='u1', object_type='forecast',
                  object_id=17, area_id=3, details={'reason': 'работы по графику'})
    stream, got = a._r.rows[0]
    assert stream == 'audit' and got == row
    assert got['service'] == 'tf-model' and got['object_id'] == '17' and got['outcome'] == 'success'


def test_audit_refuses_secrets():
    with pytest.raises(ValueError):
        tfkit.Audit('tf-model', redis_url='').event('x', details={'token': 'abc'})


def test_audit_spools_and_flushes(tmp_path):
    a = tfkit.Audit('tf-notify', redis_url='', spool=tmp_path / 'audit.jsonl')
    a._r = FakeRedis(fail=True)
    a.event('notify.failed', 'error', details={'channel': 'email'})
    assert (tmp_path / 'audit.jsonl').exists() and a._r is None
    a._r = FakeRedis()
    a.event('notify.sent', details={'channel': 'email'})
    assert [r['event_type'] for _, r in a._r.rows] == ['notify.failed', 'notify.sent']
    assert not (tmp_path / 'audit.jsonl').exists()


def test_audit_flush_without_new_events(tmp_path):
    """Файл досылается и без нового события — по такту сервиса; при лежащем Redis файл цел."""
    a = tfkit.Audit('tf-model', redis_url='', spool=tmp_path / 'audit.jsonl')
    a.event('forecast.muted', details={'reason': 'works'})
    a._r = FakeRedis(fail=True)
    assert a.flush() == 0 and (tmp_path / 'audit.jsonl').exists()
    a._r = FakeRedis()
    r = a._r
    assert a.flush() == 1 and not (tmp_path / 'audit.jsonl').exists()
    assert [row['event_type'] for _, row in r.rows] == ['forecast.muted']


def test_request_log_one_row_per_request(tmp_path):
    """§6.1: шаблон маршрута, метод, код, время, кто спросил; проба живости не пишется; при лежащем
    Redis строка — в свой файл рядом с файлом событий."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    app = FastAPI()

    @app.get('/items/{item_id}')
    def item(item_id: int):
        if item_id == 0:
            raise HTTPException(404, 'нет')
        return {'id': item_id}

    @app.get('/api/x/health')
    def health():
        return {'ok': True}

    rl = tfkit.request_log(app, tfkit.Audit('tf-model', redis_url='', spool=tmp_path / 'audit.jsonl'),
                           tfkit.Verifier(PEM))
    rl.out._r = FakeRedis()
    c = TestClient(app)
    c.get('/items/5?q=1', headers={'Authorization': f'Bearer {token(scope="ml.read")}', 'X-Request-ID': 'r-1'})
    c.get('/items/0', headers={'Authorization': 'Bearer not-a-jwt'})
    c.get('/nowhere/7')
    c.get('/api/x/health')
    rl.q.join()
    rows = [r for s, r in rl.out._r.rows if s == 'audit:requests']
    assert [(r['method'], r['route'], r['status']) for r in rows] == [
        ('GET', '/items/{item_id}', 200), ('GET', '/items/{item_id}', 404), ('GET', '(нет маршрута)', 404)]
    assert (rows[0]['actor_kind'], rows[0]['actor_id'], rows[0]['request_id']) == ('service', 'u1', 'r-1')
    assert (rows[1]['actor_kind'], rows[1]['actor_id']) == ('anonymous', None)
    assert all(isinstance(r['duration_ms'], int) and r['service'] == 'tf-model' for r in rows)
    rl.out._r = FakeRedis(fail=True)
    c.get('/items/6')
    rl.q.join()
    assert (tmp_path / 'audit-requests.jsonl').exists() and not (tmp_path / 'audit.jsonl').exists()
