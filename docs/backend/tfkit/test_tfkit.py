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
    for v in ('TF_ENV', 'VAULT_ADDR', 'VAULT_TOKEN', 'VAULT_TOKEN_FILE', 'TF_REDIS_URL', 'X_PASS'):
        monkeypatch.delenv(v, raising=False)
    tfkit._cache.clear()


@pytest.fixture
def vault(monkeypatch):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get('X-Vault-Token')))
            if self.headers.get('X-Vault-Token') != 't0' or not self.path.endswith('/tf/kafka'):
                self.send_response(403 if self.path.endswith('/tf/kafka') else 404)
                self.end_headers()
                return
            body = json.dumps({'data': {'data': {'model_password': 'p1'}, 'metadata': {}}}).encode()
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
