"""Тесты сервиса уведомлений без сети: почта, Telegram, Vault и брокер подменены.

    python -m pytest -q -p no:cacheprovider test_notify.py

Файл .env не читается (TF_NOTIFY_ENV_FILE=''), ничего никуда не отправляется.
"""
import importlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
os.environ['TF_NOTIFY_ENV_FILE'] = ''
os.environ['TF_ENV'] = 'dev'
for _k in ('VAULT_ADDR', 'VAULT_TOKEN', 'VAULT_TOKEN_FILE', 'TF_REDIS_URL', 'TF_RABBIT_URL', 'SMTP_USER',
           'SMTP_PASSWORD', 'TELEGRAM_BOT_TOKEN', 'JWT_PUBLIC_KEY', 'SERVICE_SUBS', 'TF_AUTH_JWKS'):
    os.environ.pop(_k, None)
sys.path.insert(0, str(HERE))

import jwt  # noqa: E402

import config  # noqa: E402
import mailer  # noqa: E402
import rabbit  # noqa: E402
import telegram_bot  # noqa: E402
import tfkit  # noqa: E402

SVC = '00000000-0000-4000-8000-00000000b0ff'       # техучётка BFF без scope (Н10)
USER = '00000000-0000-4000-8000-000000000042'
NID = str(uuid.uuid4())
TEXT = 'Канал 196771 «Газовая охрана». Заявка 1042.'


class Recorder:
    def __init__(self):
        self.events = []

    def event(self, event_type, outcome='success', **kw):
        self.events.append((event_type, outcome, kw))

    def last(self):
        return self.events[-1]


def private_key():
    from cryptography.hazmat.primitives.asymmetric import rsa
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope='module')
def keys(tmp_path_factory):
    from cryptography.hazmat.primitives import serialization
    priv = private_key()
    path = tmp_path_factory.mktemp('keys') / 'auth.pem'
    path.write_bytes(priv.public_key().public_bytes(serialization.Encoding.PEM,
                                                    serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv, path


@pytest.fixture(scope='module')
def app(keys):
    env = {'JWT_PUBLIC_KEY': str(keys[1]), 'SERVICE_SUBS': SVC, 'SMTP_USER': 'bot@example.com',
           'TELEGRAM_BOT_TOKEN': '123:fake'}
    os.environ.update(env)
    try:
        main = importlib.reload(sys.modules['main']) if 'main' in sys.modules else importlib.import_module('main')
        yield main
    finally:
        for k in env:
            os.environ.pop(k, None)


@pytest.fixture
def client(app, monkeypatch):
    from fastapi.testclient import TestClient
    rec = Recorder()
    monkeypatch.setattr(app, 'audit', rec)
    c = TestClient(app.app)
    c.rec = rec
    return c


def token(priv, **claims):
    body = {'sub': SVC, 'iss': 'tf-auth', 'aud': 'api', 'typ': 'access', 'jti': 'j1',
            'exp': int(time.time()) + 600, 'scope': 'notify.send'}
    body.update(claims)
    return jwt.encode({k: v for k, v in body.items() if v is not None}, priv, algorithm='RS256')


def bearer(t):
    return {'Authorization': f'Bearer {t}', 'X-Request-ID': 'req-1'}


MAIL = {'subject': 'Загазованность', 'text': TEXT, 'to': ['a@example.com', 'b@example.com'], 'ticket_id': 1042}


# ----- ручки: токен ----------------------------------------------------------------------------
def test_no_token_is_refused_and_audited(client):
    r = client.post('/mail', json=MAIL)
    assert r.status_code == 401
    event, outcome, kw = client.rec.last()
    assert (event, outcome, kw['actor_kind'], kw['object_id']) == ('token.refused', 'denied', 'anonymous', '/mail')


def test_expired_token_is_401_without_event(client, keys):
    r = client.post('/mail', json=MAIL, headers=bearer(token(keys[0], exp=int(time.time()) - 30)))
    assert r.status_code == 401
    assert client.rec.events == []


def test_foreign_signature_is_audited_with_jti(client):
    r = client.post('/mail', json=MAIL, headers=bearer(token(private_key(), jti='j-bad')))
    assert r.status_code == 401
    event, _, kw = client.rec.last()
    assert event == 'token.refused' and kw['details']['jti'] == 'j-bad' and kw['request_id'] == 'req-1'


@pytest.mark.parametrize('claims, kind', [({'sub': USER, 'scope': None}, 'user'),
                                          ({'scope': 'ml.read'}, 'service')])
def test_without_right_is_403(client, keys, claims, kind):
    r = client.post('/mail', json=MAIL, headers=bearer(token(keys[0], **claims)))
    assert r.status_code == 403
    event, _, kw = client.rec.last()
    assert (event, kw['actor_kind']) == ('access.denied', kind)


def test_think_auth_token_with_aud_and_no_scope(client, keys, monkeypatch):
    """think-auth кладёт aud и не кладёт scope: техучётка из SERVICE_SUBS проходит (Н10)."""
    monkeypatch.setattr(mailer, 'send', lambda s, to, subj, text: {})
    t = token(keys[0], scope=None, token_type='access', typ=None, iss='auth-service')
    r = client.post('/mail', json=MAIL, headers=bearer(t))
    assert r.status_code == 200, r.text


# ----- ручки: отправка и журнал ----------------------------------------------------------------
def test_mail_sent_is_audited_without_text_and_addresses(client, keys, monkeypatch):
    monkeypatch.setattr(mailer, 'send', lambda s, to, subj, text: {'b@example.com': '550 5.1.1 <b@example.com>'})
    r = client.post('/mail', json=MAIL, headers=bearer(token(keys[0])))
    assert r.status_code == 200 and r.json()['sent'] == ['a@example.com']
    event, outcome, kw = client.rec.last()
    assert (event, outcome, kw['object_type'], kw['object_id']) == ('notify.sent', 'success', 'ticket', 1042)
    assert kw['details'] == {'channel': 'email', 'via': 'http', 'subject': 'Загазованность', 'recipients': 2,
                             'sent': 1, 'failed': 1}
    assert '@' not in json.dumps(kw, ensure_ascii=False) and 'Канал' not in json.dumps(kw, ensure_ascii=False)


def test_mail_server_down(client, keys, monkeypatch):
    def down(*a):
        raise mailer.MailError(503, 'почтовый сервер smtp.example.com:587 недоступен')
    monkeypatch.setattr(mailer, 'send', down)
    r = client.post('/mail', json=MAIL, headers=bearer(token(keys[0])))
    assert r.status_code == 503
    event, outcome, kw = client.rec.last()
    assert (event, outcome, kw['details']['reason']) == ('notify.failed', 'error', '503')


def test_telegram_nobody_reached(client, keys, monkeypatch):
    monkeypatch.setattr(telegram_bot, 'send', lambda s, to, subj, text: ([], {'5': 'Forbidden'}))
    r = client.post('/telegram', json={'subject': 'x', 'text': 'y', 'to': [5]}, headers=bearer(token(keys[0])))
    assert r.status_code == 422
    event, _, kw = client.rec.last()
    assert event == 'notify.failed' and kw['object_type'] is None and kw['details']['failed'] == 1


def test_health(client):
    h = client.get('/health').json()
    assert (h['env'], h['token_check'], h['queues']) == ('dev', True, [])


# ----- ключ проверки токенов -------------------------------------------------------------------
def test_verifier_open_only_in_dev(app):
    assert app.make_verifier(config.Settings(tf_env='dev', jwt_public_key=None)) is None
    v = app.make_verifier(config.Settings(tf_env='prod', jwt_public_key=None))
    assert v.jwks_url == 'http://tf-auth:8080/.well-known/jwks'
    v = app.make_verifier(config.Settings(tf_env='dev', jwt_public_key=None, tf_auth_jwks='http://a/jwks'))
    assert v.jwks_url == 'http://a/jwks'


# ----- настройки и секреты ---------------------------------------------------------------------
def test_prod_takes_secrets_only_from_vault(monkeypatch):
    monkeypatch.setenv('SMTP_USER', 'leak@example.com')
    assert config.load(tf_env='prod').smtp_user == ''
    assert config.load(tf_env='dev').smtp_user == 'leak@example.com'
    vault = {('app/tf-notify', 'TF_NOTIFY_SMTP_USER'): 'bot@example.com',
             ('app/tf-notify', 'TF_NOTIFY_TELEGRAM_BOT_TOKEN'): '1:v'}
    monkeypatch.setattr(config.tfkit, 'secret', lambda p, f, required=True: vault.get((p, f)))
    s = config.load(tf_env='prod')
    assert (s.smtp_user, s.telegram_bot_token.get_secret_value()) == ('bot@example.com', '1:v')


def test_rabbit_password():
    with pytest.raises(tfkit.SecretError):
        config.rabbit_password(config.Settings(tf_env='prod', rabbit_password='dev-only'), 'email')
    assert config.rabbit_password(config.Settings(tf_env='dev', rabbit_password='p'), 'telegram') == 'p'


def test_rabbit_user_per_channel():
    s = config.Settings(tf_env='dev', tf_rabbit_url='amqp://tf-rabbit:5672/tf', rabbit_password='p')
    p = rabbit.parameters(s, 'telegram')
    assert (p.credentials.username, p.virtual_host) == ('tf-notify-telegram', 'tf')


# ----- очереди ---------------------------------------------------------------------------------
def settings(**over):
    base = dict(tf_env='dev', smtp_user='bot@example.com', telegram_bot_token='123:x', max_recipients=2,
                notify_retries=5)
    return config.Settings(**{**base, **over})


def notice(**over):
    n = {'schema': 1, 'notice_id': NID, 'ticket_id': 1042, 'kind': 'fact', 'subject': 'Загазованность:\nобъект 5',
         'text': TEXT, 'to': {'emails': ['A@Example.com', 'b@example.com', 'c@example.com'], 'chat_ids': [1, '@ch']},
         'request_id': 'e81b07c4f2a9'}
    n.update(over)
    return json.dumps(n, ensure_ascii=False).encode()


class Mail:
    """Почтовый сервер: `plan` — что вернуть на каждый вызов (dict отказов или MailError)."""

    def __init__(self, *plan):
        self.plan, self.calls = list(plan), []

    def __call__(self, s, to, subject, text):
        self.calls.append(list(to))
        step = self.plan.pop(0) if self.plan else {}
        if isinstance(step, Exception):
            raise step
        return step


def consumer(channel='email', mail=None, tg=None, sent=None, **over):
    rec = Recorder()
    c = rabbit.Consumer(channel, settings(**over), rec, sent or rabbit.Sent(), send_mail=mail or Mail(),
                        send_telegram=tg)
    return c, rec


@pytest.mark.parametrize('body, why', [
    (b'{', 'не JSON'), (notice(notice_id='42'), 'notice_id'), (notice(to={'emails': []}), 'нет адресов'),
    (notice(to={'emails': ['не адрес']}), 'с ошибкой'), (notice(subject=' '), 'тема'),
])
def test_bad_message_goes_to_dlq(body, why):
    c, rec = consumer()
    assert c.decide(body) == rabbit.REJECT
    event, outcome, kw = rec.last()
    assert (event, outcome) == ('notify.failed', 'error') and why in kw['details']['reason']


def test_telegram_limit_is_permanent():
    c, rec = consumer('telegram', tg=lambda *a, **k: pytest.fail('не должно уйти'))
    assert c.decide(notice(text='x' * 4090)) == rabbit.REJECT


def test_mail_in_chunks_and_audit_has_no_addresses():
    mail = Mail({}, {})
    c, rec = consumer(mail=mail)
    assert c.decide(notice()) == rabbit.ACK
    assert mail.calls == [['A@example.com', 'b@example.com'], ['c@example.com']]
    event, outcome, kw = rec.last()
    assert (event, outcome, kw['object_type'], kw['object_id'], kw['request_id']) == \
        ('notify.sent', 'success', 'ticket', '1042', 'e81b07c4f2a9')
    assert kw['details'] == {'channel': 'email', 'sent': 3, 'failed': 0, 'attempt': 1, 'notice_id': NID,
                             'kind': 'fact', 'subject': 'Загазованность: объект 5', 'recipients': 3}
    dump = json.dumps(kw, ensure_ascii=False)
    assert '@' not in dump and 'Канал' not in dump


def test_retry_sends_only_the_rest():
    mail = Mail({}, mailer.MailError(503, 'недоступен'), {})
    c, rec = consumer(mail=mail)
    assert c.decide(notice()) == rabbit.RETRY
    assert c.delay(notice()) == rabbit.BACKOFF[0] and rec.events == []
    assert c.decide(notice(), {'x-delivery-count': 1}) == rabbit.ACK
    assert mail.calls[-1] == ['c@example.com']
    assert rec.last()[2]['details']['sent'] == 3 and rec.last()[2]['details']['attempt'] == 2


def test_refused_chunk_is_permanent():
    refused = mailer.MailError(422, 'отверг всех', {'A@example.com': '550 x', 'b@example.com': '550 y'})
    c, rec = consumer(mail=Mail(refused, {}))
    assert c.decide(notice()) == rabbit.ACK
    event, _, kw = rec.last()
    assert (event, kw['details']['sent'], kw['details']['failed']) == ('notify.sent', 1, 2)


def test_nobody_reached_goes_to_dlq():
    c, rec = consumer(mail=Mail(*[mailer.MailError(422, 'отверг всех')] * 2))
    assert c.decide(notice()) == rabbit.REJECT
    assert rec.last()[0] == 'notify.failed'


def test_last_attempt_rejects():
    c, rec = consumer(mail=Mail(*[mailer.MailError(503, 'недоступен')] * 9))
    assert c.decide(notice(), {'x-delivery-count': 4}) == rabbit.REJECT
    event, _, kw = rec.last()
    assert event == 'notify.failed' and kw['details']['attempt'] == 5 and kw['details']['reason'] == 'недоступен'


def test_duplicate_delivery_is_not_sent_again():
    class Redis:                       # общий Redis переживает перезапуск сервиса
        def __init__(self):
            self.sets = {}

        def smembers(self, k):
            return {x.encode() for x in self.sets.get(k, set())}

        def pipeline(self):
            outer = self

            class P:
                def sadd(self, k, *v):
                    outer.sets.setdefault(k, set()).update(v)

                def expire(self, k, ttl):
                    assert ttl == rabbit.DAY

                def execute(self):
                    pass
            return P()

    r = Redis()
    c, _ = consumer(mail=Mail(), sent=rabbit.Sent(r))
    assert c.decide(notice()) == rabbit.ACK
    again = Mail()
    c2, rec2 = consumer(mail=again, sent=rabbit.Sent(r))
    assert c2.decide(notice()) == rabbit.ACK and again.calls == [] and rec2.events == []


def test_sent_survives_redis_outage():
    class Down:
        def smembers(self, k):
            raise ConnectionError

        def pipeline(self):
            raise ConnectionError

    s = rabbit.Sent(Down())
    s.add(NID, 'email', ['a@example.com'])
    assert s.get(NID, 'email') == {'a@example.com'}


def test_telegram_later_is_retried_alone():
    calls = []

    def tg(s, chats, subject, text, later=None):
        calls.append(list(chats))
        if len(calls) == 1:
            later.append('@ch')
            return ['1'], {'@ch': 'Too Many Requests — Telegram ограничил частоту отправки'}
        return list(chats), {}

    c, rec = consumer('telegram', tg=tg)
    assert c.decide(notice()) == rabbit.RETRY
    assert c.decide(notice(), {'x-delivery-count': 1}) == rabbit.ACK
    assert calls == [['1', '@ch'], ['@ch']] and rec.last()[2]['details']['sent'] == 2


def test_telegram_down_retries_everyone():
    def tg(*a, **k):
        raise telegram_bot.TelegramError(503, 'Telegram недоступен')
    c, rec = consumer('telegram', tg=tg)
    assert c.decide(notice()) == rabbit.RETRY and rec.events == []


def test_reason_has_no_address():
    assert rabbit._first({'a@example.com': '550 5.1.1 <a@example.com> unknown'}, ['a@example.com']) == '550'


def test_on_message_answers_broker():
    class Ch:
        def __init__(self):
            self.log = []

        def basic_ack(self, tag):
            self.log.append(('ack', tag))

        def basic_reject(self, tag, requeue):
            self.log.append(('reject', tag, requeue))

        def basic_nack(self, tag, requeue):
            self.log.append(('nack', tag, requeue))

    class Conn:
        def __init__(self):
            self.delays = []

        def call_later(self, delay, fn):
            self.delays.append(delay)
            fn()

    method, props = SimpleNamespace(delivery_tag=7), SimpleNamespace(headers=None)
    for mail, want in ((Mail(), ('ack', 7)), (Mail(mailer.MailError(503, 'x')), ('nack', 7, True))):
        c, _ = consumer(mail=mail)
        ch, conn = Ch(), Conn()
        c.on_message(conn, ch, method, props, notice(notice_id=str(uuid.uuid4())))
        assert ch.log == [want]
    c, _ = consumer()
    ch = Ch()
    c.on_message(Conn(), ch, method, props, b'{')
    assert ch.log == [('reject', 7, False)]


def test_start_reads_only_configured_channels(monkeypatch):
    started = []
    monkeypatch.setattr(rabbit, 'parameters', lambda s, ch: ch)
    monkeypatch.setattr(rabbit, 'run', lambda c, stop, p: started.append(p))
    assert rabbit.start(settings(), Recorder(), threading.Event()) == []        # dev без адреса — выключено
    s = settings(tf_rabbit_url='amqp://tf-rabbit:5672/tf', telegram_bot_token='')
    threads = rabbit.start(s, Recorder(), threading.Event())
    for t in threads:
        t.join(5)
    assert [t.name for t in threads] == ['rabbit-email'] and started == ['email']
