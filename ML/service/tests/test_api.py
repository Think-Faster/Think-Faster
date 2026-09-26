"""Ручки 13.1 и токены 13.2: RS256 think-auth, права техучётки и пользователя, аудит отказов."""
import time
import unittest
from datetime import datetime

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import api
import tfkit

KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PEM = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def token(key=KEY, **over):
    claims = {'sub': 'u1', 'jti': 'j1', 'iss': 'auth-service', 'aud': 'api', 'token_type': 'access',
              'exp': int(time.time()) + 600}
    claims.update(over)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm='RS256')


BFF = token(sub='svc-bff', jti='jb', scope='ml.read')


class FakeAudit:
    def __init__(self):
        self.events = []

    def event(self, event_type, outcome='success', **kw):
        self.events.append({'event_type': event_type, 'outcome': outcome, **kw})


class FakeService:
    last_now = datetime(2026, 1, 4, 12)

    def __init__(self):
        self.audit, self.asked = FakeAudit(), []

    def health(self):
        return {'ok': True, 'clock': 'live', 'last_hour': '2026-01-04T12:00+03:00'}

    def status(self):
        return {'env': 'test'}

    def estimate(self, tp, share):
        if not 0 < share < 1:
            raise ValueError('доля — в (0, 1)')
        return {'type': tp, 'share': share}

    def forecast(self, object_id):
        return {'object_id': object_id} if object_id == 5122 else None

    def forecast_history(self, object_id, tp, a, b):
        self.asked.append((a, b))
        return [{'hour_end': '2026-01-04T12:00+03:00', 'status': 'MUTED'}]


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.svc = FakeService()
        self.web = TestClient(api.create_app(self.svc, tfkit.Verifier(PEM, service_subs=['svc-old'])))

    def get(self, path, tok=None, **params):
        return self.web.get(path, params=params, headers={'Authorization': f'Bearer {tok}'} if tok else {})

    def refused(self):
        return [(e['event_type'], e['actor_kind'], e['details']['jti']) for e in self.svc.audit.events]

    def test_health_is_open_on_both_prefixes(self):
        for p in ('/health', '/api/ml/health'):
            r = self.get(p)
            self.assertEqual((r.status_code, r.json()['ok']), (200, True))

    def test_service_account_reads_everything(self):
        self.assertEqual(self.get('/api/ml/forecast', BFF, object_id=5122).json(), {'object_id': 5122})
        self.assertEqual(self.get('/forecast', BFF, object_id=1).status_code, 404)
        self.assertEqual(self.get('/status', BFF).json(), {'env': 'test'})
        old = token(sub='svc-old', jti='jo')                     # think-auth пока без scope — по списку sub
        self.assertEqual(self.get('/forecast', old, object_id=5122).status_code, 200)
        self.assertEqual(self.svc.audit.events, [])

    def test_user_token_only_on_admin_routes(self):
        user = token()
        self.assertEqual(self.get('/status', user).status_code, 200)
        self.assertEqual(self.get('/estimate', user, type='fire', share=0.02).json()['share'], 0.02)
        r = self.get('/forecast', user, object_id=5122)
        self.assertEqual(r.status_code, 403)
        ev = self.svc.audit.events[-1]
        self.assertEqual((ev['event_type'], ev['actor_kind'], ev['actor_id'], ev['object_id']),
                         ('access.denied', 'user', 'u1', '/forecast'))

    def test_bad_tokens_are_401_with_jti_only(self):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cases = [(None, None), (token(jti='j2', exp=int(time.time()) - 5), None),
                 (token(key=other, jti='j3'), 'j3'), (token(jti='j4', token_type='refresh'), 'j4'),
                 (token(jti='j5', iss='evil'), 'j5')]
        for tok, _ in cases:
            self.assertEqual(self.get('/status', tok).status_code, 401)
        self.assertEqual([e[0] for e in self.refused()], ['token.refused'] * len(cases))
        self.assertTrue(all(e[1] == 'anonymous' for e in self.refused()))
        self.assertEqual([e[2] for e in self.refused()][2:], ['j3', 'j4', 'j5'])
        text = repr(self.svc.audit.events)
        self.assertNotIn('eyJ', text)                             # сам токен в аудит не попадает

    def test_service_without_scope_is_403(self):
        r = self.get('/history', token(sub='svc-x', jti='j6', scope='ml.write'), object_id=1, type='gas')
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.refused(), [('access.denied', 'service', 'j6')])

    def test_history_period(self):
        r = self.get('/history', BFF, object_id=7, type='flood')
        self.assertEqual(r.json()['hours'][0]['status'], 'MUTED')
        a, b = self.svc.asked[-1]
        self.assertEqual((b - a).days, 7)                         # по умолчанию неделя до последнего часа
        r = self.get('/history', BFF, object_id=7, type='flood', **{'from': '2026-01-01T00:00:00+03:00',
                                                                    'to': '2026-01-02T00:00:00Z'})
        self.assertEqual(self.svc.asked[-1], (datetime(2026, 1, 1), datetime(2026, 1, 2, 3)))
        for bad in ({'type': 'пожар'}, {'type': 'gas', 'from': 'вчера'},
                    {'type': 'gas', 'from': '2026-01-02T00:00', 'to': '2026-01-01T00:00'},
                    {'type': 'gas', 'from': '2025-01-01T00:00', 'to': '2026-01-01T00:00'}):
            self.assertEqual(self.get('/history', BFF, object_id=7, **bad).status_code, 400, bad)

    def test_estimate_bad_share_is_400(self):
        self.assertEqual(self.get('/estimate', BFF, type='fire', share=2).status_code, 400)

    def test_dev_without_key_is_open(self):
        web = TestClient(api.create_app(self.svc, None))
        self.assertEqual(web.get('/api/ml/forecast', params={'object_id': 5122}).status_code, 200)


if __name__ == '__main__':
    unittest.main()
