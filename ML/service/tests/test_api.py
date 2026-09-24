import unittest

import api

import svc as config


class JwtTest(unittest.TestCase):
    def test_sign_verify_roundtrip(self):
        tok = api.make_token({'sub': 'ingest', 'scope': 'ml'})
        claims = api.verify_token(tok)
        self.assertEqual(claims['sub'], 'ingest')

    def test_accepts_bytes_and_plain(self):
        tok = api.make_token({'sub': 'x'})
        self.assertEqual(api.verify_token(tok.encode())['sub'], 'x')

    def test_tampered_rejected(self):
        tok = api.make_token({'sub': 'x'})
        head, body, sig = tok.split('.')
        bad = f'{head}.{api._b64(b"rgbtamper")}.{sig}'
        self.assertIsNone(api.verify_token(bad))
        self.assertIsNone(api.verify_token(tok[:-4] + 'AAAA'))

    def test_wrong_secret_rejected(self):
        tok = api.make_token({'sub': 'x'})
        self.assertIsNone(api.verify_token(tok, secret='other-secret'))

    def test_expired_rejected(self):
        tok = api.make_token({'sub': 'x'}, ttl_hours=-1)
        self.assertIsNone(api.verify_token(tok))

    def test_exp_in_past_rejected(self):
        tok = api.make_token({'sub': 'x'}, ttl_hours=-1)
        self.assertIsNone(api.verify_token(tok))


if __name__ == '__main__':
    unittest.main()