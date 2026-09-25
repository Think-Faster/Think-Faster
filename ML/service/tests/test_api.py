import unittest
from pathlib import Path
import tempfile

import api

import svc as config
from helpers import make_settings
import numpy as np


class FakeHistory:
    """Минимум для estimate_view: история оценок парка текущей версии (M5)."""

    def __init__(self, scores: dict):
        self.scores = scores

    def history(self, tp: str):
        h = np.arange(len(self.scores[tp]), dtype=np.int64)
        return h, np.asarray(self.scores[tp], np.float32)


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


class EstimateViewTest(unittest.TestCase):
    """M12: ядро /api/ml/estimate — доля часа → тревог в сутки по парку (сверено с calib)."""

    def _state(self, d, scores):
        st = api.State(settings=make_settings(Path(d)))
        st.history = FakeHistory(scores)
        return st

    def test_returns_threshold_and_per_day(self):
        with tempfile.TemporaryDirectory() as d:
            hist = np.random.default_rng(3).random(60 * 24)
            res = api.estimate_view(self._state(d, {'fire': hist, 'gas': hist}), 'fire', 0.03)
            self.assertAlmostEqual(res['threshold'], float(np.quantile(hist, 0.97)))
            expected = int((hist >= res['threshold']).sum()) / 60.0
            self.assertAlmostEqual(res['alarms_per_day'], expected)
            self.assertEqual(res['window_days'], 60.0)

    def test_rejects_bad_share(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._state(d, {'fire': np.random.random(24)})
            with self.assertRaises(Exception) as e:
                api.estimate_view(st, 'fire', 1.5)
            self.assertEqual(e.exception.status_code, 422)

    def test_unknown_type_422(self):
        with tempfile.TemporaryDirectory() as d:
            st = self._state(d, {'fire': np.random.random(24)})
            with self.assertRaises(Exception) as e:
                api.estimate_view(st, 'whatever', 0.03)
            self.assertEqual(e.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()