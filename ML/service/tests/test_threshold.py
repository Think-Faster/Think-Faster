import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import make_settings
from threshold import Thresholds


def pool(n: int = 90 * 24, seed: int = 3) -> tuple:
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    p = rng.random(n)
    return h, p.astype(np.float32)


def quantile_of_window(h_hist, p_hist, h, share):
    lo, hi = np.searchsorted(h_hist, [h - 90 * 24 + 1, h + 1])
    if lo >= hi:
        lo, hi = np.searchsorted(h_hist, [h_hist[-1] - 90 * 24 + 1, h_hist[-1] + 1])
    return float(np.quantile(p_hist[lo:hi], 1 - share))


class ThresholdTest(unittest.TestCase):
    def _mk(self, d: str):
        tp = Path(d)
        st = make_settings(tp)
        score = {'fire': pool(), 'gas': pool(seed=4)}
        t = Thresholds(tp / 'history.parquet')
        return st, t, score

    def test_bootstrap_equals_retro_rolling(self):
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            h = 200 * 24 + 1
            t.bootstrap(scores, st, h)
            self.assertAlmostEqual(
                t.thresholds['fire'],
                quantile_of_window(scores['fire'][0], scores['fire'][1], h,
                                   st.share('fire')), places=7)

    def test_extend_advances_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 100)
            n0 = len(t.history('fire')[0])
            t.extend(101, {'fire': np.float32(0.99)}, st)
            self.assertEqual(len(t.history('fire')[0]), n0 + 1)
            self.assertTrue(0.0 < t.thresholds['fire'] < 1.0)

    def test_dump_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            tp = Path(d)
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 50)
            t.dump()
            t2 = Thresholds(tp / 'history.parquet')
            self.assertTrue(t2.load(st, 50))            # тот же h, что и bootstrap
            self.assertEqual(len(t2.history('fire')[0]), len(t.history('fire')[0]))
            self.assertAlmostEqual(t2.thresholds['fire'], t.thresholds['fire'], places=7)

    def test_empty_window_falls_back_without_crash(self):
        """Бутстрап впереди хвоста парка: порог из последних 90 суток, без выброса."""
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 10_000)             # окно не пересекается с историей
            self.assertTrue(0.0 < t.thresholds['fire'] < 1.0)

    def test_apply_settings_recomputes_with_new_share(self):
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 60)
            old = t.thresholds['fire']
            new = st.rebase({'fire': {'share': 0.04}}, by='x', reason='y')
            changed = t.apply_settings(new, 61)
            self.assertIn('fire', changed)
            self.assertNotEqual(t.thresholds['fire'], old)


if __name__ == '__main__':
    unittest.main()