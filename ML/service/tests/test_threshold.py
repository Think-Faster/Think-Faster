import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

import features as ft
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
            hh0, pp0 = scores['fire']
            cut = np.searchsorted(hh0, 2200 - 90 * 24)      # окно 90 суток обрезает хвост истории
            t.extend(2200, {'fire': np.float32(0.99)}, st)
            hh, pp = t.history('fire')
            np.testing.assert_array_equal(hh, np.concatenate([hh0[cut:], [2200]]))
            np.testing.assert_array_equal(pp, np.concatenate([pp0[cut:], [np.float32(0.99)]]))
            self.assertTrue(0.0 < t.thresholds['fire'] < 1.0)

    def test_extend_appends_row_per_object(self):
        """B4: оценки часа — массив на объект; час повторяется столько же раз (как mix_history)."""
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 100)
            hh0, pp0 = scores['fire']
            cut = np.searchsorted(hh0, 2200 - 90 * 24)
            v = np.linspace(0.1, 0.9, 12, dtype=np.float32)
            t.extend(2200, {'fire': v}, st)
            hh, pp = t.history('fire')
            np.testing.assert_array_equal(
                hh, np.concatenate([hh0[cut:], np.full(12, 2200, hh0.dtype)]))
            np.testing.assert_array_equal(pp[: len(hh0) - cut], pp0[cut:])
            np.testing.assert_array_equal(pp[-12:], v)

    def test_extend_same_hour_replaces_rows(self):
        """Повторный такт того же часа заменяет его блок, не плодит строки парка."""
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 100)
            v1 = np.linspace(0.2, 0.6, 5, dtype=np.float32)
            v2 = np.linspace(0.4, 0.8, 5, dtype=np.float32)
            t.extend(2200, {'fire': v1}, st)
            hh1, pp1 = t.history('fire')
            t.extend(2200, {'fire': v2}, st)
            hh, pp = t.history('fire')
            np.testing.assert_array_equal(hh, hh1)        # длина и часы не изменились
            np.testing.assert_array_equal(pp[:-5], pp1[:-5])
            np.testing.assert_array_equal(pp[-5:], v2)

    def test_extend_hour_covered_by_bootstrap_is_not_duplicated(self):
        """--tick-now на час, который уже покрыт витриной 2025, не плодит строки."""
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 100)
            n0 = len(t.history('fire')[0])
            t.extend(101, {'fire': np.linspace(0.1, 0.9, 12, dtype=np.float32)}, st)
            self.assertEqual(len(t.history('fire')[0]), n0)

    def test_dump_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            tp = Path(d)
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 50, model_version='2026-09-25')
            t.dump()
            t2 = Thresholds(tp / 'history.parquet')
            self.assertTrue(t2.load(st, 50, model_version='2026-09-25'))
            self.assertEqual(t2.model_version, '2026-09-25')
            self.assertEqual(len(t2.history('fire')[0]), len(t.history('fire')[0]))
            self.assertAlmostEqual(t2.thresholds['fire'], t.thresholds['fire'], places=7)

    def test_load_rejects_other_model_version(self):
        """B5: история другой выгрузки не поднимается после рестарта — пересчитывается заново."""
        with tempfile.TemporaryDirectory() as d:
            tp = Path(d)
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 50, model_version='2026-09-25')
            t.dump()
            self.assertFalse(Thresholds(tp / 'history.parquet').load(st, 50, model_version='2026-09-26'))

    def test_estimate_uses_features_hour(self):
        """B7: оценка по часам витрины (T0), а не UNIX-часу — иначе окно не пересекает историю."""
        with tempfile.TemporaryDirectory() as d:
            st, t, scores = self._mk(d)
            t.bootstrap(scores, st, 100)
            h = 120
            when = ft.T0 + timedelta(hours=h)
            t.estimate(when, st)
            expect = quantile_of_window(scores['fire'][0], scores['fire'][1], h, st.share('fire'))
            self.assertAlmostEqual(t.thresholds['fire'], expect, places=7)

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