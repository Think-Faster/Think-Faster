import tempfile
import unittest
from pathlib import Path

from helpers import make_settings, write_operating
from settings import OperatingSettings, SettingError, estimate

import numpy as np


class SettingsTest(unittest.TestCase):
    def test_load_and_methods(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            self.assertEqual(st.version, 1)
            self.assertEqual(st.share('fire'), 0.030)
            self.assertIsNone(st.reject_k('fire'))
            self.assertEqual(st.reject_k('gas'), 0.2)
            self.assertTrue(st.is_rejectable('gas'))
            self.assertFalse(st.is_rejectable('fire'))

    def test_bounds_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_operating(Path(d))
            # share вне схемной границы (fire: 0.015..0.045)
            raw = __import__('json').loads(p.read_text(encoding='utf-8'))
            raw['types']['fire']['share'] = 2.0
            p.write_text(__import__('json').dumps(raw), encoding='utf-8')
            with self.assertRaises(Exception):
                OperatingSettings.load(p)

    def test_rebase_creates_new_version(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            new = st.rebase({'fire': {'share': 0.035}}, by='disp-01',
                            reason='ползунок', now='2026-09-24T00:00:00+03:00')
            self.assertEqual(new.version, 2)
            self.assertEqual(new.changed_by, 'disp-01')
            self.assertEqual(new.share('fire'), 0.035)
            self.assertEqual(st.share('fire'), 0.030)   # оригинал не тронут

    def test_estimate_sanity(self):
        rng = np.random.default_rng(7)
        hist = rng.random(200 * 24)
        res = estimate(hist, 0.1, 90)
        self.assertTrue(0 < res['alarms_per_day'] < 200)
        self.assertEqual(res['threshold'], np.quantile(hist, 0.9))

    def test_check_rejects_bad_share(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            with self.assertRaises(SettingError):
                st.check(0.0)


if __name__ == '__main__':
    unittest.main()