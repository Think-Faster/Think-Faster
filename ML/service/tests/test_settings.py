import json
import tempfile
import unittest
from pathlib import Path

from settings import OperatingSettings, SettingError, estimate

import numpy as np


def make_settings(tmp: Path) -> OperatingSettings:
    raw = {
        'version': 3, 'changed': '2026-09-23T00:00:00+03:00', 'changed_by': 'ml',
        'reason': 'тест', 'horizon_hours': 24,
        'shares': {'fire': 0.025, 'gas': 0.026, 'flood': 0.018, 'equipment': 0.059,
                   'sensor': 0.009, 'intrusion': 0.005},
        'reject_k': 0.2, 'reject_types': ['gas', 'flood'], 'max_share': 0.1,
        'threshold_window_days': 90, 'chatter_gap_hours': 6, 'mute_max_hours': 72}
    p = tmp / 'operating.json'
    p.write_text(json.dumps(raw), encoding='utf-8')
    return OperatingSettings.load(p)


class SettingsTest(unittest.TestCase):
    def test_load_and_shares(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            self.assertEqual(st.version, 3)
            self.assertEqual(st.share('fire'), 0.025)
            self.assertTrue(st.is_rejectable('gas'))
            self.assertFalse(st.is_rejectable('fire'))

    def test_bounds_reject(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            with self.assertRaises(SettingError):
                st.check(0.3)
            with self.assertRaises(SettingError):
                OperatingSettings._from(dict(st.as_dict(), shares=dict(st.shares, fire=0.5)), Path(d))

    def test_rebase_creates_new_version(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            new = st.rebase({'shares': dict(st.shares, fire=0.03)}, by='disp-01',
                            reason='ползунок', now='2026-09-24T00:00:00+03:00')
            self.assertEqual(new.version, 4)
            self.assertEqual(new.changed_by, 'disp-01')
            self.assertEqual(new.share('fire'), 0.03)
            self.assertEqual(st.share('fire'), 0.025)   # оригинал не тронут

    def test_estimate_sanity(self):
        rng = np.random.default_rng(7)
        hist = rng.random(200 * 24)
        res = estimate(hist, 0.1)
        self.assertTrue(0 < res['alarms_per_day'] < 200)
        self.assertEqual(res['threshold'], np.quantile(hist, 0.9))


if __name__ == '__main__':
    unittest.main()