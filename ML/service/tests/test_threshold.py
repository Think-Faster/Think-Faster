import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from settings import OperatingSettings
from threshold import ScoreHistory


def make_settings(tmp: Path) -> OperatingSettings:
    raw = {'version': 1, 'changed': '2026-09-23T00:00:00+03:00', 'changed_by': 'ml', 'reason': 't',
           'horizon_hours': 24,
           'shares': {t: 0.1 for t in ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')},
           'reject_k': 0.2, 'reject_types': ['gas', 'flood'], 'max_share': 0.1,
           'threshold_window_days': 90, 'chatter_gap_hours': 6, 'mute_max_hours': 72}
    p = tmp / 'operating.json'
    p.write_text(json.dumps(raw), encoding='utf-8')
    return OperatingSettings.load(p)


class ThresholdTest(unittest.TestCase):
    def test_window_and_quantile(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            hist = ScoreHistory(st, path=Path(d) / 'history.parquet')
            rng = np.random.default_rng(1)
            for h in range(24):
                hist.update(h, {t: rng.random(3) for t in st.shares})
            t = hist.threshold('fire')
            self.assertAlmostEqual(t, float(np.quantile(hist.window('fire'), 0.9)), places=6)

    def test_persist_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'history.parquet'
            st = make_settings(Path(d))
            a = ScoreHistory(st, path=p)
            a.update(1, {t: np.array([0.1, 0.2]) for t in st.shares})
            a.save()
            b = ScoreHistory(st, path=p)
            self.assertEqual(b.window('fire').size, 2)
            self.assertTrue(np.allclose(b.window('fire'), [0.1, 0.2]))


if __name__ == '__main__':
    unittest.main()