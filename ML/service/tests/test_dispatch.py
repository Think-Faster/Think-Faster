import unittest

import numpy as np
import polars as pl

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))

from dispatch import card_metrics, split_series  # noqa: E402


class DispatchTest(unittest.TestCase):
    def test_series_split_and_metrics(self):
        a = pl.DataFrame({
            'object_id': [1, 1, 1, 1, 2],
            'h': [0, 1, 3, 4, 0],
            'type': ['fire'] * 5,
            'alarm': [True] * 5})
        s = split_series(a)
        # пары (1,fire): часы 0,1 — серия 1; 3,4 — серия 2; (2,fire): час 0 — серия 1 длины 1
        by = s.group_by(['object_id', 'type', 'series']).len().sort('len')
        self.assertEqual(by['len'].to_list(), [1, 2, 2])
        m = card_metrics(a)
        self.assertEqual(m['open_max'], 2)          # в час 0 открыто (1,fire) и (2,fire)
        self.assertGreaterEqual(m['overlap_share'], 0.0)


if __name__ == '__main__':
    unittest.main()