import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from predict import val_scale_file, load_manifest


def rank_on_scale(p: np.ndarray, base: np.ndarray) -> np.ndarray:
    """П1: место оценки на отсортированной шкале 2025 — как retro.load_mix_models."""
    base = np.sort(base)
    return np.searchsorted(base, p, side='right') / len(base)


class PredictTest(unittest.TestCase):
    def test_rank_maps_to_0_1(self):
        rng = np.random.default_rng(0)
        base = rng.random(2000)
        p = np.linspace(0, 1, 9)
        s = rank_on_scale(p, base)
        self.assertTrue((s >= 0).all() and (s <= 1).all())
        self.assertAlmostEqual(float(s[-1]), 1.0, places=4)

    def test_rank_is_monotone_on_scale(self):
        """Место на фиксированной шкале 2025 монотонно: большая оценка — больше ранг (П1)."""
        base = np.linspace(0.001, 0.999, 1000)
        p = np.linspace(0, 1, 200)
        s = rank_on_scale(p, base)
        self.assertTrue((np.diff(s) >= 0).all())

    def test_mixture_is_mean_of_seed_ranks(self):
        rng = np.random.default_rng(1)
        bases = [rng.random(2000) for _ in range(2)]
        p = rng.random(40)
        mix = np.mean([rank_on_scale(p, b) for b in bases], axis=0)
        self.assertTrue((mix >= 0).all() and (mix <= 1).all())

    def test_val_scale_file_search_order(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            preds = work / 'runs' / 'main_h24' / 'preds'
            preds.mkdir(parents=True)
            np.save(preds / 'tcn_s1_fire_val.npy', np.arange(3, dtype=float))
            self.assertEqual(val_scale_file('main_h24', 'tcn', 1, 'fire', work),
                             preds / 'tcn_s1_fire_val.npy')
            self.assertIsNone(val_scale_file('main_h24', 'tcn', 2, 'fire', work))

    def test_manifest_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                load_manifest(Path(d))

    def test_manifest_parses_main_format(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            (exp / 'models').mkdir(parents=True)
            manifest = {
                'exported': '2026-09-24', 'features': ['smoke_24h', 'temp_1h'],
                'types': ['fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion'],
                'models': {
                    'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb',
                                   'from_run': 'main_h24', 'params': {}, 'trees': 500}},
                    'equipment': {'1': {'file': 'equipment/tcn_s1.pt', 'family': 'tcn',
                                        'from_run': 'prod_s1_2026-07-01', 'params': {}, 'trees': 0}}}}
            (exp / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            m = load_manifest(exp)
            self.assertEqual(m['models']['fire']['0']['family'], 'xgb')
            self.assertEqual(m['models']['equipment']['1']['from_run'], 'prod_s1_2026-07-01')


if __name__ == '__main__':
    unittest.main()