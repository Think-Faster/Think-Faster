import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from predict import Predictor, val_scale_file, load_manifest

TYPES = ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')


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

    def test_val_scale_file_falls_back_from_roll_to_working_run(self):
        """B3: у сети from_run — roll-прогон без 2025-шкалы; путь рабочего прогона ветки."""
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            preds1 = work / 'runs' / 'prod_s1' / 'preds'
            preds1.mkdir(parents=True)
            np.save(preds1 / 'tcn_s1_equipment_val.npy', np.arange(3, dtype=float))
            self.assertEqual(val_scale_file('prod_s1_2026-07-01', 'tcn', 1, 'equipment', work),
                             preds1 / 'tcn_s1_equipment_val.npy')
            preds0 = work / 'runs' / 'prod' / 'preds'
            preds0.mkdir(parents=True)
            np.save(preds0 / 'tcn_equipment_val.npy', np.arange(4, dtype=float))
            self.assertEqual(val_scale_file('prod_s0_2026-07-01', 'tcn', 0, 'equipment', work),
                             preds0 / 'tcn_equipment_val.npy')

    def test_manifest_missing_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                load_manifest(Path(d))

    def test_manifest_parses_main_format(self):
        with tempfile.TemporaryDirectory() as d:
            exp = Path(d)
            (exp / 'models').mkdir(parents=True)
            manifest = {
                'built': '2026-09-24', 'features': ['smoke_24h', 'temp_1h'],
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

    def _predictor(self, work: Path, exp: Path, models: dict, features=('tmp',)) -> Predictor:
        feat = work / 'features'
        feat.mkdir(parents=True, exist_ok=True)
        (feat / 'meta.json').write_text(json.dumps({'features': list(features)}), encoding='utf-8')
        exp.mkdir(parents=True)
        manifest = {'built': '2026-09-25', 'features': list(features),
                    'types': list(TYPES), 'models': models}
        (exp / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return Predictor(exp, work)

    def test_version_is_manifest_built(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._predictor(Path(d), Path(d) / 'export', {})
            self.assertEqual(p.version, '2026-09-25')

    def test_bootstrap_history_from_saved_preds(self):
        """B2: история из preds-файлов зёрен (ранг на собственной шкале), без пересчёта сетей."""
        import polars as pl
        from predict import Predictor
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            rng = np.random.default_rng(7)
            h = np.tile(np.arange(24), 3)
            feat = work / 'features'
            feat.mkdir(parents=True)
            (feat / 'meta.json').write_text(json.dumps({'features': ['tmp']}), encoding='utf-8')
            pl.DataFrame({'object_id': np.repeat([1, 2, 3], 24), 'h': h,
                          'tmp': rng.random(72)}).write_parquet(feat / '2025.parquet')
            models = {'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb',
                                     'from_run': 'main_h24', 'params': {}, 'trees': 10}},
                      'equipment': {'1': {'file': 'equipment/tcn_s1.pt', 'family': 'tcn',
                                          'from_run': 'prod_s1_2026-07-01', 'params': {}, 'trees': 0}}}
            p = self._predictor(work, work / 'export', models)
            p_fire = np.linspace(0.001, 0.999, 72, dtype=np.float32)
            (work / 'runs' / 'main_h24' / 'preds').mkdir(parents=True)
            np.save(work / 'runs' / 'main_h24' / 'preds' / 'xgb_fire_val.npy', p_fire)
            p_eq = np.linspace(0.999, 0.001, 72, dtype=np.float32)
            (work / 'runs' / 'prod_s1' / 'preds').mkdir(parents=True)
            np.save(work / 'runs' / 'prod_s1' / 'preds' / 'tcn_s1_equipment_val.npy', p_eq)
            hist = p.bootstrap_history(2025)
            self.assertIn('fire', hist)
            self.assertIn('equipment', hist)   # tcn: не KeyError и не требует seqdata (B2)
            hh, pp = hist['fire']
            order = np.argsort(h, kind='stable')   # тот же порядок, что в bootstrap_history
            expect = np.searchsorted(np.sort(p_fire), p_fire, side='right') / len(p_fire)
            np.testing.assert_allclose(pp, expect[order], rtol=1e-6)
            np.testing.assert_array_equal(hh, h[order])

    def test_bootstrap_history_raises_without_scale_files(self):
        import polars as pl
        from predict import Predictor
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            h = np.arange(24)
            feat = work / 'features'
            feat.mkdir(parents=True)
            (feat / 'meta.json').write_text(json.dumps({'features': ['tmp']}), encoding='utf-8')
            pl.DataFrame({'object_id': np.ones(24, int), 'h': h,
                          'tmp': np.linspace(0, 1, 24)}).write_parquet(feat / '2025.parquet')
            models = {'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb',
                                     'from_run': 'main_h24', 'params': {}, 'trees': 10}}}
            p = self._predictor(work, work / 'export', models)
            with self.assertRaises(FileNotFoundError):
                p.bootstrap_history(2025)


if __name__ == '__main__':
    unittest.main()