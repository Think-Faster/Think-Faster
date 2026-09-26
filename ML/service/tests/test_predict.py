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

    def test_seed_of_both_key_forms(self):
        from predict import seed_of
        self.assertEqual([seed_of('0'), seed_of('cat_s3'), seed_of('tcn_s12')], [0, 3, 12])

    def test_roll_scale_is_exact_net_and_wins(self):
        """Н20: сеть прогона вперёд — своя шкала roll/<прогон>_<тип>_val.npy раньше рабочего прогона ветки."""
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / 'roll').mkdir()
            np.save(work / 'roll' / 'all_s1_2024-01-01_equipment_val.npy', np.arange(3, dtype=float))
            (work / 'runs' / 'all_s1' / 'preds').mkdir(parents=True)
            np.save(work / 'runs' / 'all_s1' / 'preds' / 'tcn_s1_equipment_val.npy', np.arange(3, dtype=float))
            self.assertEqual(val_scale_file('all_s1_2024-01-01', 'tcn', 1, 'equipment', work),
                             work / 'roll' / 'all_s1_2024-01-01_equipment_val.npy')

    def test_bundle_scale_first(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'scales').mkdir()
            np.save(root / 'scales' / 'fire_cat_s2.npy', np.arange(3, dtype=float))
            self.assertEqual(val_scale_file('main_h24_s2', 'cat', 2, 'fire', None, root),
                             root / 'scales' / 'fire_cat_s2.npy')
            self.assertIsNone(val_scale_file('main_h24_s2', 'cat', 1, 'fire', None, root))

    def _version(self, work: Path, n: int, models: dict, blend: list, features=('tmp',)):
        d = work / f'export_equipment_v{n}'
        d.mkdir(parents=True)
        (d / 'manifest.json').write_text(json.dumps({
            'built': '2026-09-26', 'features': list(features), 'blend': blend, 'models': {'equipment': models},
            'version': {'type': 'equipment', 'number': n, 'name': f'в{n}', 'about': ''}}), encoding='utf-8')

    def test_version_overlay_and_weighted_blend(self):
        """M9a: версия заменяет только свой тип; смесь — среднее зёрен части, части со своими весами."""
        from predict import available_versions
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            main = {'fire': {'0': {'file': 'fire/cat_s0.cbm', 'family': 'cat', 'from_run': 'main_h24'}},
                    'equipment': {'0': {'file': 'equipment/tcn_s0.pt', 'family': 'tcn',
                                        'from_run': 'prod_s0_2026-07-01'}}}
            self._version(work, 1, {'cat_s0': {'file': 'equipment/cat_s0.cbm', 'family': 'cat', 'from_run': 'a'},
                                    'cat_s1': {'file': 'equipment/cat_s1.cbm', 'family': 'cat', 'from_run': 'a_s1'},
                                    'tcn_s0': {'file': 'equipment/tcn_s0.pt', 'family': 'tcn',
                                               'from_run': 'all_s0_2024-01-01'}},
                          [{'family': 'cat', 'weight': 0.75}, {'family': 'tcn', 'weight': 0.25}])
            p = self._predictor(work, work / 'export', main)
            self.assertEqual(available_versions(work / 'export')['equipment'][0]['number'], 1)
            p.use_version('equipment', 1)
            self.assertEqual(p.version, '2026-09-25+equipment_v1')
            self.assertEqual([e['seed'] for e in p.entries['equipment']], [0, 1, 0])
            self.assertEqual(p.entries['equipment'][0]['path'], work / 'export_equipment_v1' / 'equipment' / 'cat_s0.cbm')
            self.assertEqual(p.entries['fire'][0]['path'], work / 'export' / 'fire' / 'cat_s0.cbm')
            mix = p._blend('equipment', {'cat': [np.array([0.2]), np.array([0.4])], 'tcn': [np.array([1.0])]})
            self.assertAlmostEqual(float(mix[0]), 0.75 * 0.3 + 0.25 * 1.0, places=6)
            self.assertAlmostEqual(float(p._blend('fire', {'cat': [np.array([0.2]), np.array([0.6])]})[0]), 0.4, places=6)
            p.use_version('equipment', None)
            self.assertEqual(p.version, '2026-09-25')
            self.assertEqual(p.entries['equipment'][0]['from_run'], 'prod_s0_2026-07-01')
            with self.assertRaises(FileNotFoundError):
                p.use_version('equipment', 7)

    def test_version_on_other_features_refused(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            self._version(work, 2, {'tcn_s0': {'file': 'equipment/tcn_s0.pt', 'family': 'tcn', 'from_run': 'x'}},
                          [{'family': 'tcn', 'weight': 1.0}], features=('other',))
            p = self._predictor(work, work / 'export', {'equipment': {}})
            with self.assertRaises(ValueError):
                p.use_version('equipment', 2)

    def test_check_lists_missing_models_and_scales(self):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            p = self._predictor(work, work / 'export', {
                'fire': {'0': {'file': 'fire/cat_s0.cbm', 'family': 'cat', 'from_run': 'main_h24'}}})
            miss = p.check()
            self.assertEqual(len(miss), 2)
            (work / 'export' / 'fire').mkdir()
            (work / 'export' / 'fire' / 'cat_s0.cbm').write_bytes(b'')
            (work / 'runs' / 'main_h24' / 'preds').mkdir(parents=True)
            np.save(work / 'runs' / 'main_h24' / 'preds' / 'cat_fire_val.npy', np.arange(3, dtype=float))
            self.assertEqual(p.check(), [])

    def test_importance_from_reports_and_reasons_per_version(self):
        """Отчёт train.py хранит важность списком пар; основания — числа строки; версия — своя важность."""
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / 'runs' / 'main_h24').mkdir(parents=True)
            (work / 'runs' / 'main_h24' / 'report_cat.json').write_text(json.dumps(
                {'equipment': {'importance': {'cat': [['temp_1h', 0.6], ['smoke_24h', 0.4]]}}}), encoding='utf-8')
            main = {'equipment': {'0': {'file': 'equipment/cat_s0.cbm', 'family': 'cat', 'from_run': 'main_h24'}}}
            feats = ('smoke_24h', 'temp_1h')
            self._version(work, 1, {'cat_s0': {'file': 'equipment/cat_s0.cbm', 'family': 'cat', 'from_run': 'v'}},
                          [{'family': 'cat', 'weight': 1.0}], features=feats)
            (work / 'export_equipment_v1' / 'importance.json').write_text(
                json.dumps({'equipment': {'smoke_24h': 1.0}}), encoding='utf-8')
            p = self._predictor(work, work / 'export', main, features=feats)
            self.assertEqual(p.importance('equipment'), {'temp_1h': 0.6, 'smoke_24h': 0.4})
            frame = pl.DataFrame({'smoke_24h': [3.0, None], 'temp_1h': [21.5, 7.0]})
            self.assertEqual(p.reasons(frame, 0, 'equipment', k=2),
                             [{'feature': 'temp_1h', 'value': 21.5}, {'feature': 'smoke_24h', 'value': 3.0}])
            self.assertEqual(p.reasons(frame, 1, 'equipment', k=2)[1], {'feature': 'smoke_24h', 'value': None})
            p.use_version('equipment', 1)
            self.assertEqual(p.importance('equipment'), {'smoke_24h': 1.0})


if __name__ == '__main__':
    unittest.main()