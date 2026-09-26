import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

import bundle
from predict import Predictor

FEATURES = ['a', 'b']


def _work(root: Path, n: int = 60) -> Path:
    """Рабочая папка: витрина 2025 на n строк и описание признаков."""
    rng = np.random.default_rng(3)
    feat = root / 'work' / 'features'
    feat.mkdir(parents=True)
    (feat / 'meta.json').write_text(json.dumps({'features': FEATURES}), encoding='utf-8')
    pl.DataFrame({'object_id': np.repeat([1, 2, 3], n // 3), 'h': np.tile(np.arange(n // 3), 3),
                  'a': (a := rng.random(n)), 'b': rng.random(n),
                  # метка калибровки: эпизод пожара в ближайшие 24 ч чаще там, где больше признак a
                  'next_fire': np.where(rng.random(n) < a, 5.0, 500.0)}).write_parquet(feat / '2025.parquet')
    return root / 'work'


def _export(root: Path, models: dict) -> Path:
    exp = root / 'export'
    exp.mkdir(parents=True)
    (exp / 'manifest.json').write_text(json.dumps({'built': '2026-09-26', 'features': FEATURES, 'models': models}),
                                       encoding='utf-8')
    return exp


class BundleTest(unittest.TestCase):
    def test_boost_scale_is_export_model_on_work_rows(self):
        """Шкала бустинга — оценки самой модели выгрузки на всех строках витрины 2025, в их порядке."""
        with tempfile.TemporaryDirectory() as d:
            work = _work(Path(d), 61 * 3)
            exp = _export(Path(d), {'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb', 'from_run': 'r'}}})
            p = Predictor(exp, work)
            e = p.entries['fire'][0]
            p._boosters[e['path']] = lambda X: X[:, 0] * 2 + X[:, 1]
            bundle.own_scales(p, work, Path(d) / 'out', [('fire', e)], chunk=7)
            df = pl.read_parquet(work / 'features' / '2025.parquet')
            np.testing.assert_allclose(np.load(e['scale']), (df['a'] * 2 + df['b']).to_numpy(), rtol=1e-6)
            self.assertEqual(e['scale'], Path(d) / 'out' / 'fire_xgb_s0.npy')
            self.assertIn('work/features/2025.parquet', e['origin'])

    def test_net_scale_only_for_same_weights(self):
        """Шкала сети из roll берётся, только если веса roll/<прогон>.pt те же, что у сети выгрузки;
        при отказе шкалы бустингов не считаются."""
        import torch
        with tempfile.TemporaryDirectory() as d:
            work = _work(Path(d))
            exp = _export(Path(d), {
                'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb', 'from_run': 'r'}},
                'equipment': {'0': {'file': 'equipment/tcn_s0.pt', 'family': 'tcn', 'from_run': 'prod_s0_2026-07-01'}}})
            (exp / 'equipment').mkdir()
            (work / 'roll').mkdir()
            torch.save({'w': torch.tensor([1.0, 2.0])}, exp / 'equipment' / 'tcn_s0.pt')
            torch.save({'w': torch.tensor([1.0, 3.0])}, work / 'roll' / 'prod_s0_2026-07-01.pt')
            np.save(work / 'roll' / 'prod_s0_2026-07-01_equipment_val.npy', np.zeros(60, np.float32))
            p = Predictor(exp, work)
            net = p.entries['equipment'][0]
            p._nets[net['path']] = object()                  # сеть не нужна: locate не считает её оценки

            def never(X):
                raise AssertionError('шкалы бустингов посчитаны при отказе сети')
            p._boosters[p.entries['fire'][0]['path']] = never
            miss = bundle.locate(p, work, Path(d) / 'out')
            self.assertEqual(len(miss), 1)
            self.assertIn('не та сеть', miss[0])
            torch.save({'w': torch.tensor([1.0, 2.0])}, work / 'roll' / 'prod_s0_2026-07-01.pt')
            p._boosters[p.entries['fire'][0]['path']] = lambda X: X[:, 0]
            self.assertEqual(bundle.locate(p, work, Path(d) / 'out'), [])
            self.assertEqual(net['scale'], work / 'roll' / 'prod_s0_2026-07-01_equipment_val.npy')
            self.assertEqual(net['origin'], 'work/roll/prod_s0_2026-07-01_equipment_val.npy')

    def test_scale_rows_must_match_work(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            work = _work(Path(d))
            exp = _export(Path(d), {'equipment': {'0': {'file': 'equipment/tcn_s0.pt', 'family': 'tcn',
                                                        'from_run': 'x_2026-07-01'}}})
            (exp / 'equipment').mkdir()
            (work / 'roll').mkdir()
            for f in (exp / 'equipment' / 'tcn_s0.pt', work / 'roll' / 'x_2026-07-01.pt'):
                torch.save({'w': torch.tensor([1.0])}, f)
            np.save(work / 'roll' / 'x_2026-07-01_equipment_val.npy', np.zeros(59, np.float32))
            p = Predictor(exp, work)
            p._nets[p.entries['equipment'][0]['path']] = object()
            miss = bundle.locate(p, work, Path(d) / 'out')
            self.assertEqual(len(miss), 1)
            self.assertIn('59 строк', miss[0])

    def test_build_scores_like_export_and_records_origin(self):
        """Пакет считает то же, что выгрузка со шкалами locate; важность — из модели выгрузки."""
        import xgboost as xgb
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work = _work(root)
            exp = _export(root, {'fire': {'0': {'file': 'fire/xgb_s0.json', 'family': 'xgb', 'from_run': 'r'}}})
            rng = np.random.default_rng(0)
            X = rng.random((300, 2)).astype(np.float32)
            b = xgb.train({'objective': 'binary:logistic', 'max_depth': 2},
                          xgb.DMatrix(X, label=(X[:, 0] > 0.7)), 5)
            (exp / 'fire').mkdir()
            b.save_model(str(exp / 'fire' / 'xgb_s0.json'))
            res = bundle.build(root / 'bundle', work=work, export=exp, settings=root / 'нет')
            self.assertEqual(res['inputs'], 'work')
            man = json.loads((root / 'bundle' / 'manifest.json').read_text(encoding='utf-8'))
            self.assertIn('work/features/2025.parquet', man['bundle']['scales']['fire/0'])
            imp = json.loads((root / 'bundle' / 'importance.json').read_text(encoding='utf-8'))['fire']
            self.assertEqual(next(iter(imp)), 'a')
            frame = pl.DataFrame({'a': [0.1, 0.75, 0.9], 'b': [0.5, 0.5, 0.5]})
            pa = Predictor(root / 'bundle').predict(frame)['fire']
            q = Predictor(exp, work)
            self.assertEqual(bundle.locate(q, work, root / 'tmp'), [])
            np.testing.assert_allclose(pa, q.predict(frame)['fire'])
            self.assertTrue((np.diff(pa) >= 0).all())
            cal = json.loads((root / 'bundle' / 'calibration.json').read_text(encoding='utf-8'))['fire']
            self.assertEqual((cal['rows'], len(cal['x'])), (60, len(cal['y'])))
            conf = Predictor(root / 'bundle').confidence('fire', pa)
            self.assertTrue(((conf >= 0) & (conf <= 1)).all() and (np.diff(conf) >= 0).all())
            np.testing.assert_allclose(conf, q.confidence('fire', pa))   # пакет и рабочая папка — одна кривая


if __name__ == '__main__':
    unittest.main()
