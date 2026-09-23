import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rules import RuleState
from settings import OperatingSettings


def make_settings(tmp: Path) -> OperatingSettings:
    raw = {
        'version': 1, 'changed': '2026-09-23T00:00:00+03:00', 'changed_by': 'ml', 'reason': 't',
        'horizon_hours': 24,
        'shares': {t: 0.02 for t in ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')},
        'reject_k': 0.2, 'reject_types': ['gas', 'flood'], 'max_share': 0.1,
        'threshold_window_days': 90, 'chatter_gap_hours': 6, 'mute_max_hours': 72}
    p = tmp / 'operating.json'
    p.write_text(json.dumps(raw), encoding='utf-8')
    return OperatingSettings.load(p)


OBJECTS = np.array([10, 20, 30])
THR = {t: 0.5 for t in ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')}


class RulesTest(unittest.TestCase):
    def test_mute_suppresses(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st)
            r.on_decision(20, 'fire', 'MUTE', 100, mute_hours=3)
            scores = {'fire': np.array([0.9, 0.9, 0.9])}
            out = r.apply(scores, OBJECTS, 100, THR)
            self.assertFalse(out['fire'][0][1])       # 20 в mute
            self.assertTrue(out['fire'][0][0])        # 10 не тронут

    def test_reject_weights_only_rejectable(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st)
            r.on_decision(10, 'gas', 'REJECT', 100)
            r.on_decision(10, 'fire', 'REJECT', 100)   # fire вне reject_types → просто история
            scores = {'gas': np.array([0.9, 0.0, 0.0]),
                      'fire': np.array([0.9, 0.0, 0.0])}
            # порог 0.5, вес 0.8: 0.9*0.8=0.72 ≥ 0.5 — газовая тревога не гасится, weight вступил
            out = r.apply(scores, OBJECTS, 100, THR)
            self.assertTrue(out['gas'][0][0])
            # газ с весом: убедимся, что вес реально применился через порог 0.8
            self.assertFalse(r.apply({'gas': np.array([0.6, 0.0, 0.0])}, OBJECTS, 100, THR)['gas'][0][0])
            self.assertTrue(out['fire'][0][0])        # fire: без веса 0.9 ≥ 0.5

    def test_fact_clears_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st)
            r.on_decision(10, 'gas', 'REJECT', 100)
            r.on_fact({(10, 'gas')}, 101)
            scores = {'gas': np.array([0.9, 0.0, 0.0])}
            out = r.apply(scores, OBJECTS, 101, THR)
            self.assertTrue(out['gas'][0][0])         # эпизод пришёл — вес больше не давит

    def test_since_hours_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st)
            scores = {'fire': np.array([0.9, 0.0, 0.0])}
            r.apply(scores, OBJECTS, 100, THR)
            scores = {'fire': np.array([0.9, 0.0, 0.0])}
            out = r.apply(scores, OBJECTS, 101, THR)
            self.assertEqual(out['fire'][1][0], 1)   # второй час той же серии


if __name__ == '__main__':
    unittest.main()