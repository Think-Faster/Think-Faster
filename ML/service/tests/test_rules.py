import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import make_settings
from rules import RuleState
from settings import OperatingSettings


OBJECTS = np.array([10, 20, 30])
TYPES = ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')
THR = {t: 0.5 for t in TYPES}


def hist(seed: int = 0, n: int = 90 * 24) -> dict:
    """История парка: почти все оценки низкие, единицы высокие — thr_mute лежит высоко."""
    rng = np.random.default_rng(seed)
    h = np.arange(n)
    p = rng.random(n) * 0.1
    return {t: (h, p.astype(np.float32)) for t in TYPES}


class RulesTest(unittest.TestCase):
    def test_mute_suppresses(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            r.on_decision(20, 'fire', 'MUTE', 100, mute_hours=3)
            out = r.apply({'fire': np.array([0.9, 0.9, 0.9])}, OBJECTS, 100, THR)
            self.assertFalse(out['fire'][0][1])       # 20 в mute
            self.assertTrue(out['fire'][0][0])        # 10 не тронут

    def test_reject_log_only_for_non_rejectable(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            r.on_decision(10, 'fire', 'REJECT', 100)   # fire: reject_k None → только история
            out = r.apply({'fire': np.array([0.9, 0.0, 0.0])}, OBJECTS, 100, THR)
            self.assertTrue(out['fire'][0][0])         # тревога НЕ гасится
            self.assertEqual(len(r.log), 1)

    def test_reject_suppresses_gas_until_above_thr_mute(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            r.on_decision(10, 'gas', 'REJECT', 100)
            # thr_mute = quantile(истории, 1 - share*k) ≈ высокий ~0.998-квантиль ≈ 0.1
            low = r.apply({'gas': np.array([0.05, 0.0, 0.0])}, OBJECTS, 101, THR)
            self.assertFalse(low['gas'][0][0])          # ниже thr_mute — гасим
            # оценка вернулась выше thr_mute — снова горим (reject не вечен)
            high = r.apply({'gas': np.array([0.99, 0.0, 0.0])}, OBJECTS, 102, THR)
            self.assertTrue(high['gas'][0][0])

    def test_reject_expires_after_n_hours(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            r.on_decision(10, 'gas', 'REJECT', 100)
            import svc as config
            r.apply({'gas': np.array([0.05, 0.0, 0.0])}, OBJECTS, 100 + config.REJECT_N_HOURS + 1, THR)
            self.assertNotIn((10, 'gas'), r.rejections)

    def test_fact_clears_rejection_and_mute(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            r.on_decision(10, 'gas', 'REJECT', 100)
            r.on_decision(20, 'gas', 'MUTE', 100, mute_hours=12)
            r.on_fact({(10, 'gas'), (20, 'gas')}, 101)
            self.assertNotIn((10, 'gas'), r.rejections)
            self.assertNotIn((20, 'gas'), r.mutes)

    def test_chatter_merges_pause_leq_gap(self):
        """П6: разрыв ≤ 6 ч не начинает новый сигнал; since 0, пока сигнал жив."""
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            schedule = {10: 1.0, 11: 1.0, 12: 1.0, 13: 0.1, 14: 0.1, 15: 1.0, 16: 1.0}
            prev = None
            for hh in range(10, 17):
                val = schedule.get(hh, 0.1)
                al, since = r.apply({'fire': np.array([val, 0.0, 0.0])},
                                    OBJECTS, hh, THR)['fire']
                if val >= 0.5:
                    self.assertTrue(al[0], f'obj1 должен гореть в {hh}')
                    self.assertEqual(since[0], 0)       # живой сигнал: since 0
                else:
                    self.assertFalse(al[0], f'в {hh} тревога не горит')
                prev = hh
            # сигнал, начатый в 10, продолжен в 15 после паузы ≤ 6 → не новый (start=10)
            self.assertEqual(r.signal[(10, 'fire')]['start'], 10)
            # после закрытия эпизода since — часы с конца, не с последней тревоги
            al, since = r.apply({'fire': np.array([0.1, 0.0, 0.0])}, OBJECTS, 17, THR)['fire']
            self.assertFalse(al[0])
            self.assertEqual(since[0], 1)               # конец эпизода в 16 (последний час сигнала)
            # разрыв больше 6 часов → НОВЫЙ сигнал (start обновился)
            al, since = r.apply({'fire': np.array([1.0, 0.0, 0.0])}, OBJECTS, 24, THR)['fire']
            self.assertTrue(al[0])
            self.assertEqual(r.signal[(10, 'fire')]['start'], 24)

    def test_since_stays_zero_while_alarm_live(self):
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            for hh in (100, 101):
                al, since = r.apply({'fire': np.array([1.0, 0.0, 0.0])},
                                    OBJECTS, hh, THR)['fire']
                self.assertEqual(since[0], 0)

    def test_state_survives_restart_and_fact_reports_recurred(self):
        """13.3: решения людей на томе до подтверждения; факт снимает их и называет снятые пары."""
        with tempfile.TemporaryDirectory() as d:
            st = make_settings(Path(d))
            r = RuleState(st, history=hist())
            self.assertTrue(r.on_decision(10, 'gas', 'REJECT', 100, ref={'command_id': 'c1', 'by': 'u1'}))
            self.assertFalse(r.on_decision(10, 'fire', 'REJECT', 100))      # fire — только в историю
            r.on_decision(20, 'flood', 'MUTE', 100, until=110, ref={'command_id': 'c2'})
            r.apply({'fire': np.array([0.9, 0.0, 0.0])}, OBJECTS, 100, THR)
            r.dump(Path(d) / 'rules.json')
            again = RuleState(st, history=hist())
            self.assertTrue(again.load(Path(d) / 'rules.json'))
            self.assertEqual(again.rejections, {(10, 'gas'): 100})
            self.assertEqual(again.mutes, {(20, 'flood'): 110})
            self.assertEqual(again.signal[(10, 'fire')]['start'], 100)
            self.assertEqual(again.status(10, 'gas', 105), ('REJECTED', {'command_id': 'c1', 'by': 'u1', 'action': 'REJECT'}))
            self.assertEqual(again.status(20, 'flood', 109)[0], 'MUTED')
            gone = again.on_fact({(10, 'gas'), (30, 'fire')}, 106)
            self.assertEqual(gone, [((10, 'gas'), 'REJECTED', {'command_id': 'c1', 'by': 'u1', 'action': 'REJECT'})])
            self.assertIsNone(again.status(10, 'gas', 106))

    def test_mute_expires(self):
        with tempfile.TemporaryDirectory() as d:
            r = RuleState(make_settings(Path(d)), history=hist())
            r.on_decision(20, 'fire', 'MUTE', 100, until=102)
            out = r.apply({'fire': np.array([0.0, 0.9, 0.0])}, OBJECTS, 101, THR)['fire'][0]
            self.assertFalse(out[1])
            out = r.apply({'fire': np.array([0.0, 0.9, 0.0])}, OBJECTS, 102, THR)['fire'][0]
            self.assertTrue(out[1])
            self.assertEqual(r.mutes, {})


if __name__ == '__main__':
    unittest.main()