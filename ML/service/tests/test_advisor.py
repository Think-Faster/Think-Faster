import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import recommend
from advisor import build, load
import outbox
import storage


def mk_store(d: str):
    st = storage.HotStore(Path(d) / 'hot.duckdb')
    st.con.sql('CREATE TABLE inc(object_id INT, type VARCHAR, t0 TIMESTAMP, confirmed BOOLEAN, noise VARCHAR)')
    st.con.sql('CREATE TABLE guard(object_id INT, ts TIMESTAMP, armed BOOLEAN)')
    st.con.sql('CREATE OR REPLACE VIEW ev AS SELECT object_id, channel_id, ts, stype, state, num FROM ev_all')
    return st


def fact_reco(con, obj, tp, at, *, stype, what, channel, k30=0):
    """Режим «факт»: рекомендация по эпизоду (своя ветка recommend(), как у main)."""
    rules, recur, ver = load()
    r = {'object_id': obj, 'type': tp, 't0': at, 'stype': stype, 'what': what,
         'channel_id': channel, 'k30': k30, 'prev_t0': None, 'prev_confirmed': False,
         'prev_channel': None, 'line_n': 0, 'armed': True, 'restart': False,
         'co_types': [], 'reason_triggers': [], 'since_hours': 0}
    r.update(recommend.ambient(con, obj, at))
    return recommend.recommend(r, rules, ver, recur)


class AdvisorTest(unittest.TestCase):
    AT = datetime(2026, 9, 23, 7, 0)

    def test_forecast_smoke_r0_standing(self):
        """Прогноз пожара: эпизодов нет — R0, основание модели → свой признак, меры с составом."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = build(st.con, 1, 'fire', self.AT, reasons=['smoke_24h'], since_h=200)
            self.assertEqual(rec['mode'], 'прогноз')
            self.assertEqual(rec['stage'], 'R0')
            self.assertIn('smoke', rec['triggers'])
            self.assertEqual(rec['pattern'], 'standing')
            self.assertEqual(rec['produced_by'], 'rules')
            self.assertEqual(len(rec['version']), 10)
            for m in rec['now'] + rec['maintenance']:
                self.assertIn('людей', m)
            self.assertIsInstance(rec['repeat_7d'], float)

    def test_stage_after_visit_r2(self):
        """Повтор после выезда бригады — R2 (правило меняет меру: искать причину через повтор)."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            for i in range(2):
                st.con.execute(
                    "INSERT INTO inc VALUES (?,?,?,?,?)",
                    (1, 'fire', self.AT - __import__('datetime').timedelta(days=3 - i, hours=12), True, None))
            rec = build(st.con, 1, 'fire', self.AT, reasons=['smoke_24h'], since_h=2)
            self.assertEqual(rec['stage'], 'R2')
            self.assertIn('after_visit', rec['context'])

    def test_gas_forecast_visit_brigada(self):
        """M13а: прогноз газа по некартируемому основанию — FC-02, бригада 4 на 24 ч (газоопасные)."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = build(st.con, 1, 'gas', self.AT, reasons=['gas_fault_168h'], since_h=0)
            self.assertEqual(rec['triggers'], ['forecast'])
            self.assertEqual(rec['now'][0]['rule_id'], 'FC-02')
            self.assertEqual(rec['visit']['состав'], 'бригада')
            self.assertEqual(rec['visit']['людей'], 4)
            self.assertEqual(rec['visit']['срок_ч'], 24)

    def test_gas_forecast_with_episode_feature_only_maintenance(self):
        """Прогноз газа по «Обнаружен газ»: выезд по факту — GA-01 в ТО, оперативно пусто (как в main)."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = build(st.con, 1, 'gas', self.AT, reasons=['gas'], since_h=0)
            self.assertEqual(rec['now'], [])
            self.assertIsNone(rec['visit'])
            self.assertEqual([m['rule_id'] for m in rec['maintenance']], ['GA-01'])

    def test_gas_fact_visit_brigada_srok0(self):
        """M13а: газ по факту — первая мера бригада из 4 со сроком 0, выезд тот же, запаски Б без выезда."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = fact_reco(st.con, 1, 'gas', self.AT, stype='Газовый датчик', what='Обнаружен газ', channel=7)
            self.assertEqual(rec['mode'], 'факт')
            self.assertEqual(rec['triggers'], ['gas'])
            self.assertEqual(rec['now'][0]['состав'], 'бригада')
            self.assertEqual(rec['now'][0]['срок_ч'], 0)
            self.assertEqual(rec['now'][0]['людей'], 4)
            self.assertIn('GA-01', {m['rule_id'] for m in rec['now']})
            self.assertEqual(rec['visit']['состав'], 'бригада')
            self.assertEqual(rec['visit']['людей'], 4)
            self.assertEqual(rec['visit']['срок_ч'], 0)
            self.assertFalse(rec['visit']['после_проверки'])

    def test_equipment_phase_energyk(self):
        """M13а: отказ оборудования по фазе — специалист-энергетик (EQ-07 A), без людей в выезд."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = build(st.con, 2, 'equipment', self.AT, reasons=['phase_off'], since_h=0)
            self.assertEqual(rec['visit']['состав'], 'специалист')
            self.assertEqual(rec['visit']['людей'], 1)
            self.assertEqual(rec['visit']['исполнители'], ['энергетик'])

    def test_compose_several_types(self):
        """Несколько типов на объекте — object_recommendation: меры по сроку, ТО без повторов."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            fire = build(st.con, 3, 'fire', self.AT, reasons=['smoke_24h'], since_h=2, co_types=['sensor'])
            sens = build(st.con, 3, 'sensor', self.AT, reasons=['smoke_fault_168h'], since_h=2, co_types=['fire'])
            comp = recommend.compose([fire, sens])
            self.assertEqual(comp['types'], ['fire', 'sensor'])
            self.assertIn('visit', comp)
            now = comp['now']
            self.assertEqual(now, sorted(now, key=lambda m: m['срок_ч']))
            pairs = [(m['rule_id'], m['мера']) for m in comp['maintenance']]
            self.assertEqual(len(pairs), len(set(pairs)))   # ТО без повторов одного правила
            msg = outbox.build_message(3, '2026-09-23T08:00', 'v', ['fire', 'sensor'])
            outbox.object_recommendation(msg, comp)
            self.assertEqual(msg['object_recommendation']['types'], ['fire', 'sensor'])

    def test_message_carries_recommendation(self):
        """§2.3: recommendation лежит только в блоке тревоги."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            rec = build(st.con, 1, 'fire', self.AT, reasons=['smoke_24h'], since_h=2)
            msg = outbox.build_message(1, '2026-09-23T08:00', 'v', ['fire', 'gas'])
            outbox.fill_type(msg, 'fire', score=0.9, threshold=0.5, alarm=True,
                             recommendation=rec)
            outbox.fill_type(msg, 'gas', score=0.1, threshold=0.5, alarm=False)
            self.assertEqual(msg['types']['fire']['recommendation']['trigger'], 'smoke')
            self.assertNotIn('recommendation', msg['types']['gas'])


if __name__ == '__main__':
    unittest.main()