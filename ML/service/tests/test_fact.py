"""M8: факт-канал — сверить фильтр Н7/Н8 и правило first с factalert.py."""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import fact
import storage


def mk_store(d: str):
    st = storage.HotStore(Path(d) / 'hot.duckdb')
    st.con.sql("""CREATE TABLE inc(object_id INT, type VARCHAR, t0 TIMESTAMP, t1 TIMESTAMP,
                                   noise VARCHAR, confirmed BOOLEAN)""")
    st.con.sql("CREATE TABLE guard(object_id INT, ts TIMESTAMP, armed BOOLEAN)")
    st.con.sql("CREATE OR REPLACE VIEW ev AS SELECT object_id, channel_id, ts, stype, state, num FROM ev_all")
    return st


class FactTest(unittest.TestCase):
    NOW = datetime(2026, 9, 23, 7, 0)

    def _fill(self, st, rows: list[tuple]):
        for o, tp, age_h, noise in rows:
            t0 = (self.NOW - timedelta(hours=age_h)).isoformat()
            st.con.execute(
                "INSERT INTO inc(object_id, type, t0, t1, noise, confirmed) VALUES (?,?,?,?,?,?)",
                (o, tp, t0, t0, noise, noise is None))

    def test_clean_in_last_hour_detected(self):
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            self._fill(st, [(1, 'fire', 0.5, None), (2, 'gas', 1.5, None), (3, 'fire', 10, None)])
            self.assertEqual(fact.detect(st, self.NOW, need_build=False),
                             {(1, 'fire')})

    def test_noise_filtered_for_fire(self):
        """Эпизод Н7/Н8 не даёт факта (правило clean/silent factalert)."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            self._fill(st, [(1, 'fire', 0.2, 'Н8')])
            self.assertEqual(fact.detect(st, self.NOW, need_build=False), set())

    def test_equipment_first_rule_without_filter(self):
        """Оборудование объявляет и шумное — правило first (фильтр вычёркивает 423 эпизода)."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            self._fill(st, [(5, 'equipment', 0.2, 'Н8')])
            self.assertEqual(fact.detect(st, self.NOW, need_build=False, noise=False),
                             {(5, 'equipment')})
            # для снятия отклонения (настоящие) шумное оборудование не идёт
            self.assertEqual(fact.detect(st, self.NOW, need_build=False, noise=True), set())

    def test_second_types_obey_filter_without_noise_param(self):
        """noise=False: у fire фильтр Н7/Н8 остаётся, оборудование — без."""
        with tempfile.TemporaryDirectory() as d:
            st = mk_store(d)
            self._fill(st, [(1, 'fire', 0.2, 'Н8'), (5, 'equipment', 0.2, 'Н8'),
                            (6, 'gas', 0.2, None)])
            self.assertEqual(fact.detect(st, self.NOW, need_build=False, noise=False),
                             {(5, 'equipment'), (6, 'gas')})

    def test_first_by_pass_types(self):
        self.assertEqual(fact.FIRST_BY_PASS, {'equipment'})


if __name__ == '__main__':
    unittest.main()