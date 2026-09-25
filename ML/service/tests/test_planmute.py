"""M7: молчание по графику плановых работ (INTEGRATION §1.6) на реальном works_2026.csv."""
import tempfile
import unittest
from datetime import datetime

import planmute
from planmute import mask, windows


def mk_obj3(d: str):
    import duckdb
    con = duckdb.connect(d)
    con.sql("""CREATE TABLE obj3(object_id BIGINT, collector_id BIGINT,
                                 object_level VARCHAR, name VARCHAR, address VARCHAR,
                                 location VARCHAR, housing VARCHAR, lon DOUBLE, lat DOUBLE)""")
    con.execute("INSERT INTO obj3 VALUES (4068,4068,'Коллектор','К','', '', '', 0,0)")
    con.execute("INSERT INTO obj3 VALUES (9,4068,'Объект','О','','','',0,0)")
    con.execute("INSERT INTO obj3 VALUES (7,4068,'Объект','О','','','',0,0)")
    con.execute("INSERT INTO obj3 VALUES (2,2,'Коллектор','К','','','',0,0)")
    return con


class PlanMuteTest(unittest.TestCase):
    def test_gas_in_window_work_id(self):
        """Объекты под коллектором 4068 в окне ППР (строка 1, 12–28.01) — MUTED с work_id."""
        with tempfile.TemporaryDirectory() as d:
            con = mk_obj3(d + '/t.db')
            self.assertTrue(windows(con, 2026))
            at = datetime(2026, 1, 20, 0, 0)
            m = mask(con, 'gas', at)
            self.assertEqual(m, {4068: '1', 7: '1', 9: '1'})   # коллектор и его объекты
            self.assertNotIn(2, m)

    def test_outside_window_no_mute(self):
        """После конца окна (29.01) глушения нет; до начала строки 1 тоже нет."""
        with tempfile.TemporaryDirectory() as d:
            con = mk_obj3(d + '/t.db')
            windows(con, 2026)
            self.assertEqual(mask(con, 'gas', datetime(2026, 1, 29, 0, 0)), {})
            self.assertEqual(mask(con, 'gas', datetime(2026, 1, 11, 23, 0)), {})

    def test_other_type_not_muted(self):
        """В окне ППР газа глушится только gas: строка 1 — incident_types gas."""
        with tempfile.TemporaryDirectory() as d:
            con = mk_obj3(d + '/t.db')
            windows(con, 2026)
            self.assertEqual(mask(con, 'fire', datetime(2026, 1, 20, 0, 0)), {})
            self.assertEqual(mask(con, 'sensor', datetime(2026, 1, 20, 0, 0)), {})

    def test_own_object_window(self):
        """Строка 3: объект 8, окно 29.01–13.02 — по самому объекту тоже глушится."""
        with tempfile.TemporaryDirectory() as d:
            con = mk_obj3(d + '/t.db')
            con.execute("INSERT INTO obj3 VALUES (8,8,'Коллектор','К','','','',0,0)")
            windows(con, 2026)
            self.assertEqual(mask(con, 'gas', datetime(2026, 2, 1, 0, 0))[8], '3')

    def test_reason_constant(self):
        self.assertEqual(planmute.REASON, 'плановые работы по графику')


if __name__ == '__main__':
    unittest.main()