import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from storage import HotStore, GUARD_DATE


def store_with_ch(tmp: Path) -> HotStore:
    st = HotStore(tmp / 'hot.duckdb')
    st.con.execute("INSERT INTO ch (channel_id, system, stype, tag, name, object_id) VALUES "
                   "(1, 'Пожарная сигнализация', 'Датчик дыма', '', 'Дым ПК1', 5122),"
                   "(2, 'Охрана', 'Состояние охраны', '', 'Охрана', 5123),"
                   "(3, 'Виртуальный', 'Газовый датчик', '', 'Газ', 5124)")
    return st


class StorageTest(unittest.TestCase):
    def test_clean_event_unknown_channel(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            self.assertIsNone(st.clean_event(999, datetime(2026, 1, 1), 'Обнаружен дым'))

    def test_clean_event_guard_date_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            self.assertIsNone(st.clean_event(2, datetime(2026, 1, 1), '01.01.1970 00:00:00'))
            self.assertIsNone(st.clean_event(2, datetime(2026, 1, 1), '##.##.####'))
            self.assertIsNone(st.clean_event(2, datetime(2026, 1, 1), '14.08.2025 12:00:00'))

    def test_clean_event_num_state_split(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            r = st.clean_event(3, datetime(2026, 1, 1), '0.12')
            self.assertEqual(r['num'], 0.12)
            self.assertIsNone(r['state'])
            r = st.clean_event(1, datetime(2026, 1, 1), 'Обнаружен дым')
            self.assertIsNone(r['num'])
            self.assertEqual(r['state'], 'Обнаружен дым')

    def test_append_dedup_and_retention(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            ev1 = st.clean_event(1, datetime(2026, 6, 1, 0, 0), 'Обнаружен дым')
            ev2 = st.clean_event(1, datetime(2026, 6, 1, 0, 0), 'Обнаружен дым')  # дубль Н6
            st.append([ev1, ev2])
            self.assertEqual(st.con.sql('SELECT count(*) FROM readings').fetchone()[0], 1)
            st.con.execute("INSERT INTO readings VALUES (5122, 1, TIMESTAMP '2026-01-01 00:00:00', "
                           "'Датчик дыма', 'Обнаружен дым', NULL, 'Обнаружен дым')")
            swept = st.retention_sweep(datetime(2026, 6, 2))
            self.assertEqual(swept, 1)

    def test_guard_date_pattern(self):
        self.assertTrue(GUARD_DATE.match('01.01.1970'))
        self.assertFalse(GUARD_DATE.match('Обнаружен дым'))


if __name__ == '__main__':
    unittest.main()