import csv
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


def write_journal(path: Path, rows: list[list]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ид_события', 'ид_канала_данных', 'дата', 'время', 'тревожное', 'значение_датчика'])
        for r in rows:
            w.writerow(r)


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

    def test_clean_event_guard_date_kept_for_non_guard(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            r = st.clean_event(1, datetime(2026, 1, 1), '01.01.1970')   # не охрана — это значение
            self.assertIsNotNone(r)
            self.assertEqual(r['state'], '01.01.1970')

    def test_clean_event_num_state_split(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            r = st.clean_event(3, datetime(2026, 1, 1), '0.12')
            self.assertEqual(r['num'], 0.12)
            self.assertIsNone(r['state'])
            r = st.clean_event(1, datetime(2026, 1, 1), 'Обнаружен дым')
            self.assertIsNone(r['num'])
            self.assertEqual(r['state'], 'Обнаружен дым')

    def test_append_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            ev1 = st.clean_event(1, datetime(2026, 6, 1, 0, 0), 'Обнаружен дым')
            ev2 = st.clean_event(1, datetime(2026, 6, 1, 0, 0), 'Обнаружен дым')   # дубль Н6
            st.append([ev1, ev2])
            self.assertEqual(st.con.sql('SELECT count(*) FROM ev_all').fetchone()[0], 1)

    def test_retention_keeps_guard(self):
        with tempfile.TemporaryDirectory() as d:
            st = store_with_ch(Path(d))
            st.append([st.clean_event(1, datetime(2026, 6, 1, 0, 0), 'Обнаружен дым'),
                       st.clean_event(2, datetime(2026, 6, 1, 0, 0), 'Поставлена')])  # свежие
            st.append([st.clean_event(2, datetime(2025, 12, 1, 0, 0), 'Снята'),   # старая охрана
                       st.clean_event(1, datetime(2025, 12, 1, 0, 0), 'Газ')])    # старый дым
            swept = st.retention_sweep(datetime(2026, 6, 2))
            self.assertEqual(swept, 1)                 # старая не-охрана ушла
            self.assertEqual(st.con.sql('SELECT count(*) FROM ev_all WHERE stype '
                                        "= 'Состояние охраны'").fetchone()[0], 2)   # охрана цела

    def test_guard_date_pattern(self):
        self.assertTrue(GUARD_DATE.match('01.01.1970 00:00:00'))
        self.assertFalse(GUARD_DATE.match('Обнаружен дым'))

    def test_bulk_import(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            j = tmp / 'journal'
            j.mkdir()
            st = store_with_ch(tmp)
            write_journal(j / 'ext-journal-2019.csv', [
                ['1', 1, '2019-01-05', '07:00:00', '1', 'Обнаружен дым'],
                ['2', 1, '2019-01-05', '07:00:00', '1', 'Обнаружен дым'],   # дубль (канал, время, значение)
                ['3', 2, '2019-01-05', '07:00:02', '0', '01.01.1970'],       # охрана: мусорная дата
                ['4', 999, '2019-01-05', '07:00:03', '1', '42'],             # нет в справочнике
                ['5', 1, '2026-08-05', '07:00:04', '1', '1.5'],              # вне until
            ])
            import svc as config
            self.addCleanup(st.close)
            config.JOURNAL = j
            n = st.bulk_import(until=datetime(2026, 1, 1), years=[2019])
            self.assertEqual((n, st.con.sql('SELECT count(*) FROM ev_all').fetchone()[0]),
                             (1, 1))                     # только первая строка, дубль схлопнулся
            row = st.con.sql('SELECT object_id, num, state FROM ev_all').fetchone()
            self.assertEqual((row[0], row[2]), (5122, 'Обнаружен дым'))


if __name__ == '__main__':
    unittest.main()