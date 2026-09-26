import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import duckdb

import labels
import works as wk

HEAD = 'work_id;object_id;work_kind;incident_types;removed_sensor;starts_at;ends_at;source;comment\n'
ROWS = [
    '1;4068;ППР датчиков метана;gas;Газовый датчик;2026-01-12 00:00;2026-01-28 00:00;организатор;Объект 1',
    '2;;ППР датчиков метана;gas;Газовый датчик;2026-02-13 00:00;2026-02-28 00:00;организатор;пары нет',
    '3;8;ППР датчиков метана;gas;Газовый датчик;2024-02-20 00:00;2024-02-29 12:00;организатор;високосный',
    '4;8;ППР датчиков метана;gas;Газовый датчик;2026-03-02 00:00;2026-03-10 00:00;организатор;',
    '5;77;ТО насосов;flood,equipment;;2025-06-01 08:00;2025-06-01 20:00;главный диспетчер;',
]


def table(folder: Path, rows=ROWS) -> Path:
    p = folder / 'works_2026.csv'
    p.write_text(HEAD + '\n'.join(rows) + '\n', encoding='utf-8')
    return p


class WorksTest(unittest.TestCase):
    def test_same_windows_as_labels(self):
        """Окна сервиса = окна разметки Н10 (labels.works_windows) на тех же годах."""
        with tempfile.TemporaryDirectory() as d:
            src = table(Path(d))
            old, labels.WORKS = labels.WORKS, src
            try:
                con = duckdb.connect()
                years = [2023, 2024, 2025, 2026, 2027]
                labels.works_windows(con, years)
                sql = {(o, tuple(t), s, a, b) for o, t, s, a, b in
                       con.sql('SELECT object_id, types, sensor, a, b FROM works_win').fetchall()}
            finally:
                labels.WORKS = old
            py = {(w.object_id, w.types, w.sensor, w.a, w.b) for w in wk.windows(wk.read_rows(src), years)}
            self.assertEqual(py, sql)

    def test_nearest_year_tie_takes_earlier_and_feb29(self):
        rows = wk.validate(wk.read_rows(table(Path(tempfile.mkdtemp()), ROWS[2:4])))
        ws = [w for w in wk.windows(rows, [2025]) if w.shifted]
        self.assertEqual({w.work_id for w in ws}, {'3'})         # 2024 и 2026 на равном удалении — 2024
        self.assertEqual(ws[0].b, datetime(2025, 3, 7, 12))       # 29.02.2024 → 28.02.2025 + 7 суток

    def test_mute_covers_collector_and_only_row_types(self):
        with tempfile.TemporaryDirectory() as d:
            w = wk.Works(Path(d) / 'svc', seed=table(Path(d)))
            ts = datetime(2026, 1, 20, 5)
            m = w.mute(ts, [4068, 5000, 5001], {5000: 4068, 5001: 9})
            self.assertEqual(set(m), {(4068, 'gas'), (5000, 'gas')})
            self.assertEqual(m[(5000, 'gas')].work_id, '1')
            self.assertEqual(w.mute(datetime(2026, 1, 28, 0), [4068], {}), {})     # конец не входит
            self.assertIsNotNone(w.fact_window(5000, 4068, 'sensor', ts, 'Газовый датчик'))
            self.assertIsNone(w.fact_window(5000, 4068, 'sensor', ts, 'Датчик дыма'))

    def test_replace_versions_and_diff(self):
        with tempfile.TemporaryDirectory() as d:
            w = wk.Works(Path(d) / 'svc', seed=table(Path(d)))
            self.assertEqual(w.version, 1)
            rows = [dict(r) for r in w.rows if r['work_id'] != '2']
            rows[0]['ends_at'] = '2026-01-30 00:00'
            rows.append({'work_id': '9', 'object_id': '12', 'work_kind': 'покраска', 'incident_types': 'fire',
                         'starts_at': '2026-05-01 09:00', 'ends_at': '2026-05-01 18:00'})
            diff = w.replace(rows, 2, 'u1', 'новый наряд', datetime(2026, 4, 1, 10))
            self.assertEqual(diff, {'version': 2, 'added': ['9'], 'removed': ['2'], 'changed': ['1']})
            self.assertTrue((Path(d) / 'svc' / 'works_versions' / 'v1.csv').exists())
            again = wk.Works(Path(d) / 'svc')
            self.assertEqual((again.version, len(again.rows)), (2, 5))
            with self.assertRaises(ValueError):
                w.replace(rows, 2, 'u1', None, datetime(2026, 4, 1, 11))          # версия не новее
            with self.assertRaises(ValueError):
                w.replace(rows + [dict(rows[0])], 3, 'u1', None, datetime(2026, 4, 1, 11))  # номер повторяется
            bad = [dict(rows[0], incident_types='smoke')]
            with self.assertRaises(ValueError):
                w.replace(bad, 3, 'u1', None, datetime(2026, 4, 1, 11))


if __name__ == '__main__':
    unittest.main()
