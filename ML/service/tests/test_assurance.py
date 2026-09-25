import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from assurance import assurance
from snapshot import evidence, freshness
import storage


class AssuranceTest(unittest.TestCase):
    def test_floor_for_fresh_singleton(self):
        """Первый час, одиночный тип, без свидетелей — нижняя граница, но не ноль."""
        self.assertAlmostEqual(assurance(0, 0, 0), 0.10, places=9)

    def test_monotone_in_each_carrier(self):
        """Уверенность растёт с длительностью, согласием типов и числом каналов."""
        base = assurance(1, 1, 1)
        self.assertGreaterEqual(assurance(24, 1, 1), base)
        self.assertGreaterEqual(assurance(1, 5, 1), base)
        self.assertGreaterEqual(assurance(1, 1, 3), base)

    def test_saturates_at_one(self):
        self.assertLessEqual(assurance(240, 5, 10), 1.0)

    def test_evidence_families_from_hot_store(self):
        """§9.7: свидетели тревоги — события семейств типа, самого свежего первым."""
        with tempfile.TemporaryDirectory() as d:
            st = storage.HotStore(Path(d) / 'hot.duckdb')
            now = datetime(2026, 9, 23, 7, 0)
            rows = [
                (1, 100, now - timedelta(minutes=5), 'Датчик дыма', 'Обнаружен дым', None, 'Обнаружен дым'),
                (1, 101, now - timedelta(minutes=10), 'Датчик температуры', None, 36.5, '36.5'),
                (1, 200, now - timedelta(minutes=3), 'Газовый датчик', 'Обнаружен газ', None, 'Обнаружен газ'),
                (2, 300, now - timedelta(minutes=2), 'Датчик дыма', 'Обнаружен дым', None, 'Обнаружен дым')]
            st.con.executemany('INSERT INTO ev_all VALUES (?,?,?,?,?,?,?)', rows)
            fire = evidence(st, 1, 'fire', now)
            self.assertEqual({e['sensor_id'] for e in fire}, {100, 101})
            self.assertEqual(fire[0]['sensor_id'], 100)   # самым свежим первым (ts desc)
            self.assertEqual(fire[0]['value'], 'Обнаружен дым')
            gas = evidence(st, 1, 'gas', now)
            self.assertEqual([e['sensor_id'] for e in gas], [200])
            # отказ датчика (sensor): дым и температура — свои семьи, газ — нет
            sensor = evidence(st, 1, 'sensor', now)
            self.assertEqual({e['sensor_id'] for e in sensor}, {100, 101})
            self.assertEqual([e['sensor_id'] for e in evidence(st, 2, 'fire', now)], [300])

    def test_evidence_case_insensitive_reference_names(self):
        """stype из справочника — «Датчик дыма», «Датчик температуры» (паттерн с заглавной)."""
        with tempfile.TemporaryDirectory() as d:
            st = storage.HotStore(Path(d) / 'hot.duckdb')
            now = datetime(2026, 9, 23, 7, 0)
            rows = [
                (1, 100, now - timedelta(minutes=4), 'Датчик дыма', 'Обнаружен дым', None, 'Обнаружен дым'),
                (1, 101, now - timedelta(minutes=3), 'Состояние насоса', None, 1, '1')]
            st.con.executemany('INSERT INTO ev_all VALUES (?,?,?,?,?,?,?)', rows)
            # «Насос» против «Состояние насоса»: pump входит в flood и equipment
            self.assertEqual([e['sensor_id'] for e in evidence(st, 1, 'flood', now)], [101])

    def test_freshness_matches_reference_names_case_insensitive(self):
        with tempfile.TemporaryDirectory() as d:
            st = storage.HotStore(Path(d) / 'hot.duckdb')
            now = datetime(2026, 9, 23, 7, 0)
            st.con.execute(
                "INSERT INTO ev_all VALUES (?,?,?,?,?,?,?)",
                (1, 100, now - timedelta(minutes=4), 'Датчик дыма', 'Обнаружен дым', None, 'Обнаружен дым'))
            fr = freshness(st, now)
            self.assertLess(fr['fire'], 0.1)
            # без событий — крайняя мера, а не 0 часов
            st.con.execute('DELETE FROM ev_all')
            self.assertEqual(freshness(st, now)['fire'], 24.0 * 30)


if __name__ == '__main__':
    unittest.main()
