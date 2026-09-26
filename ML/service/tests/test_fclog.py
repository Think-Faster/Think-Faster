import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from fclog import ForecastLog

TYPES = ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')


class ForecastLogTest(unittest.TestCase):
    def test_history_with_statuses_and_rewrite(self):
        with tempfile.TemporaryDirectory() as d:
            log = ForecastLog(Path(d) / 'fc.duckdb')
            objs = np.array([10, 20])
            t0 = datetime(2026, 1, 12, 5)
            for k in range(3):
                sc = {tp: np.array([0.1 * k, 0.9]) for tp in TYPES}
                st = [(20, 'gas', 'MUTED', 'works', '1'), (20, 'fire', 'ALARM', None, None)]
                log.write(t0 + timedelta(hours=k), objs, sc, {tp: 0.8 for tp in TYPES}, st, '2026-09-25+equipment_v1', 3)
            log.write(t0, objs, {tp: np.array([0.5, 0.5]) for tp in TYPES}, {tp: 0.7 for tp in TYPES}, [], 'x', 4)
            h = log.history(20, 'gas', t0, t0 + timedelta(hours=3))
            self.assertEqual([r['status'] for r in h], [None, 'MUTED', 'MUTED'])   # первый час переписан
            self.assertEqual(h[0]['threshold'], 0.7)
            self.assertEqual(h[1]['reason'], 'works')
            self.assertEqual(h[1]['hour_end'], '2026-01-12T06:00+03:00')
            self.assertEqual(log.last_hour(), t0 + timedelta(hours=2))
            log.sweep(t0 + timedelta(days=100, hours=1))                      # край — t0+1ч: уходит только t0
            self.assertEqual(len(log.history(20, 'gas', t0, t0 + timedelta(hours=3))), 2)

    def test_reader_thread(self):
        with tempfile.TemporaryDirectory() as d:
            log = ForecastLog(Path(d) / 'fc.duckdb')
            t0 = datetime(2026, 1, 1, 1)
            log.write(t0, np.array([1]), {tp: np.array([0.3]) for tp in TYPES}, {tp: 0.5 for tp in TYPES}, [], 'v', 1)
            got = []
            th = threading.Thread(target=lambda: got.append(log.history(1, 'fire', t0, t0 + timedelta(hours=1))))
            th.start(); th.join()
            self.assertEqual(len(got[0]), 1)


if __name__ == '__main__':
    unittest.main()
