import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from clock import Clock


class ClockTest(unittest.TestCase):
    def test_live_once_per_hour_after_delay_and_persisted(self):
        with tempfile.TemporaryDirectory() as d:
            now = [datetime(2026, 9, 26, 10, 1)]
            c = Clock('live', Path(d) / 'clock.json', now=lambda: now[0])
            self.assertIsNone(c.due())                       # 10:01 — ещё ждём опоздавшие строки
            now[0] = datetime(2026, 9, 26, 10, 2, 30)
            self.assertEqual(c.due(), datetime(2026, 9, 26, 10))
            c.done(datetime(2026, 9, 26, 10))
            now[0] = datetime(2026, 9, 26, 10, 40)
            self.assertIsNone(c.due())                       # тот же час второй раз не считается
            again = Clock('live', Path(d) / 'clock.json', now=lambda: now[0])
            self.assertIsNone(again.due())                   # и после рестарта
            now[0] = datetime(2026, 9, 26, 14, 5)
            self.assertEqual(again.due(), datetime(2026, 9, 26, 14))   # пропуск не досчитывается
            again.done(datetime(2026, 9, 26, 14))
            self.assertEqual(again.gaps, 3)

    def test_replay_steps_from_start_then_from_state(self):
        with tempfile.TemporaryDirectory() as d:
            c = Clock('replay:2026-01-01T07:00:3600000', Path(d) / 'clock.json')
            self.assertEqual(c.label, 'replay')
            t = c.wait()
            self.assertEqual(t, datetime(2026, 1, 1, 7))
            c.done(t)
            self.assertEqual(c.wait(), datetime(2026, 1, 1, 8))
            c2 = Clock('replay:2026-01-01T07:00:3600000', Path(d) / 'clock.json')
            self.assertEqual(c2.due(), datetime(2026, 1, 1, 8))
            self.assertEqual(Clock('live', Path(d) / 'clock.json').last, None)   # другой режим — своё

    def test_bad_spec(self):
        with self.assertRaises(ValueError):
            Clock('fast', Path(tempfile.mkdtemp()) / 'c.json')


if __name__ == '__main__':
    unittest.main()
