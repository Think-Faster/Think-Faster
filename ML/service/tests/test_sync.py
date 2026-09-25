"""Сквозной тест такта: главный цикл tick() с факт-каналом и раздельными сообщениями.

Тяжёлые входы (витрина retro.snapshot, план-график) подменяются — проверяется склейка
сообщений: объект кадра витрины уходит одним сообщением с прогнозом и факт-блоком, объект
только «по факту» — отдельным сообщением (M8, §2.3).
"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

import main
import rules as rulesmod
import storage
import svc as config_svc
from settings import OperatingSettings
from threshold import Thresholds
from helpers import write_operating

TYPES = config_svc.TYPES
NOW = datetime(2026, 9, 23, 7, 0)


def now_min_t0(now: datetime) -> str:
    return (now - timedelta(minutes=6)).isoformat(sep=' ', timespec='seconds')


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame

    def __call__(self, store, t, meta, cal, seq=False):
        return self.frame, None


class FakePredictor:
    version = '2026-09-26-test'
    meta = {'features': [], 'stypes': []}

    def has_nets(self):
        return False

    def predict(self, frame, seq_in):
        n = len(frame)
        return {tp: np.zeros(n, np.float32) for tp in TYPES}

    def reasons(self, frame, i, tp, k=3):
        return [{'feature': 'smoke_24h', 'value': 1.0} for _ in range(k)]

    def bootstrap_history(self, year=2025):
        hh = np.arange(0, 3000, 1, np.int64)
        pp = np.full(3000, 0.4, np.float32)
        return {tp: (hh.copy(), pp.copy()) for tp in TYPES}


class FakeObserver:
    def observe(self, **kw):
        return None

    def dump(self):
        pass


class ListSink:
    def __init__(self):
        self.items = []

    def send(self, msg: dict):
        self.items.append(msg)


def seed_store(d: str):
    """Горячий журнал: объект витрины 5122, факт-эпизод оборудования у 90 (без кадра)."""
    st = storage.HotStore(Path(d) / 'hot.duckdb')
    st.con.sql("""CREATE TABLE obj3(object_id INT PRIMARY KEY, collector_id INT, kind VARCHAR, name VARCHAR)""")
    st.con.execute("INSERT INTO obj3 VALUES (5122, 9005, 'guardObject', 'ЦИТА'), (90, 9090, 'object', 'ИР')")
    st.con.sql("CREATE TABLE guard(object_id INT, collector_id INT, ts TIMESTAMP, armed BOOLEAN)")
    st.con.sql("CREATE OR REPLACE VIEW ev AS SELECT object_id, channel_id, ts, stype, state, num FROM ev_all")
    st.con.sql("""CREATE TABLE trig(object_id INT, channel_id INT, ts TIMESTAMP, stype VARCHAR,
                                    what VARCHAR, type VARCHAR)""")
    st.con.sql("""CREATE TABLE inc(object_id INT, collector_id INT, type VARCHAR, t0 TIMESTAMP,
                                   t1 TIMESTAMP, rows INT, channels INT, noise VARCHAR,
                                   confirmed BOOLEAN, primary_ BOOLEAN)""")
    t0 = NOW - timedelta(minutes=6)      # эпизод шёл в окне последнего часа
    st.con.execute(
        f"""INSERT INTO inc VALUES (90, 9090, 'equipment', TIMESTAMP '{t0.isoformat()}',
            TIMESTAMP '{NOW.isoformat()}', 2, 1, NULL, false, true)""")
    st.con.execute(
        "INSERT INTO trig VALUES (?,?,?,?,?,?), (?,?,?,?,?,?)",
        (90, 7001, t0, 'Состояние насоса', 'Неисправен', 'equipment',
         90, 7001, t0.replace(minute=56), 'Состояние насоса', 'Неисправен', 'equipment'))
    return st


class SyncTest(unittest.TestCase):
    def test_tick_facts_and_split_messages(self):
        with tempfile.TemporaryDirectory() as d:
            st = seed_store(d)
            h = main.hour_index(NOW) - 1
            frame = pl.DataFrame({'object_id': [5122], 'h': [h]})

            settings = OperatingSettings.load(write_operating(Path(d)))
            hist = Thresholds(Path(d) / 'history.parquet')
            hist.bootstrap(FakePredictor().bootstrap_history(), settings, h,
                           model_version=FakePredictor.version)
            sink = ListSink()

            with mock.patch('main.planmute.windows', return_value=False), \
                 mock.patch('main.planmute.mask', return_value={}), \
                 mock.patch('main.snapmod.snapshot', FakeSnapshot(frame)):
                res = main.tick(st, FakePredictor(), hist, rulesmod.RuleState(settings), settings,
                                sink, FakeObserver(), NOW)

            self.assertEqual(res['objects'], 1)
            self.assertEqual(res['facts'], 1)
            self.assertEqual(len(sink.items), 2)

            msg_in, msg_out = sorted(sink.items, key=lambda m: m['object_id'])
            self.assertEqual(msg_in['object_id'], 90)          # только по факту — отдельное сообщение
            self.assertEqual(msg_out['object_id'], 5122)
            self.assertEqual(msg_out['hour_end'], main.hour_iso(h))
            self.assertNotIn('facts', msg_out)                 # у объекта кадра фактов нет
            self.assertIn('data_freshness_hours', msg_out)
            self.assertTrue(msg_out['types']['fire']['alarm'])  # адвайзер внутри такта
            self.assertIn('recommendation', msg_out['types']['fire'])
            self.assertIn('object_recommendation', msg_out)

            fb = msg_in['facts']['equipment']
            t0 = now_min_t0(NOW)
            self.assertEqual(fb['recommendation']['mode'], 'факт')
            self.assertEqual(fb['episode_t0'], t0)
            self.assertGreaterEqual(fb['since_hours'], 0)
            self.assertFalse(msg_in['types']['equipment']['alarm'])   # прогноз не тронут

    def test_hour_end_has_zone(self):
        with tempfile.TemporaryDirectory() as d:
            st = seed_store(d)
            h = main.hour_index(NOW) - 1
            self.assertEqual(main.hour_iso(h)[-6:], config_svc.TZ)


if __name__ == '__main__':
    unittest.main()