"""Аварии и слепота по факту (INTEGRATION.md §13.11): температура по медиане, маршрут нарушителя,
слепота объекта; полное молчание в окне работ и без дублей с пожаром."""
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import fact as factmod
import labels
import storage
import test_core as tc

T = datetime(2026, 1, 20, 11)          # такт: час 10:00–11:00 закончился
COLL, TEMP, GUARD, BLIND, FLICK, SILENT = 60000, 60001, 60002, 60003, 60004, 60005


class Sink(list):
    def send(self, msg):
        self.append(msg)


class AccidentsTest(unittest.TestCase):
    """Сервис из test_core (настройки, аудит) без его тестов; горячая база — своя."""
    tearDown, make = tc.CoreTest.tearDown, tc.CoreTest.make

    def setUp(self):
        tc.CoreTest.setUp(self)
        st = storage.HotStore(Path(self.tmp.name) / 'hot.duckdb')
        self.addCleanup(st.close)
        st.con.execute(f"INSERT INTO obj VALUES ({COLL}, 2, NULL, 'коллектор', 'Объект 1')")
        for o in (TEMP, GUARD, BLIND, FLICK, SILENT):
            st.con.execute(f"INSERT INTO obj VALUES ({o}, 3, {COLL}, 'подсистема', 'Объект {o - COLL + 1}')")
        chans = [(11, 'Датчик температуры', TEMP), (12, 'Датчик температуры', TEMP),
                 (13, 'Датчик температуры', TEMP), (14, 'Датчик температуры', TEMP), (15, 'Датчик дыма', TEMP),
                 (21, 'Состояние охраны', GUARD), (22, 'КД Дверь', GUARD), (23, 'КД Дверь', GUARD),
                 (24, 'Датчик движения', GUARD)]
        chans += [(30 + i, 'Датчик дыма', BLIND) for i in range(5)]
        chans += [(40 + i, 'Датчик дыма', FLICK) for i in range(5)]
        chans += [(50 + i, 'Датчик дыма', SILENT) for i in range(5)]
        st.con.execute('INSERT INTO ch VALUES ' + ', '.join(
            f"({c}, 'с', '{s}', '', 'к{c}', {o})" for c, s, o in chans))
        self.svc.store, self.svc.sink = st, Sink()
        self.st = st

    def put(self, *rows):
        self.st.append([self.st.clean_event(c, ts, v) for c, ts, v in rows])

    def base(self, channels, value=12.0, days=30):
        """Отсчёт раз в сутки: база канала устоялась."""
        self.put(*[(c, T - timedelta(days=d, hours=1), value) for c in channels for d in range(1, days + 1)])

    def emit(self, t=T):
        self.st.ev_view(t)
        labels.build(self.st.con)
        eps = factmod.episodes(self.st, t)
        accs = factmod.accidents(self.st, t)
        self.svc.sink.clear()
        self.svc._emit_facts(t, 0, eps, self.svc.collectors(), 'v', 'live', accs)
        return {m['object_id']: m['types'] for m in self.svc.sink}

    # ----- температура ----------------------------------------------------------------------------
    def test_cold_on_two_channels(self):
        self.base([11, 12])
        self.put((11, datetime(2026, 1, 20, 10, 5), -5), (12, datetime(2026, 1, 20, 10, 20), -4))
        blk = self.emit()[TEMP]['temperature']
        self.assertTrue(blk['new'])
        self.assertEqual(blk['direction'], 'cold')
        self.assertEqual(blk['started_at'], '2026-01-20T10:00+03:00')
        self.assertEqual([(c['sensor_id'], c['value'], c['baseline']) for c in blk['channels']],
                         [(11, -5.0, 12.0), (12, -4.0, 12.0)])

    def test_single_reading_waits_for_second(self):
        """Один канал, один отсчёт — не объявление; второй отсчёт подряд — объявление."""
        self.base([11])
        self.put((11, datetime(2026, 1, 20, 10, 5), -5))
        self.assertNotIn(TEMP, self.emit())
        self.put((11, datetime(2026, 1, 20, 11, 5), -6))
        self.assertTrue(self.emit(T + timedelta(hours=1))[TEMP]['temperature']['new'])

    def test_cold_entrance_channel_is_not_anomaly(self):
        """Канал у входа, который всегда около нуля: ниже нормы, но не ниже своей базы на DEVIATION."""
        self.base([11, 12], value=1.0)
        self.put((11, datetime(2026, 1, 20, 10, 5), -2), (12, datetime(2026, 1, 20, 10, 20), -3))
        self.assertNotIn(TEMP, self.emit())

    def test_heat_with_fire_goes_into_fire_block(self):
        self.base([11, 12], value=20.0)
        self.put((15, datetime(2026, 1, 20, 10, 1), 'Обнаружен дым'),
                 (11, datetime(2026, 1, 20, 10, 5), 55), (12, datetime(2026, 1, 20, 10, 6), 60))
        types = self.emit()[TEMP]
        self.assertNotIn('temperature', types)
        self.assertEqual(types['fire']['temperature']['direction'], 'hot')

    def test_fire_from_temperature_only_is_announced_as_temperature(self):
        self.base([11, 12], value=20.0)
        self.put((11, datetime(2026, 1, 20, 10, 5), 55), (12, datetime(2026, 1, 20, 10, 6), 60))
        types = self.emit()[TEMP]
        self.assertNotIn('fire', types)
        self.assertEqual(types['temperature']['direction'], 'hot')

    # ----- маршрут --------------------------------------------------------------------------------
    def test_intrusion_route_in_order_and_collapsed(self):
        d = datetime(2026, 1, 20)
        self.put((21, d.replace(hour=8), 'На охране'),
                 (22, d.replace(hour=10, minute=5), 'Не замкнут'), (24, d.replace(hour=10, minute=8), 'Обнаружено движение'),
                 (23, d.replace(hour=10, minute=20), 'Не замкнут'), (24, d.replace(hour=10, minute=25), 'Обнаружено движение'),
                 (24, d.replace(hour=10, minute=26), 'Обнаружено движение'))
        blk = self.emit()[GUARD]['intrusion']
        self.assertEqual([(p['sensor_id'], p['stype']) for p in blk['route']],
                         [(22, 'КД Дверь'), (24, 'Датчик движения'), (23, 'КД Дверь'), (24, 'Датчик движения')])
        self.assertEqual(blk['route'][0]['at'], '2026-01-20T10:05:00+03:00')

    # ----- слепота --------------------------------------------------------------------------------
    def test_link_loss_is_blind_after_confirm(self):
        d = datetime(2026, 1, 20, 10)
        self.put(*[(30 + i, d + timedelta(minutes=i), 'Неопределен') for i in range(5)])
        blk = self.emit()[BLIND]['blind']
        self.assertEqual((blk['cause'], blk['possible_accident'], blk['new']), ('link', True, True))
        self.assertEqual(blk['started_at'], '2026-01-20T10:03+03:00')   # 4 из 5 каналов — 80%

    def test_flicker_is_not_announced(self):
        d = datetime(2026, 1, 20, 10)
        self.put(*[(40 + i, d + timedelta(minutes=i), 'Неопределен') for i in range(5)])
        self.put(*[(40 + i, d + timedelta(minutes=12 + i), 'Норма') for i in range(5)])
        self.assertNotIn(FLICK, self.emit())

    def test_power_event_before_link_loss_is_power(self):
        self.st.con.execute(f"INSERT INTO ch VALUES (60, 'с', 'ИБП', '', 'ИБП', {TEMP})")
        self.st._ch = None
        d = datetime(2026, 1, 20, 10)
        self.put((60, d - timedelta(minutes=10), 'Питание от батарей'))
        self.put(*[(30 + i, d + timedelta(minutes=i), 'Неопределен') for i in range(5)])
        self.assertEqual(self.emit()[BLIND]['blind']['cause'], 'power')

    def test_funnel_silence_is_blind(self):
        for i in range(4):
            self.st.apply_reference({'kind': 'channel.status', 'ид_канала_данных': 50 + i, 'status': 'silent',
                                     'since': f'2026-01-20T09:{10 * i:02d}:00+03:00', 'at': '2026-01-20T10:00:00+03:00'})
        blk = self.emit()[SILENT]['blind']
        self.assertEqual((blk['cause'], blk['started_at']), ('link', '2026-01-20T09:30+03:00'))

    # ----- плановые работы ------------------------------------------------------------------------
    def test_works_window_silences_fully(self):
        """§13.11: окно любых работ на коллекторе — ни объявления, ни MUTED, ни события аудита."""
        rows = [dict(r) for r in self.svc.works.rows]
        rows.append({'work_id': '901', 'object_id': str(COLL), 'work_kind': 'ТО АКМ', 'incident_types': 'equipment',
                     'starts_at': '2026-01-20 08:00', 'ends_at': '2026-01-20 18:00'})
        self.svc.handle(tc.env('settings.works', {'version': 2, 'rows': rows, 'reason': 'ТО'}))
        self.base([11, 12])
        d = datetime(2026, 1, 20, 10)
        self.put((11, d.replace(minute=5), -5), (12, d.replace(minute=20), -4))
        self.put(*[(30 + i, d + timedelta(minutes=i), 'Неопределен') for i in range(5)])
        out = self.emit()
        self.assertNotIn(TEMP, out)
        self.assertNotIn(BLIND, out)
        self.assertEqual(self.audit.of('forecast.muted'), [])
