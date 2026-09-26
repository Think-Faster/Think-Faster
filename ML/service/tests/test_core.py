"""Команды 13.3 в ядре сервиса: идемпотентность, версии таблиц, решения диспетчера, аудит 13.4."""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import svc as config
import core
import features as ft
import labels
from threshold import Thresholds

ML = Path(__file__).resolve().parents[2]
WHEN = '2026-01-04T12:00:00+03:00'


class FakeAudit:
    def __init__(self):
        self.events = []

    def event(self, event_type, outcome='success', **kw):
        self.events.append({'event_type': event_type, 'outcome': outcome, **kw})

    def of(self, name):
        return [e for e in self.events if e['event_type'] == name]


def env(kind, payload, cid=None):
    env.n = getattr(env, 'n', 0) + 1
    return {'schema': 1, 'command_id': cid or f'c{env.n}', 'kind': kind, 'issued_at': WHEN,
            'issued_by': {'sub': 'u-1', 'login': 'petrova'}, 'request_id': 'r1', 'payload': payload}


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.saved = {k: getattr(config, k) for k in ('RULES_STATE', 'SERVICE_STATE', 'COMMANDS_SEEN',
                                                     'AUDIT_SPOOL', 'HISTORY')}
        self.gaps0, self.holes0, self.works0 = list(config.GAPS), list(ft.HOLES), labels.WORKS
        for k, name in (('RULES_STATE', 'rules.json'), ('SERVICE_STATE', 'state.json'),
                        ('COMMANDS_SEEN', 'commands.json'), ('AUDIT_SPOOL', 'audit.jsonl'),
                        ('HISTORY', 'history.parquet')):
            setattr(config, k, d / 'out' / name)
        self.dir = d / 'settings'
        self.audit = FakeAudit()
        self.svc = self.make()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(config, k, v)
        config.GAPS[:] = self.gaps0
        ft.HOLES[:] = self.holes0
        labels.WORKS = self.works0
        self.tmp.cleanup()

    def make(self):
        return core.Service(settings_dir=self.dir, seed=ML / 'settings', audit=self.audit)

    # ----- сообщение прогноза ---------------------------------------------------------------------
    def test_forecast_carries_confidence_evidence_silence_staleness(self):
        """§2.3, §7, §9.7: уверенность по калибровке, минус цена молчания дыма (раздел 55); свидетели —
        последнее событие каналов семейств типа; несвежесть — у каждого типа, чей поток отстал."""
        import storage

        class P:
            version = 'v'

            def reasons(self, frame, i, tp, k=3):
                return [{'feature': 'smoke_24h', 'value': 3.0}]

            def confidence(self, tp, score):
                return np.full(len(score), 0.5)

        class Sink(list):
            def send(self, msg):
                self.append(msg)

        st = storage.HotStore(Path(self.tmp.name) / 'hot.duckdb')
        self.addCleanup(st.close)
        st.con.execute("INSERT INTO ch VALUES (1, 'ПС', 'Датчик дыма', '', 'Дым', 5122),"
                       "(4, 'Климат', 'Датчик температуры', '', 'Т', 5122),"
                       "(3, 'Газ', 'Газовый датчик', '', 'Г', 5123)")
        st.append([st.clean_event(1, datetime(2026, 1, 4, 11, 30), 'Обнаружен дым'),
                   st.clean_event(4, datetime(2026, 1, 4, 10), '25'),
                   st.clean_event(3, datetime(2026, 1, 4, 11), '0.1')])
        st.apply_reference({'kind': 'channel.status', 'ид_канала_данных': 1, 'status': 'silent', 'at': WHEN})
        self.svc.store, self.svc.predictor, self.svc.sink = st, P(), Sink()
        objects = np.array([5122, 5123])
        scores = {tp: np.array([0.99, 0.1]) for tp in config.TYPES}
        applied = {tp: (np.array([tp == 'fire', False]), np.array([2 if tp == 'fire' else 0, 0]))
                   for tp in config.TYPES}
        fres = {tp: 0.2 for tp in config.TYPES} | {'gas': 5.0}
        n = self.svc._emit_forecasts(datetime(2026, 1, 4, 12), 1, None, objects, scores,
                                     {tp: 0.9 for tp in config.TYPES}, applied, 'v', 'live', fres)
        self.assertEqual(n, 1)
        a, b = self.svc.sink
        fire = a['types']['fire']
        self.assertAlmostEqual(fire['confidence'], 0.5 * (1 - config.SILENCE_COST['fire']['smoke']))
        self.assertEqual(fire['silent'], ['smoke'])
        self.assertEqual([e['sensor_id'] for e in fire['evidence']], [1, 4])
        self.assertEqual(fire['evidence'][0], {'sensor_id': 1, 'ts': '2026-01-04T11:30:00+03:00',
                                               'value': 'Обнаружен дым'})
        self.assertEqual((a['types']['gas']['stale_hours'], b['types']['gas']['stale_hours']), (5.0, 5.0))
        self.assertNotIn('stale_hours', fire)
        self.assertEqual(a['types']['sensor']['silent'], ['smoke'])        # молчание видно и без тревоги
        self.assertNotIn('confidence', a['types']['sensor'])
        self.assertNotIn('evidence', b['types']['fire'])

    # ----- конверт и повторы ----------------------------------------------------------------------
    def test_broken_envelope_is_rejected(self):
        for bad in ({'schema': 2}, {'schema': 1, 'command_id': 'x', 'kind': 'nope', 'payload': {}},
                    {'schema': 1, 'command_id': 'x', 'kind': 'decision.reject', 'payload': []}):
            with self.assertRaises(core.CommandError):
                self.svc.handle(bad)
        with self.assertRaises(core.CommandError):
            self.svc.handle(env('decision.reject', {'object_id': 1, 'type': 'пожар'}))

    def test_duplicate_command_is_acked_without_action(self):
        e = env('decision.reject', {'object_id': 5122, 'type': 'gas', 'reason_code': 'false'}, cid='dup')
        self.svc.handle(e)
        self.assertEqual(self.svc.handle(e), {'status': 'duplicate'})
        self.assertEqual(len(self.audit.of('forecast.rejected')), 1)
        again = self.make()                                 # после рестарта command_id помнится
        self.assertEqual(again.handle(e), {'status': 'duplicate'})
        self.assertIn((5122, 'gas'), again.rules.rejections)   # и решение пережило рестарт

    # ----- решения диспетчера ---------------------------------------------------------------------
    def test_reject_rule_only_for_rejectable_types(self):
        r = self.svc.handle(env('decision.reject', {'object_id': 5122, 'type': 'gas', 'reason_code': 'works'}))
        self.assertTrue(r['rule'])
        r = self.svc.handle(env('decision.reject', {'object_id': 5122, 'type': 'fire', 'reason_code': 'false'}))
        self.assertFalse(r['rule'])                         # у пожара только история
        ev = self.audit.of('forecast.rejected')
        self.assertEqual([e['details']['reject_k'] for e in ev], [0.2, None])
        self.assertTrue(ev[0]['object_id'].startswith('5122:gas:'))
        self.assertEqual(ev[0]['actor_login'], 'petrova')

    def test_mute_reopen_and_confirmed(self):
        self.svc.handle(env('decision.mute', {'object_id': 7, 'type': 'flood', 'until': '2026-01-05T12:00:00+03:00'}))
        h = core.hour_index(datetime(2026, 1, 4, 12)) - 1
        self.assertEqual(self.svc.rules.status(7, 'flood', h)[0], 'MUTED')
        self.assertEqual(self.audit.of('forecast.muted')[0]['details']['reason'], 'decision')
        with self.assertRaises(core.CommandError):          # until в прошлом
            self.svc.handle(env('decision.mute', {'object_id': 7, 'type': 'flood', 'until': '2026-01-01T00:00'}))
        self.svc.handle(env('decision.reopen', {'object_id': 7, 'type': 'flood'}))
        self.assertIsNone(self.svc.rules.status(7, 'flood', h))
        # отклонили, а происшествие подтвердилось — «переросло в эпизод»
        self.svc.handle(env('decision.reject', {'object_id': 7, 'type': 'gas', 'reason_code': 'false'}))
        r = self.svc.handle(env('decision.confirmed', {'object_id': 7, 'type': 'gas', 'incident_id': 42,
                                                       'occurred_at': WHEN}))
        self.assertTrue(r['recurred'])
        rec = self.audit.of('forecast.recurred')[0]['details']
        self.assertEqual((rec['was'], rec['incident_id'], rec['within_horizon']), ('REJECTED', 42, True))

    # ----- таблицы главного диспетчера ------------------------------------------------------------
    def test_operating_versions(self):
        cur = self.svc.settings.as_dict()
        new = {**cur, 'version': cur['version'] + 1, 'reason': 'меньше тревог по пожару',
               'types': {**cur['types'], 'fire': {'share': 0.02, 'reject_k': None}}}
        r = self.svc.handle(env('settings.operating', new))
        self.assertEqual(r['changes'], {'fire': {'share': [cur['types']['fire']['share'], 0.02]}})
        self.assertEqual(self.svc.settings.share('fire'), 0.02)
        self.assertTrue((self.dir / 'operating_versions' / f'v{cur["version"]}.json').exists())
        self.assertEqual(json.loads((self.dir / 'operating.json').read_text(encoding='utf-8'))['version'],
                         cur['version'] + 1)
        self.assertEqual(self.audit.of('settings.changed')[0]['details']['to'], cur['version'] + 1)
        # тот же номер ещё раз — снимок не новее, отброшен без аудита
        self.assertEqual(self.svc.handle(env('settings.operating', new))['status'], 'stale')
        self.assertEqual(len(self.audit.of('settings.changed')), 1)
        bad = {**new, 'version': new['version'] + 1, 'types': {**new['types'], 'fire': {'share': 0.9, 'reject_k': None}}}
        with self.assertRaises(core.CommandError):
            self.svc.handle(env('settings.operating', bad))

    def test_works_table_versions(self):
        rows = [dict(r) for r in self.svc.works.rows]
        rows.append({'work_id': '900', 'object_id': '4068', 'work_kind': 'ТО насосов',
                     'incident_types': 'flood', 'starts_at': '2026-02-01 08:00', 'ends_at': '2026-02-01 20:00'})
        r = self.svc.handle(env('settings.works', {'version': 2, 'rows': rows, 'reason': 'добавлено ТО'}))
        self.assertEqual(r['added'], ['900'])
        self.assertEqual(labels.WORKS, self.svc.works.path)
        m = self.svc.works.mute(datetime(2026, 2, 1, 9), [4068], {})
        self.assertEqual(m[(4068, 'flood')].work_id, '900')
        self.assertEqual(self.audit.of('works.changed')[0]['details']['added'], ['900'])
        with self.assertRaises(core.CommandError):
            self.svc.handle(env('settings.works', {'version': 3, 'rows': rows + [rows[0]]}))   # номер повторяется

    def test_gaps_need_retrain_and_leave_history(self):
        hist = Thresholds()
        h0 = core.hour_index(datetime(2026, 1, 1))
        hh = np.repeat(np.arange(h0, h0 + 48), 3)
        hist.bootstrap({tp: (hh, np.linspace(0, 1, len(hh)).astype(np.float32)) for tp in config.TYPES},
                       self.svc.settings, h0 + 47)
        self.svc.history = hist
        rows = self.svc.gaps.rows + [{'a': '2026-01-01 10:00', 'b': '2026-01-01 14:00', 'comment': 'сбой шлюза'}]
        r = self.svc.handle(env('settings.gaps', {'version': 2, 'rows': rows, 'reason': 'брак'}))
        self.assertTrue(r['retrain_needed'] and self.svc.state['retrain_needed'])
        self.assertIn((datetime(2026, 1, 1, 10), datetime(2026, 1, 1, 14)), config.GAPS)
        self.assertIn((datetime(2026, 1, 1, 10), datetime(2026, 1, 1, 14)), ft.HOLES)
        self.assertEqual(len(hist.history('fire')[0]), len(hh) - 4 * 3)
        self.assertTrue(self.svc.gaps.covers(datetime(2026, 1, 1, 11)))
        self.assertEqual(self.audit.of('gaps.changed')[0]['details']['history_rows_dropped'], 6 * 4 * 3)

    # ----- модель и переобучение ------------------------------------------------------------------
    def test_switch_needs_models_and_known_version(self):
        with self.assertRaises(RuntimeError):               # временная ошибка — nack с повтором
            self.svc.handle(env('model.switch', {'type': 'equipment', 'version_id': 2, 'reason': 'тест'}))

        class P:
            version = 'x'
            def use_version(self, tp, n):
                if n == 9:
                    raise FileNotFoundError('нет версии 9')
                self.version = f'x+{tp}_v{n}'
            def bootstrap_history(self, year):
                return {tp: (np.arange(10), np.linspace(0, 1, 10)) for tp in config.TYPES}

        self.svc.predictor, self.svc.history = P(), Thresholds()
        r = self.svc.handle(env('model.switch', {'type': 'equipment', 'version_id': 'v2', 'reason': 'меньше ложных'}))
        self.assertEqual((r['from'], r['to']), (1, 2))
        self.assertEqual(self.svc.state['versions']['equipment'], 2)
        self.assertEqual(self.audit.of('model.switched')[0]['details']['to'], 2)
        with self.assertRaises(core.CommandError):
            self.svc.handle(env('model.switch', {'type': 'equipment', 'version_id': 9}))

    def test_retrain_off_is_denied_in_audit(self):
        r = self.svc.handle(env('retrain.request', {'strategy': 'all', 'reason': 'брак'}))
        self.assertEqual(r['status'], 'disabled')
        self.assertEqual(self.audit.of('retrain.requested')[0]['outcome'], 'denied')


if __name__ == '__main__':
    unittest.main()
