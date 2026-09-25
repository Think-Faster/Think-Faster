"""M11: обратный поток решений (INTEGRATION §9.1) — схема, часы от витрины, действия, аудит §9.2."""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

import decisions
import features as ft


class Recorder:
    """Записывает вызовы rules.on_decision — чтобы не тащить полные настройки."""

    def __init__(self):
        self.calls = []

    def on_decision(self, object_id, tp, action, ts, mute_hours=None):
        self.calls.append((object_id, tp, action, ts, mute_hours))


class DecisionsTest(unittest.TestCase):
    def test_reject_decided_at(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {'object_id': 5122, 'type': 'gas', 'action': 'REJECT',
             'decided_at': '2026-01-04T12:25:00+03:00', 'reason_code': 'known_works',
             'user_id': 'disp-01'}])
        self.assertEqual(n, 1)
        oid, tp, action, h, mute = rules.calls[0]
        self.assertEqual((oid, tp, action), (5122, 'gas', 'REJECT'))
        expect = int((datetime(2026, 1, 4, 12, 25) - ft.T0)
                     .total_seconds() // 3600)      # +03:00 в сообщении = местное наивное время
        self.assertEqual(h, expect)
        self.assertIsNone(mute)

    def test_mute_with_hours_and_fallback(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {'object_id': 1, 'type': 'fire', 'action': 'MUTE', 'mute_hours': 24,
             'decided_at': '2026-01-04T12:00:00+03:00'},
            {'object_id': 2, 'type': 'gas', 'action': 'MUTE',
             'decided_at': '2026-01-04T12:00:00+03:00'}])
        self.assertEqual(n, 2)
        self.assertEqual(rules.calls[0][4], 24)
        self.assertIsNone(rules.calls[1][4])          # без срока — настройка правил по умолчанию

    def test_legacy_ts_key(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {'object_id': 7, 'type': 'flood', 'action': 'REOPEN', 'ts': '2026-01-04T12:00:00'}])
        self.assertEqual(n, 1)
        self.assertEqual(rules.calls[0][3], decisions.to_hour(datetime(2026, 1, 4, 12, 0)))

    def test_unknown_action_skipped(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {'object_id': 1, 'type': 'gas', 'action': 'SOMETHING',
             'decided_at': '2026-01-04T12:00:00+03:00'}])
        self.assertEqual(n, 0)
        self.assertEqual(rules.calls, [])

    def test_tape_and_confirmed_pass(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {'object_id': 1, 'type': 'gas', 'action': 'TAKE',
             'decided_at': '2026-01-04T12:00:00+03:00'},
            {'object_id': 1, 'type': 'gas', 'action': 'CONFIRMED',
             'decided_at': '2026-01-04T12:00:00+03:00'}])
        self.assertEqual(n, 2)
        self.assertEqual([c[2] for c in rules.calls], ['TAKE', 'CONFIRMED'])

    def test_missing_fields_skipped(self):
        rules = Recorder()
        n = decisions.apply(rules, [
            {}, {'action': 'REJECT'}, {'object_id': 'x', 'type': 'gas', 'action': 'REJECT',
                                       'decided_at': '2026-01-04T12:00:00+03:00'},
            {'object_id': 1, 'type': 'gas', 'action': 'REJECT'}])
        self.assertEqual(n, 0)

    def test_audit_writes_applied_decisions(self):
        """§9.2: применённые решения складываются в parquet-таблицу для обучения."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'audit.parquet'
            rules = Recorder()
            n = decisions.apply(rules, [
                {'object_id': 5122, 'type': 'gas', 'action': 'REJECT', 'reason_code': 'known_works',
                 'user_id': 'disp-01', 'task_id': 7,
                 'decided_at': '2026-01-04T12:25:00+03:00'},
                {'object_id': 5, 'type': 'gas', 'action': 'SOMETHING',
                 'decided_at': '2026-01-04T12:25:00+03:00'}], audit_path=p)
            self.assertEqual(n, 1)                    # неизвестное действие в аудит не пишется
            n2 = decisions.apply(rules, [
                {'object_id': 6, 'type': 'fire', 'action': 'MUTE', 'mute_hours': 3,
                 'decided_at': '2026-01-04T13:00:00+03:00'}], audit_path=p)
            self.assertEqual(n2, 1)
            df = pl.read_parquet(p)
            self.assertEqual(df.height, 2)            # оба вызова дописались к одной таблице
            rows = df.sort('object_id').rows(named=True)
            self.assertEqual([r['object_id'] for r in rows], [6, 5122])
            self.assertEqual(rows[0]['action'], 'MUTE')
            self.assertEqual(rows[1]['action'], 'REJECT')
            self.assertEqual(rows[1]['type'], 'gas')
            self.assertEqual(rows[1]['reason_code'], 'known_works')


if __name__ == '__main__':
    unittest.main()