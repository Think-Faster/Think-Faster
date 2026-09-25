"""M11: обратный поток решений (INTEGRATION §9.1) — схема, часы от витрины, действия."""
import unittest
from datetime import datetime, timedelta

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


if __name__ == '__main__':
    unittest.main()