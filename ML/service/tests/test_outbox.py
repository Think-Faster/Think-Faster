import unittest

from outbox import build_message, fill_type


class OutboxTest(unittest.TestCase):
    def test_message_shape_schema1(self):
        m = build_message(5122, '2026-09-23T07:00:00+03:00', 'export-2026-09-23', ['fire', 'gas'])
        self.assertEqual(m['schema'], 1)
        self.assertEqual(m['object_id'], 5122)
        self.assertEqual(m['types']['fire'], {'score': 0.0, 'threshold': 0.0, 'alarm': False})

    def test_reasons_only_for_alarms(self):
        m = build_message(5122, 'h', 'v', ['fire'])
        fill_type(m, 'fire', score=0.9, threshold=0.5, alarm=True, since_hours=4,
                  reasons=[{'feature': 'smoke_24h', 'value': 3.0}])
        self.assertEqual(m['types']['fire']['since_hours'], 4)
        self.assertIn('reasons', m['types']['fire'])
        m2 = build_message(5123, 'h', 'v', ['fire'])
        fill_type(m2, 'fire', score=0.1, threshold=0.5, alarm=False, since_hours=4,
                  reasons=[{'feature': 'x', 'value': 1.0}])
        self.assertNotIn('reasons', m2['types']['fire'])


if __name__ == '__main__':
    unittest.main()