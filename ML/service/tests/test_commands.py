"""Приём команд 13.3 и журнала M1 без брокеров: ack/reject/повтор, DLQ, коммит после записи."""
import json
import threading
import unittest
from types import SimpleNamespace

import commands
from core import CommandError
from ingest import RawEvent, consume


class FakeService:
    def __init__(self, fail=None):
        self.fail, self.done = fail, []

    def handle(self, env):
        if env.get('kind') == 'broken':
            raise CommandError('нет поля')
        if self.fail:
            raise self.fail
        self.done.append(env['command_id'])
        return {'status': 'ok'}


def body(cid='c1', kind='decision.reject'):
    return json.dumps({'command_id': cid, 'kind': kind}).encode()


class Channel:
    def __init__(self):
        self.calls = []

    def basic_ack(self, tag):
        self.calls.append(('ack', tag))

    def basic_reject(self, tag, requeue):
        self.calls.append(('reject', tag, requeue))

    def basic_nack(self, tag, requeue):
        self.calls.append(('nack', tag, requeue))


class Conn:
    def __init__(self):
        self.later = []

    def call_later(self, delay, fn):
        self.later.append(delay)
        fn()


class CommandsTest(unittest.TestCase):
    def test_ack_after_handle(self):
        svc = FakeService()
        self.assertEqual(commands.Consumer(svc).decide(body()), commands.ACK)
        self.assertEqual(svc.done, ['c1'])

    def test_broken_goes_to_dlq_without_retry(self):
        c = commands.Consumer(FakeService())
        for b in (b'not json', b'\xff\xfe', body(kind='broken')):
            self.assertEqual(c.decide(b), commands.REJECT)

    def test_transient_retries_then_dlq(self):
        c = commands.Consumer(FakeService(fail=RuntimeError('модель ещё не поднялась')), retries=5)
        got = [c.decide(body('c9')) for _ in range(5)]
        self.assertEqual(got, [commands.RETRY] * 4 + [commands.REJECT])
        self.assertNotIn('c9', c.attempts)

    def test_quorum_delivery_count_is_honoured(self):
        c = commands.Consumer(FakeService(fail=OSError('диск')), retries=5)
        self.assertEqual(c.decide(body('c7'), {'x-delivery-count': 4}), commands.REJECT)

    def test_on_message_maps_verdicts(self):
        ch, conn = Channel(), Conn()
        ok = commands.Consumer(FakeService())
        ok.on_message(conn, ch, SimpleNamespace(delivery_tag=1), SimpleNamespace(headers=None), body())
        ok.on_message(conn, ch, SimpleNamespace(delivery_tag=2), SimpleNamespace(headers=None), b'{')
        bad = commands.Consumer(FakeService(fail=RuntimeError('x')))
        bad.on_message(conn, ch, SimpleNamespace(delivery_tag=3), SimpleNamespace(headers={}), body('c3'))
        self.assertEqual(ch.calls, [('ack', 1), ('reject', 2, False), ('nack', 3, True)])
        self.assertEqual(conn.later, [commands.BACKOFF[0]])


# ----- приём журнала ------------------------------------------------------------------------------
class Msg:
    def __init__(self, value, key=b'k'):
        self._v, self._k = value, key

    def value(self):
        return self._v

    def key(self):
        return self._k

    def topic(self):
        return 'tf.ingest.readings'

    def error(self):
        return None


class Reader:
    """Отдаёт очередь сообщений, потом тишину; после тишины останавливает приём."""

    def __init__(self, msgs, stop):
        self.msgs, self.stop, self.commits = list(msgs), stop, []
        self.consumer = self

    def poll(self, timeout):
        if self.msgs:
            return self.msgs.pop(0)
        self.stop.set()
        return None

    @staticmethod
    def _parse(payload):
        d = json.loads(payload)
        return RawEvent(int(d['channel_id']), d['ts'], d['value'])

    def commit(self):
        self.commits.append(True)


class Store:
    def __init__(self):
        self.rows, self.appends = [], 0

    def clean_event(self, channel_id, ts, value):
        return None if channel_id == 0 else {'channel_id': channel_id, 'ts': ts, 'value': value}

    def append(self, rows):
        self.rows += rows
        self.appends += 1


class IngestTest(unittest.TestCase):
    def test_consume_keeps_going_and_sends_garbage_to_dlq(self):
        stop, store, dead = threading.Event(), Store(), []
        ev = lambda ch: Msg(json.dumps({'channel_id': ch, 'ts': '2026-01-04T11:00:00', 'value': '1'}).encode())
        msgs = [ev(1), ev(2), Msg(b'{bad'), ev(0), ev(3)]
        stats = consume(store, Reader(msgs, stop), stop, dead=lambda *a: dead.append(a), batch_size=2)
        self.assertEqual([r['channel_id'] for r in store.rows], [1, 2, 3])
        self.assertEqual((stats['accepted'], stats['dropped'], stats['dead']), (3, 1, 1))
        self.assertEqual(dead[0][3], 'tf.ingest.readings')
        self.assertTrue(dead[0][2].startswith('JSONDecodeError'))


if __name__ == '__main__':
    unittest.main()
