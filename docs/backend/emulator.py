# -*- coding: utf-8 -*-
"""Эмулятор потока СМВУ.

Проигрывает данные 2026 года из датасета так, будто датчики работают сейчас:
событие, случившееся в исходных сутках в 03:09:27, выдаётся сегодня в 03:09:27.
Формат события в точности повторяет журнал датасета, поэтому потребитель не
отличает эмулятор от настоящей выгрузки.

Живой интеграции с СМВУ нам не дадут (ответ заказчика на разборе задачи),
поэтому источник эмулируется — но не случайными числами, а реальными сутками
из журнала: сохраняются состав активных каналов, частота событий по каналам,
набор значений, доля тревожных и суточный профиль нагрузки.

Запуск:
    python emulator.py --data ../dataset --port 8080

Ускорение времени для тестов (сутки за 2,4 минуты):
    python emulator.py --data ../dataset --speed 600

Ручки:
    GET /events?cursor=0&limit=1000    события после курсора (JSON или CSV)
        &channel_id=  &object_id=  &system=  &type=  &alarm_only=true
        &format=csv                                        как в датасете
    GET /stream                        те же события потоком (SSE)
    GET /channels?object_id=&system=&active=true   справочник каналов
    GET /objects                       справочник объектов
    GET /health                        состояние: часы эмулятора, счётчики
"""
import argparse
import csv
import io
import json
import os
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

JOURNAL = 'журнал_событий_пример.csv'
CHANNELS = 'справочник_каналов_датчиков.csv'
OBJECTS = 'справочник_объектов_диспетчер.csv'

NUMERIC = re.compile(r'^-?\d+(\.\d+)?$')


def read_csv(path):
    with io.open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class Day(object):
    """Сутки из журнала: события по времени плюс разброс значений по каналам.

    Разброс нужен, чтобы вторые и последующие сутки не были посимвольной копией
    первых: числовые значения берутся в наблюдавшемся по каналу диапазоне,
    время немного смещается, порядок пересобирается.
    """

    def __init__(self, journal_path, jitter, rnd):
        self.jitter = jitter
        self.rnd = rnd
        self.events = []       # (секунда от полуночи, канал, тревожное, значение)
        self.numeric = {}      # канал -> (минимум, максимум, знаков после точки)
        self.max_event_id = 0
        self.source_date = None

        for row in read_csv(journal_path):
            h, m, s = (int(x) for x in row['время'].split(':'))
            channel = int(row['ид_канала_данных'])
            value = row['значение_датчика']
            # в примере флаг записан как true/false, в годовых журналах — t/f
            alarm = row['тревожное'].strip().lower() in ('true', 't', '1')
            self.events.append((h * 3600 + m * 60 + s, channel, alarm, value))
            self.max_event_id = max(self.max_event_id, int(row['ид_события']))
            if self.source_date is None:
                self.source_date = row['дата']
            if NUMERIC.match(value):
                lo, hi, prec = self.numeric.get(channel, (None, None, 0))
                v = float(value)
                prec = max(prec, len(value.split('.')[1]) if '.' in value else 0)
                self.numeric[channel] = (v if lo is None else min(lo, v),
                                         v if hi is None else max(hi, v), prec)

        self.events.sort(key=lambda e: e[0])
        self.channels = set(e[1] for e in self.events)

    def render(self, first_day):
        """События суток: первые — как в журнале, последующие — с вариацией."""
        if first_day:
            return list(self.events)
        out = []
        for sec, channel, alarm, value in self.events:
            sec = min(86399, max(0, sec + self.rnd.randint(-self.jitter, self.jitter)))
            if channel in self.numeric:
                lo, hi, prec = self.numeric[channel]
                if hi > lo:
                    value = ('%%.%df' % prec) % self.rnd.uniform(lo, hi)
            out.append((sec, channel, alarm, value))
        out.sort(key=lambda e: e[0])
        return out


class Stream(object):
    """Часы эмулятора, выдача событий и буфер для чтения по курсору."""

    def __init__(self, day, channels, objects, speed, buffer_size):
        self.day = day
        self.channels = channels
        # связь канала с объектом появилась в справочнике 17.09.2026
        self.object_name = {r['ид_объект']: r['диспетчерское_название_объекта']
                            for r in objects}
        self.speed = speed
        self.buffer = deque(maxlen=buffer_size)
        self.cursor = 0                       # сквозной номер выдачи, он же курсор
        self.event_id = day.max_event_id      # нумерацию продолжаем с конца журнала
        self.lock = threading.Lock()
        self.subscribers = []                 # очереди SSE-подписчиков
        self.started_at = time.time()
        self.emitted = 0
        self.alarms = 0
        self.origin_real = time.time()
        self.origin_model = datetime.now()
        self.now = self.origin_model

    def model_now(self):
        """Модельное время: при speed=1 совпадает с настоящим, иначе идёт быстрее."""
        return self.origin_model + timedelta(
            seconds=(time.time() - self.origin_real) * self.speed)

    def _emit(self, moment, channel, alarm, value):
        meta = self.channels.get(channel, {})
        obj = meta.get('ид_объект') or None
        event = {
            'курсор': 0,
            'ид_события': 0,
            'ид_канала_данных': channel,
            'дата': moment.strftime('%Y-%m-%d'),
            'время': moment.strftime('%H:%M:%S'),
            'тревожное': alarm,
            'значение_датчика': value,
            'тип_инж_системы': meta.get('тип_инж_системы'),
            'тип_датчика': meta.get('тип_датчика'),
            'название_датчика': meta.get('название_датчика'),
            'ид_объект': int(obj) if obj else None,
            'название_объекта': self.object_name.get(obj),
        }
        with self.lock:
            self.cursor += 1
            self.event_id += 1
            event['курсор'] = self.cursor
            event['ид_события'] = self.event_id
            self.buffer.append(event)
            self.emitted += 1
            if alarm:
                self.alarms += 1
            for q in list(self.subscribers):
                q.append(event)

    def run(self):
        self.origin_real = time.time()
        self.origin_model = datetime.now()
        day_start = self.origin_model.replace(hour=0, minute=0, second=0, microsecond=0)
        first_day = True
        while True:
            for sec, channel, alarm, value in self.day.render(first_day):
                target = day_start + timedelta(seconds=sec)
                # то, что по часам уже прошло до запуска, задним числом не выдаём
                if first_day and target < self.origin_model:
                    continue
                while True:
                    wait = (target - self.model_now()).total_seconds() / self.speed
                    if wait <= 0:
                        break
                    time.sleep(min(wait, 1.0))
                self.now = target
                self._emit(target, channel, alarm, value)
            first_day = False
            day_start = day_start + timedelta(days=1)

    def read(self, cursor, limit, channel_id=None, object_id=None, system=None,
             sensor_type=None, alarm_only=False):
        with self.lock:
            rows = [e for e in self.buffer if e['курсор'] > cursor]
        if channel_id is not None:
            rows = [e for e in rows if e['ид_канала_данных'] == channel_id]
        if object_id is not None:
            rows = [e for e in rows if e['ид_объект'] == object_id]
        if system:
            rows = [e for e in rows if e['тип_инж_системы'] == system]
        if sensor_type:
            rows = [e for e in rows if e['тип_датчика'] == sensor_type]
        if alarm_only:
            rows = [e for e in rows if e['тревожное']]
        return rows[:limit]


def make_handler(stream, channels, objects):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, fmt, *args):      # не засорять консоль каждым запросом
            pass

        def _send(self, code, body, content_type='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else body.encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)

        def _json(self, payload, code=200):
            self._send(code, json.dumps(payload, ensure_ascii=False, indent=1))

        def _health(self):
            uptime = time.time() - stream.started_at
            return {
                'состояние': 'работает',
                'время_эмулятора': stream.now.strftime('%Y-%m-%d %H:%M:%S'),
                'исходные_сутки': stream.day.source_date,
                'ускорение': stream.speed,
                'выдано_событий': stream.emitted,
                'из_них_тревожных': stream.alarms,
                'событий_в_секунду': round(stream.emitted / uptime, 2) if uptime else 0,
                'курсор': stream.cursor,
                'в_буфере': len(stream.buffer),
                'подписчиков_на_поток': len(stream.subscribers),
            }

        def _events(self, q):
            one = lambda k, d=None: q.get(k, [d])[0]
            cursor = int(one('cursor', 0))
            limit = min(int(one('limit', 1000)), 10000)
            rows = stream.read(
                cursor, limit,
                channel_id=int(one('channel_id')) if one('channel_id') else None,
                object_id=int(one('object_id')) if one('object_id') else None,
                system=one('system'),
                sensor_type=one('type'),
                alarm_only=one('alarm_only', '').lower() in ('true', '1'),
            )
            if one('format') == 'csv':
                # раскладка кавычек ровно как в журнале датасета:
                # идентификаторы и флаг без кавычек, остальное в кавычках
                lines = ['"ид_события","ид_канала_данных","дата","время",'
                         '"тревожное","значение_датчика"']
                for e in rows:
                    lines.append('%d,%d,"%s","%s",%s,"%s"' % (
                        e['ид_события'], e['ид_канала_данных'], e['дата'], e['время'],
                        'true' if e['тревожное'] else 'false',
                        e['значение_датчика'].replace('"', '""')))
                self._send(200, '\n'.join(lines) + '\n', 'text/csv; charset=utf-8')
            else:
                self._json({
                    'курсор': rows[-1]['курсор'] if rows else cursor,
                    'событий': len(rows),
                    'события': rows,
                })

        def _stream(self):
            queue = deque(maxlen=10000)
            stream.subscribers.append(queue)
            self.close_connection = True        # SSE идёт до разрыва соединения
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    if queue:
                        e = queue.popleft()
                        chunk = 'id: %d\ndata: %s\n\n' % (
                            e['курсор'], json.dumps(e, ensure_ascii=False))
                    else:
                        time.sleep(0.2)
                        chunk = ': ping\n\n'    # чтобы прокси не рвал тишину
                    self.wfile.write(chunk.encode('utf-8'))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                if queue in stream.subscribers:
                    stream.subscribers.remove(queue)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == '/health':
                self._json(self._health())
            elif url.path == '/events':
                self._events(q)
            elif url.path == '/stream':
                self._stream()
            elif url.path == '/channels':
                rows = list(channels.values())
                system = q.get('system', [None])[0]
                if system:
                    rows = [c for c in rows if c['тип_инж_системы'] == system]
                obj = q.get('object_id', [None])[0]
                if obj:
                    rows = [c for c in rows if c.get('ид_объект') == obj]
                if q.get('active', [''])[0].lower() in ('true', '1'):
                    rows = [c for c in rows
                            if int(c['ид_канала_данных']) in stream.day.channels]
                self._json({'каналов': len(rows), 'каналы': rows})
            elif url.path == '/objects':
                self._json({'объектов': len(objects), 'объекты': objects})
            else:
                self._json({'ошибка': 'нет такой ручки',
                            'ручки': ['/events', '/stream', '/channels',
                                      '/objects', '/health']}, 404)

    return Handler


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description='Эмулятор потока СМВУ на данных 2026 года')
    p.add_argument('--data', default=os.path.join(here, '..', 'dataset'),
                   help='папка с журналом и справочниками')
    p.add_argument('--journal', default=None,
                   help='журнал, по умолчанию %s из папки --data' % JOURNAL)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--speed', type=float, default=1.0,
                   help='ускорение времени: 1 — реальное, 600 — сутки за 2,4 минуты')
    p.add_argument('--jitter', type=int, default=90,
                   help='разброс времени событий в последующих сутках, секунд')
    p.add_argument('--buffer', type=int, default=50000,
                   help='сколько последних событий держать для чтения по курсору')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args()

    data = os.path.abspath(args.data)
    journal = args.journal or os.path.join(data, JOURNAL)
    rnd = random.Random(args.seed)

    channels = {int(r['ид_канала_данных']): r
                for r in read_csv(os.path.join(data, CHANNELS))}
    objects = read_csv(os.path.join(data, OBJECTS))
    day = Day(journal, args.jitter, rnd)

    stream = Stream(day, channels, objects, args.speed, args.buffer)
    threading.Thread(target=stream.run, daemon=True).start()

    print('исходные сутки %s: %d событий, %d активных каналов'
          % (day.source_date, len(day.events), len(day.channels)))
    print('справочники: %d каналов, %d объектов' % (len(channels), len(objects)))
    print('ускорение %g, слушаю http://%s:%d' % (args.speed, args.host, args.port))
    ThreadingHTTPServer((args.host, args.port),
                        make_handler(stream, channels, objects)).serve_forever()


if __name__ == '__main__':
    main()
