# -*- coding: utf-8 -*-
"""Движок эмулятора СМВУ: часы, поток событий, ручное управление датчиками.

Проигрывает сутки из журнала датасета по текущим часам: событие, случившееся
в исходных сутках в 03:09:27, выдаётся сегодня в 03:09:27. Формат события
повторяет журнал, поэтому потребитель не отличает стенд от настоящей выгрузки.

Поверх воспроизведения — управление для тестов: ускорение времени, ручное
значение датчика в пределах наблюдавшихся крайних значений и три сбоя
(залипание, отключение, замыкание). Всё состояние живёт в памяти процесса.
"""
import csv
import io
import os
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timedelta

JOURNAL = 'журнал_событий_пример.csv'
CHANNELS = 'справочник_каналов_датчиков.csv'
OBJECTS = 'справочник_объектов_диспетчер.csv'

NUMERIC = re.compile(r'^-?\d+(\.\d+)?$')

STUCK, OFFLINE, SHORT, OK = 'stuck', 'offline', 'short', 'ok'
FAULT_NAMES = {STUCK: 'залипание', OFFLINE: 'отключение', SHORT: 'замыкание'}
BROKEN_STATE = 'Неисправен'          # чем отвечает текстовый канал при замыкании

MIN_SPEED, MAX_SPEED = 1.0, 3600.0


def read_csv(path):
    with io.open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


class Day(object):
    """Сутки из журнала плюс наблюдавшийся разброс значений.

    Разброс служит двум целям: вторые и последующие сутки не повторяют первые
    посимвольно, и он же задаёт крайние значения, дальше которых руками
    выкрутить датчик нельзя.
    """

    def __init__(self, journal_path, channels, jitter, rnd):
        self.jitter = jitter
        self.rnd = rnd
        self.events = []        # (секунда от полуночи, канал, тревожное, значение)
        self.counts = {}        # канал -> сколько раз отметился за сутки
        self.numeric = {}       # канал -> (минимум, максимум, знаков после точки)
        self.max_event_id = 0
        self.source_date = None
        type_numeric = {}       # тип датчика -> (минимум, максимум, знаков)
        type_states = {}        # тип датчика -> {состояние: сколько раз}

        for row in read_csv(journal_path):
            h, m, s = (int(x) for x in row['время'].split(':'))
            channel = int(row['ид_канала_данных'])
            value = row['значение_датчика']
            # в примере флаг записан как true/false, в годовых журналах — t/f
            alarm = row['тревожное'].strip().lower() in ('true', 't', '1')
            self.events.append((h * 3600 + m * 60 + s, channel, alarm, value))
            self.counts[channel] = self.counts.get(channel, 0) + 1
            self.max_event_id = max(self.max_event_id, int(row['ид_события']))
            if self.source_date is None:
                self.source_date = row['дата']

            sensor_type = channels.get(channel, {}).get('тип_датчика', '')
            if NUMERIC.match(value):
                prec = len(value.split('.')[1]) if '.' in value else 0
                v = float(value)
                self.numeric[channel] = _widen(self.numeric.get(channel), v, prec)
                type_numeric[sensor_type] = _widen(type_numeric.get(sensor_type), v, prec)
            else:
                states = type_states.setdefault(sensor_type, {})
                states[value] = states.get(value, 0) + 1

        self.events.sort(key=lambda e: e[0])
        self.channels = set(e[1] for e in self.events)
        self.type_numeric = type_numeric
        # состояния по убыванию частоты: сначала то, что датчик показывает обычно
        self.type_states = {t: [s for s, _ in sorted(c.items(), key=lambda x: -x[1])]
                            for t, c in type_states.items()}

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


def _widen(current, value, prec):
    if current is None:
        return (value, value, prec)
    lo, hi, old_prec = current
    return (min(lo, value), max(hi, value), max(old_prec, prec))


class Engine(object):
    """Часы стенда, выдача событий и ручное управление датчиками."""

    def __init__(self, data_dir, speed=1.0, jitter=90, buffer_size=50000,
                 fault_interval=10, seed=None):
        rnd = random.Random(seed)
        self.data_dir = data_dir
        self.channels = {int(r['ид_канала_данных']): r
                         for r in read_csv(os.path.join(data_dir, CHANNELS))}
        self.objects = read_csv(os.path.join(data_dir, OBJECTS))
        self.object_name = {r['ид_объект']: r['диспетчерское_название_объекта']
                            for r in self.objects}
        self.day = Day(os.path.join(data_dir, JOURNAL), self.channels, jitter, rnd)

        self.speed = float(speed)
        self.fault_interval = fault_interval
        self.buffer = deque(maxlen=buffer_size)
        self.cursor = 0
        self.event_id = self.day.max_event_id
        self.lock = threading.RLock()
        self.subscribers = []
        self.started_at = time.time()
        self.emitted = 0
        self.alarms = 0
        self.origin_real = time.time()
        self.origin_model = datetime.now()
        self.now = self.origin_model

        self.overrides = {}      # канал -> значение, выставленное руками
        self.faults = {}         # канал -> вид сбоя
        self.last_value = {}     # канал -> последнее выданное значение
        self.last_emit = {}      # канал -> модельное время последнего события
        self._started = False

    # --- часы -------------------------------------------------------------

    def model_now(self):
        """Модельное время: при скорости 1 совпадает с настоящим, иначе идёт быстрее."""
        return self.origin_model + timedelta(
            seconds=(time.time() - self.origin_real) * self.speed)

    def set_speed(self, speed):
        """Сменить ускорение на ходу, не дёргая модельные часы назад или вперёд."""
        speed = max(MIN_SPEED, min(MAX_SPEED, float(speed)))
        with self.lock:
            self.origin_model = self.model_now()   # переносим точку отсчёта
            self.origin_real = time.time()         # и только потом меняем множитель
            self.speed = speed
        return speed

    # --- пределы и значения ----------------------------------------------

    def limits(self, channel):
        """Крайние значения датчика — из данных, а не придуманные.

        Числовой канал: минимум и максимум этого канала за сутки примера, а если
        канал показывал одно значение или молчал — пределы его типа датчика.
        Текстовый канал: список состояний, которые показывал его тип.
        """
        sensor_type = self.channels.get(channel, {}).get('тип_датчика', '')
        own = self.day.numeric.get(channel)
        if own and own[1] > own[0]:
            lo, hi, prec = own
            return {'вид': 'число', 'мин': lo, 'макс': hi, 'точность': prec,
                    'шаг': round(10.0 ** -prec, prec) if prec else 1}
        by_type = self.day.type_numeric.get(sensor_type)
        states = self.day.type_states.get(sensor_type)
        if by_type and by_type[1] > by_type[0] and not states:
            lo, hi, prec = by_type
            return {'вид': 'число', 'мин': lo, 'макс': hi, 'точность': prec,
                    'шаг': round(10.0 ** -prec, prec) if prec else 1}
        if states:
            return {'вид': 'состояние', 'состояния': states}
        if by_type:
            lo, hi, prec = by_type
            return {'вид': 'число', 'мин': lo, 'макс': hi, 'точность': prec,
                    'шаг': round(10.0 ** -prec, prec) if prec else 1}
        return {'вид': 'нет'}     # тип датчика ни разу не встретился в сутках

    def check_value(self, channel, value):
        """Проверить, что ручное значение не выходит за крайние. Вернуть ошибку или None."""
        lim = self.limits(channel)
        if lim['вид'] == 'число':
            if not NUMERIC.match(str(value).strip()):
                return 'значение должно быть числом'
            v = float(value)
            if v < lim['мин'] or v > lim['макс']:
                return ('значение вне пределов датчика: допустимо от %s до %s'
                        % (lim['мин'], lim['макс']))
            return None
        if lim['вид'] == 'состояние':
            if value not in lim['состояния']:
                return 'состояние не встречалось у этого типа датчика'
            return None
        return 'у этого датчика нет наблюдавшихся значений, задать вручную нечего'

    def short_value(self, channel):
        """Что показывает датчик при замыкании: заведомо вне рабочего диапазона."""
        lim = self.limits(channel)
        if lim['вид'] == 'число':
            hi, prec = lim['макс'], lim['точность']
            return ('%%.%df' % prec) % max(hi * 5, hi + 1)
        return BROKEN_STATE

    # --- выдача событий ---------------------------------------------------

    def _emit(self, moment, channel, alarm, value):
        """Выдать событие с учётом ручного значения и сбоя. Вернуть False, если канал молчит."""
        with self.lock:
            fault = self.faults.get(channel)
            if fault == OFFLINE:
                return False
            if fault == STUCK:
                value = self.last_value.get(channel, value)
            elif fault == SHORT:
                value, alarm = self.short_value(channel), True
            elif channel in self.overrides:
                value = self.overrides[channel]

            meta = self.channels.get(channel, {})
            obj = meta.get('ид_объект') or None
            self.cursor += 1
            self.event_id += 1
            event = {
                'курсор': self.cursor,
                'ид_события': self.event_id,
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
                'сбой': fault if fault in FAULT_NAMES else None,
            }
            self.buffer.append(event)
            self.last_value[channel] = value
            self.last_emit[channel] = moment
            self.emitted += 1
            if alarm:
                self.alarms += 1
            for q in list(self.subscribers):
                q.append(event)
        return True

    def _play(self):
        """Основной цикл: выдавать события суток синхронно модельным часам."""
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
                    time.sleep(min(wait, 1.0))   # шаг в секунду, чтобы ловить смену скорости
                self.now = target
                self._emit(target, channel, alarm, value)
            first_day = False
            day_start = day_start + timedelta(days=1)

    def _keep_alive(self):
        """Подпитка для каналов под управлением.

        Залипший или закороченный датчик надо чем-то показывать, а по расписанию
        редкий канал шлёт раз в часы. Поэтому такие каналы плюс те, у которых
        выставлено ручное значение, повторяют своё состояние раз в fault_interval
        модельных секунд.
        """
        while True:
            time.sleep(0.5)
            now = self.model_now()
            with self.lock:
                managed = set(self.overrides) | set(
                    c for c, k in self.faults.items() if k in (STUCK, SHORT))
            for channel in managed:
                last = self.last_emit.get(channel)
                if last is None or (now - last).total_seconds() >= self.fault_interval:
                    self.now = now
                    self._emit(now, channel, False, self.last_value.get(channel, ''))

    def start(self):
        if self._started:
            return self
        self._started = True
        self.origin_real = time.time()
        self.origin_model = datetime.now()
        threading.Thread(target=self._play, daemon=True).start()
        threading.Thread(target=self._keep_alive, daemon=True).start()
        return self

    # --- управление -------------------------------------------------------

    def set_override(self, channel, value):
        """Выставить датчику значение вручную и сразу его показать."""
        error = self.check_value(channel, value)
        if error:
            return error
        with self.lock:
            self.overrides[channel] = str(value)
        self._emit(self.model_now(), channel, False, str(value))
        return None

    def clear_override(self, channel):
        with self.lock:
            self.overrides.pop(channel, None)

    def set_fault(self, channel, kind):
        """Навесить сбой на датчик или снять его."""
        if kind not in (STUCK, OFFLINE, SHORT, OK):
            return 'неизвестный вид сбоя'
        with self.lock:
            if kind == OK:
                self.faults.pop(channel, None)
                return None
            self.faults[channel] = kind
        if kind in (STUCK, SHORT):
            # сразу показать, что стало с датчиком
            self._emit(self.model_now(), channel, False, self.last_value.get(channel, ''))
        return None

    def reset(self):
        with self.lock:
            self.overrides.clear()
            self.faults.clear()

    # --- чтение -----------------------------------------------------------

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

    def sensors(self, query='', only_active=True, limit=200):
        """Строки для таблички датчиков в панели."""
        query = query.strip().lower()
        out = []
        for channel, meta in self.channels.items():
            if only_active and channel not in self.day.channels:
                continue
            obj = meta.get('ид_объект') or None
            if query:
                haystack = ' '.join([
                    str(channel), meta.get('название_датчика', ''),
                    meta.get('тип_датчика', ''), meta.get('тип_инж_системы', ''),
                    self.object_name.get(obj, '')]).lower()
                if query not in haystack:
                    continue
            out.append({
                'ид_канала_данных': channel,
                'название_датчика': meta.get('название_датчика'),
                'тип_датчика': meta.get('тип_датчика'),
                'тип_инж_системы': meta.get('тип_инж_системы'),
                'ид_объект': int(obj) if obj else None,
                'название_объекта': self.object_name.get(obj),
                'активен': channel in self.day.channels,
                'событий_в_сутки': self.day.counts.get(channel, 0),
                'значение': self.last_value.get(channel),
                'пределы': self.limits(channel),
                'вручную': self.overrides.get(channel),
                'сбой': self.faults.get(channel),
            })
        # порядок должен быть устойчивым: если подмешать сюда «есть значение»,
        # строки будут всплывать по ходу суток и ползунок уедет из-под курсора.
        # частота за сутки постоянна, и вверху оказываются самые говорливые датчики
        out.sort(key=lambda r: (r['сбой'] is None and r['вручную'] is None,
                                -r['событий_в_сутки'],
                                r['название_датчика'] or '￿',
                                r['ид_канала_данных']))
        return out[:limit], len(out)

    def health(self):
        uptime = time.time() - self.started_at
        with self.lock:
            return {
                'состояние': 'работает',
                'время_эмулятора': self.now.strftime('%Y-%m-%d %H:%M:%S'),
                'исходные_сутки': self.day.source_date,
                'ускорение': self.speed,
                'выдано_событий': self.emitted,
                'из_них_тревожных': self.alarms,
                'событий_в_секунду': round(self.emitted / uptime, 2) if uptime else 0,
                'курсор': self.cursor,
                'в_буфере': len(self.buffer),
                'подписчиков_на_поток': len(self.subscribers),
                'ручных_значений': len(self.overrides),
                'сбоев': {FAULT_NAMES[k]: sum(1 for v in self.faults.values() if v == k)
                          for k in (STUCK, OFFLINE, SHORT)},
            }

    def subscribe(self):
        queue = deque(maxlen=10000)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)


_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Один движок на процесс, поднимается при первом обращении."""
    global _engine
    with _engine_lock:
        if _engine is None:
            here = os.path.dirname(os.path.abspath(__file__))
            data = os.environ.get('TF_DATA') or os.path.join(here, '..', '..', 'dataset')
            _engine = Engine(
                os.path.abspath(data),
                speed=float(os.environ.get('TF_SPEED', 1)),
                jitter=int(os.environ.get('TF_JITTER', 90)),
                fault_interval=int(os.environ.get('TF_FAULT_INTERVAL', 10)),
                seed=int(os.environ['TF_SEED']) if os.environ.get('TF_SEED') else None,
            ).start()
        return _engine
