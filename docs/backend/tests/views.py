# -*- coding: utf-8 -*-
"""Ручки стенда: прежнее API эмулятора плюс управление для тестов."""
import json

from django.http import (HttpResponse, JsonResponse, HttpResponseNotAllowed,
                         StreamingHttpResponse)
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from engine import FAULT_NAMES, MAX_SPEED, MIN_SPEED, get_engine

import time


def _json(payload, status=200):
    return JsonResponse(payload, status=status, safe=False,
                        json_dumps_params={'ensure_ascii': False, 'indent': 1})


def _body(request):
    try:
        return json.loads(request.body.decode('utf-8') or '{}')
    except ValueError:
        return {}


def _int(value):
    return int(value) if value not in (None, '') else None


# --- панель ---------------------------------------------------------------

def panel(request):
    engine = get_engine()
    return render(request, 'panel.html', {
        'исходные_сутки': engine.day.source_date,
        'активных_каналов': len(engine.day.channels),
        'всего_каналов': len(engine.channels),
        'скорость': engine.speed,
        'мин_скорость': MIN_SPEED,
        'макс_скорость': MAX_SPEED,
    })


# --- поток событий: те же ручки, что у эмулятора до стенда -----------------

def events(request):
    engine = get_engine()
    q = request.GET
    cursor = int(q.get('cursor', 0))
    limit = min(int(q.get('limit', 1000)), 10000)
    rows = engine.read(
        cursor, limit,
        channel_id=_int(q.get('channel_id')),
        object_id=_int(q.get('object_id')),
        system=q.get('system'),
        sensor_type=q.get('type'),
        alarm_only=q.get('alarm_only', '').lower() in ('true', '1'),
    )
    if q.get('format') == 'csv':
        # раскладка кавычек ровно как в журнале датасета:
        # идентификаторы и флаг без кавычек, остальное в кавычках
        lines = ['"ид_события","ид_канала_данных","дата","время",'
                 '"тревожное","значение_датчика"']
        for e in rows:
            lines.append('%d,%d,"%s","%s",%s,"%s"' % (
                e['ид_события'], e['ид_канала_данных'], e['дата'], e['время'],
                'true' if e['тревожное'] else 'false',
                str(e['значение_датчика']).replace('"', '""')))
        return HttpResponse('\n'.join(lines) + '\n',
                            content_type='text/csv; charset=utf-8')
    return _json({
        'курсор': rows[-1]['курсор'] if rows else cursor,
        'событий': len(rows),
        'события': rows,
    })


def stream(request):
    engine = get_engine()
    queue = engine.subscribe()

    def generate():
        try:
            while True:
                if queue:
                    e = queue.popleft()
                    yield 'id: %d\ndata: %s\n\n' % (
                        e['курсор'], json.dumps(e, ensure_ascii=False))
                else:
                    time.sleep(0.2)
                    yield ': ping\n\n'      # чтобы прокси не рвал тишину
        finally:
            engine.unsubscribe(queue)

    response = StreamingHttpResponse(generate(),
                                     content_type='text/event-stream; charset=utf-8')
    response['Cache-Control'] = 'no-cache'
    return response


def channels(request):
    engine = get_engine()
    rows = list(engine.channels.values())
    system = request.GET.get('system')
    if system:
        rows = [c for c in rows if c['тип_инж_системы'] == system]
    obj = request.GET.get('object_id')
    if obj:
        rows = [c for c in rows if c.get('ид_объект') == obj]
    if request.GET.get('active', '').lower() in ('true', '1'):
        rows = [c for c in rows if int(c['ид_канала_данных']) in engine.day.channels]
    return _json({'каналов': len(rows), 'каналы': rows})


def objects(request):
    engine = get_engine()
    return _json({'объектов': len(engine.objects), 'объекты': engine.objects})


def health(request):
    return _json(get_engine().health())


# --- управление стендом ---------------------------------------------------

def sensors(request):
    engine = get_engine()
    rows, total = engine.sensors(
        query=request.GET.get('q', ''),
        only_active=request.GET.get('active', '1').lower() not in ('0', 'false'),
        limit=min(int(request.GET.get('limit', 200)), 2000),
    )
    return _json({
        'найдено': total,
        'показано': len(rows),
        'датчики': rows,
        'часы': engine.health(),
    })


@csrf_exempt
def speed(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    try:
        value = float(_body(request).get('speed', 1))
    except (TypeError, ValueError):
        return _json({'ошибка': 'ускорение должно быть числом'}, 400)
    return _json({'ускорение': get_engine().set_speed(value)})


@csrf_exempt
def override(request):
    engine = get_engine()
    data = _body(request)
    channel = _int(data.get('channel_id'))
    if channel is None or channel not in engine.channels:
        return _json({'ошибка': 'нет такого канала'}, 400)
    if request.method == 'DELETE':
        engine.clear_override(channel)
        return _json({'канал': channel, 'вручную': None})
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST', 'DELETE'])
    error = engine.set_override(channel, data.get('value'))
    if error:
        return _json({'ошибка': error, 'пределы': engine.limits(channel)}, 400)
    return _json({'канал': channel, 'вручную': str(data.get('value'))})


@csrf_exempt
def fault(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    engine = get_engine()
    data = _body(request)
    channel = _int(data.get('channel_id'))
    if channel is None or channel not in engine.channels:
        return _json({'ошибка': 'нет такого канала'}, 400)
    kind = data.get('kind')
    error = engine.set_fault(channel, kind)
    if error:
        return _json({'ошибка': error, 'виды': list(FAULT_NAMES)}, 400)
    return _json({'канал': channel, 'сбой': engine.faults.get(channel)})


@csrf_exempt
def reset(request):
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    get_engine().reset()
    return _json({'состояние': 'ручные значения и сбои сняты'})
