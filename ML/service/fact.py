"""Канал «по факту» (M8): происшествие уже идёт, а прогноз его не дал.

Разметку (guard/trig/inc/…) на такт строит один `retro.snapshot()` — общая для признаков и фактов.
Отсюда берутся эпизоды `inc` с «чистого» начала: правило `clean` из factalert — первый триггер
эпизода не помечен шумом (Н7 mass-сработка, Н8 «затоплен после питания»), поэтому `noise IS NULL`;
плюс Н10 — эпизоды в окне графика работ, которые по §1.6 не глушатся, а идут с пометкой. Задержка —
не больше цикла разметки (граница часа). Здесь же конец последнего эпизода пары (inc.t1) — для
since_hours карточки (П6): считается с конца эпизода, а не с тревоги.

Аварии и слепота §13.11 (pipeline/accidents.py) — тоже здесь: аномальная температура, слепота
объекта и маршрут нарушителя у проникновения. Эпизод пожара из одних датчиков температуры по факту
объявляется как `temperature`, а не как `fire`.
"""
import math
from datetime import datetime, timedelta

import svc as config
import accidents as acc


def detect(store, t: datetime, back: str = '1 hour', need_build: bool = True) -> set:
    """Пары (object_id, тип) чистых происшествий, начавшихся за час до t."""
    return {(e['object_id'], e['type']) for e in episodes(store, t, need_build)
            if e['noise'] is None and e['new']}


def episodes(store, t: datetime, need_build: bool = False) -> list[dict]:
    """Эпизоды, живые в последний час до t: чистые и Н10 — плановые работы по графику (§1.6).

    Н10 в цель модели не входит, но по факту не глушится: газ в окне ППР идёт диспетчеру с пометкой
    «идёт ППР по графику», а отказ снятого датчика — в историю со статусом MUTED (M7). Поэтому здесь,
    в отличие от признаков, Н10 не отбрасывается. `new` — эпизод начался в этот час; остальные живые —
    обновление объявления по факту (§2.3: поток обновлений, а не одно сообщение). `stype` — тип
    датчика первого срабатывания: по нему отказ снятого датчика отличается от прочих.
    """
    if need_build:
        store.ev_view(t)
        import labels
        labels.build(store.con)
    a = (t - timedelta(hours=1)).isoformat()
    rows = store.con.sql(f"""
        SELECT i.object_id, i.collector_id, i.type, i.t0, i.t1, i.noise,
               (SELECT any_value(g.stype) FROM trig g
                WHERE g.object_id = i.object_id AND g.type = i.type AND g.ts = i.t0) AS stype
        FROM inc i
        WHERE (i.noise IS NULL OR i.noise = 'Н10') AND i.t1 >= TIMESTAMP '{a}'
          AND (i.type <> 'fire' OR EXISTS (
               SELECT 1 FROM trig g WHERE g.object_id = i.object_id AND g.type = 'fire'
                 AND g.ts BETWEEN i.t0 AND i.t1 AND g.stype <> '{acc.TEMP_STYPE}'))
        ORDER BY i.object_id, i.type, i.t0""").fetchall()
    ta = t - timedelta(hours=1)
    out = [{'object_id': int(o), 'collector_id': None if c is None else int(c), 'type': tp, 't0': t0,
            't1': t1, 'noise': nz, 'stype': st, 'new': t0 >= ta}
           for o, c, tp, t0, t1, nz, st in rows]
    if any(e['type'] == 'intrusion' for e in out):
        route = acc.routes(store.con, f"""SELECT object_id, collector_id, t0, t1 FROM inc
            WHERE type = 'intrusion' AND (noise IS NULL OR noise = 'Н10') AND t1 >= TIMESTAMP '{a}'""")
        for e in out:
            if e['type'] == 'intrusion':
                e['route'] = [{'sensor_id': int(r['sensor_id']), 'stype': r['stype'], 'at': r['at']}
                              for r in route.get((e['object_id'], e['t0']), [])]
    return out


def accidents(store, t: datetime) -> list[dict]:
    """Эпизоды §13.11, живые в последний час до t и уже объявленные: `temperature` и `blind`.

    Разметка (obj3, guard) уже построена тем же тактом. Объявление — с момента подтверждения:
    у температуры — второй канал или второй отсчёт подряд, у слепоты — BLIND_CONFIRM. `new` —
    подтверждение пришлось на этот час. Слепота — по журналу за BLIND_LOOKBACK и по молчанию
    каналов в воронке (13.9); на объект — одно объявление, питание как причина старше связи.
    """
    ta = t - timedelta(hours=1)
    out = []
    acc.temperature(store.con)
    for o, c, d, t0, t1, ch, chans in store.con.sql(f"""
            SELECT object_id, collector_id, direction, t0, t1, confirm_h, channels FROM acc_temp
            WHERE confirm_h IS NOT NULL AND t1 >= TIMESTAMP '{ta}'
            ORDER BY object_id, t0""").fetchall():
        out.append({'object_id': int(o), 'collector_id': None if c is None else int(c),
                    'type': 'temperature', 't0': t0, 't1': t1, 'new': ch >= ta, 'direction': d,
                    'channels': [{'sensor_id': int(x['sensor_id']), 'value': x['value'],
                                  'baseline': x['baseline'], 'at': x['at']} for x in chans]})

    acc.blind(store.con, since=t - timedelta(days=acc.BLIND_LOOKBACK))
    confirm = timedelta(minutes=acc.BLIND_CONFIRM_MIN)
    found: dict[int, dict] = {}
    for o, c, cause, share, t0, t1 in store.con.sql(f"""
            SELECT object_id, collector_id, cause, share, t0, t1 FROM acc_blind
            WHERE (t1 IS NULL OR t1 >= TIMESTAMP '{ta}') AND coalesce(t1, TIMESTAMP '{t}') - t0 >= INTERVAL {acc.BLIND_CONFIRM}
            ORDER BY object_id, t0""").fetchall():
        _blind(found, int(o), c, cause, float(share), t0, t1)
    silent = store.silent_channels()
    if silent.height:
        total = dict(store.con.sql(f"""SELECT object_id, count(*) FROM ch
                                       WHERE stype <> '{acc.LINK_EXCLUDE}' GROUP BY 1""").fetchall())
        coll = dict(store.con.sql('SELECT object_id, collector_id FROM obj3').fetchall())
        for o, since in _silent_by_object(silent, acc.LINK_EXCLUDE).items():
            n = total.get(o, 0)
            k = max(1, math.ceil(acc.LINK_SHARE * n - 1e-9))
            if n and len(since) >= k:
                t0 = sorted(since)[k - 1]         # момент, когда замолчала нужная доля каналов
                if t - t0 >= confirm:
                    _blind(found, o, coll.get(o), 'link', round(len(since) / n, 2), t0, None)
    for o, b in found.items():
        out.append({'object_id': o, 'collector_id': b['collector_id'], 'type': 'blind', 't0': b['t0'],
                    't1': b['t1'] or t, 'new': b['t0'] + confirm >= ta, 'cause': b['cause'],
                    'share': b['share']})
    return out


def _blind(found: dict, o: int, c, cause: str, share: float, t0, t1) -> None:
    """Одно объявление слепоты на объект: раннее начало, питание как причина старше связи."""
    b = found.get(o)
    if b is None:
        found[o] = {'collector_id': None if c is None else int(c), 'cause': cause, 'share': share,
                    't0': t0, 't1': t1}
        return
    if cause == 'power':
        b['cause'] = 'power'
    b['share'] = max(b['share'], share)
    b['t0'] = min(b['t0'], t0)
    b['t1'] = None if b['t1'] is None or t1 is None else max(b['t1'], t1)


def _silent_by_object(silent, exclude: str) -> dict[int, list]:
    out: dict[int, list] = {}
    for o, s, since in silent.select(['object_id', 'stype', 'since']).iter_rows():
        if s != exclude and since is not None:
            out.setdefault(int(o), []).append(since)
    return out


def last_end(store, t: datetime, pairs: set[tuple[int, str]] | None = None) -> dict:
    """(object_id, тип) → часов с конца последнего эпизода этой пары (inc.t1, отрицательных нет).

    Служит источником since_hours для карточки при рестарте: склейка в rules.py — память процесса
    и после рестарта стартует заново, а разметка в горячей базе помнит, когда кончился эпизод.
    """
    if not pairs:
        return {}
    cond = ' OR '.join(f'(object_id = {o} AND type = \'{tp}\')' for o, tp in pairs)
    rows = store.con.sql(
        f"""SELECT object_id, type, max(t1) AS t1 FROM inc WHERE {cond} GROUP BY ALL"""
    ).fetchall()
    return {(int(o), tp): max(0.0, (t - t1).total_seconds() / 3600.0) for o, tp, t1 in rows}