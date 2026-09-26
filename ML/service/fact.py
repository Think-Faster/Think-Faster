"""Канал «по факту» (M8): происшествие уже идёт, а прогноз его не дал.

Разметку (guard/trig/inc/…) на такт строит один `retro.snapshot()` — общая для признаков и фактов.
Отсюда берутся эпизоды `inc` с «чистого» начала: правило `clean` из factalert — первый триггер
эпизода не помечен шумом (Н7 mass-сработка, Н8 «затоплен после питания»), поэтому `noise IS NULL`;
плюс Н10 — эпизоды в окне графика работ, которые по §1.6 не глушатся, а идут с пометкой. Задержка —
не больше цикла разметки (граница часа). Здесь же конец последнего эпизода пары (inc.t1) — для
since_hours карточки (П6): считается с конца эпизода, а не с тревоги.
"""
from datetime import datetime, timedelta

import svc as config


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
        ORDER BY i.object_id, i.type, i.t0""").fetchall()
    ta = t - timedelta(hours=1)
    return [{'object_id': int(o), 'collector_id': None if c is None else int(c), 'type': tp, 't0': t0,
             't1': t1, 'noise': nz, 'stype': st, 'new': t0 >= ta}
            for o, c, tp, t0, t1, nz, st in rows]


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