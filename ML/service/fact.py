"""Канал «по факту» (M8): происшествие уже идёт, а прогноз его не дал.

Разметку (guard/trig/inc/…) на такт строит один `retro.snapshot()` — общая для признаков и фактов.
Объявление — по правилам той же панели, что `factalert.py` (шаг 17): выбор правила по типу — ноль
ложных объявлений, при равенстве — большая полнота. У пожара, газа, подтопления, отказа датчика и
проникновения это «чистый» первый триггер: первые события эпизода не помечены шумом Н7/Н8
(`noise IS NULL`; Н8 — массовое срабатывание извещателей, Н7 — «Затоплен» сразу за событием питания,
оба видны из журнала в пределах десяти минут). У отказа оборудования фильтр шума, наоборот,
вычёркивает 423 настоящих эпизода ни за что — там объявление на первом триггере без фильтра
(правило `first`). Задержка — не больше цикла разметки (граница часа).
"""
from datetime import datetime, timedelta

import svc as config

# типы, где факт объявляется без фильтра Н7/Н8 — правило `first` из factalert (раздел 15 аналитики)
FIRST_BY_PASS = {'equipment'}


def detect(store, t: datetime, back: str = '1 hour', need_build: bool = True,
           noise: bool = True) -> set:
    """Пары (object_id, тип) происшествий, которые уже идут к моменту t.

    noise=True — чистое начало: только `noise IS NULL`, так снимается отклонение и молчание правил,
    снимать их шумным эпизодом нельзя. noise=False — правила объявления factalert: фильтр Н7/Н8 у всех
    типов, кроме оборудования (для него — правило `first`, без фильтра). Отбираются эпизоды,
    начавшиеся не раньше `back`.
    """
    if need_build:
        store.ev_view(t)
        import labels
        labels.build(store.con)
    from_ts = f"TIMESTAMP '{(t - timedelta(hours=1)).isoformat()}'"
    res = set()
    for tp in config.TYPES:
        filter7_8 = noise or tp not in FIRST_BY_PASS
        where = f"type = '{tp}' AND t0 >= {from_ts}"
        if filter7_8:
            where += ' AND noise IS NULL'
        rows = store.con.sql(f'SELECT object_id FROM inc WHERE {where}').fetchall()
        res.update((int(o), tp) for o, in rows)
    return res


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