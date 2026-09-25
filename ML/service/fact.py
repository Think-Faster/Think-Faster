"""Канал «по факту» (M8): происшествие уже идёт, а прогноз его не дал.

Разметку (guard/trig/inc/…) на такт строит один `retro.snapshot()` — общая для признаков и фактов.
Отсюда берётся только эпизоды `inc` с «чистого» начала: правило `clean` из factalert — первый
триггер эпизода не помечен шумом (Н7 mass-сработка, Н8 «затоплен после питания»), поэтому
`noise IS NULL`. Задержка — не больше цикла разметки (граница часа). Здесь же конец последнего
эпизода пары (inc.t1) — для since_hours карточки (П6): считается с конца эпизода, а не с тревоги.
"""
from datetime import datetime, timedelta

import svc as config


def detect(store, t: datetime, back: str = '1 hour', need_build: bool = True) -> set:
    """Пары (object_id, тип) чистых происшествий, которые уже идут к моменту t."""
    if need_build:
        store.ev_view(t)
        import labels
        labels.build(store.con)
    rows = store.con.sql(
        f"""SELECT object_id, type, t0 FROM inc
            WHERE noise IS NULL AND t0 >= TIMESTAMP '{(t - timedelta(hours=1)).isoformat()}'"""
    ).fetchall()
    return {(int(o), tp) for o, tp, _ in rows}


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