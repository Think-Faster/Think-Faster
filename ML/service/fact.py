"""Канал «по факту» (M8): происшествие уже идёт, а прогноз его не дал.

Логика — как в факлаерте исследования, но на живом журнале: эпизоды и метки шума (Н7/Н8)
строятся ровно той же разметкой (labels.build), что и витрина. Разница только в момент объявления —
на границе часа, пока разметку достраиваем; Н7/Н8 видны из самого журнала за минуты, поэтому
задержка не больше цикла. Функция вызывается на общем с snapshot() коннекте (один labels.build
на такт): строить повторно дорого.
"""
from datetime import datetime, timedelta

import svc as config


def detect(store, t: datetime, back: str = '1 hour', need_build: bool = True) -> set:
    """Пары (object_id, тип) происшествий, которые уже идут к моменту t.

    Возвращаются только не-шумные эпизоды: mass-сработка (Н8) и «затоплен после питания» (Н7)
    в разметке помечены noise и сюда не попадают.
    """
    if need_build:
        store.ev_view(t)
        import labels
        labels.build(store.con)
    rows = store.con.sql(
        f"""SELECT object_id, type, t0 FROM inc
            WHERE noise IS NULL AND t0 >= TIMESTAMP '{(t - timedelta(hours=1)).isoformat()}'"""
    ).fetchall()
    return {(int(o), tp) for o, tp, _ in rows}