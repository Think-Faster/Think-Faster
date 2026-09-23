"""Онлайн-сборка признаков (M3): строка на каждый объект в момент прогноза.

Повторяет проверенный ретропрогон — `retro.snapshot()` и `retro.seq_pack()` — на кольцевом
журнале сервиса (INTEGRATION §1.2: ровно это делает retro.snapshot(); он же проверен на отсутствие
заглядывания в будущее). Пересбор всего парка — секунды на границе часа (требование ≤60 с).

Рядом считаются две вещи из INTEGRATION §7: свежесть данных по типам (для флага «данные несвежие»,
а не для остановки выдачи) и молчание семейств датчиков (записывается в сообщение тревоги).
"""
from datetime import datetime, timedelta

import polars as pl

import svc as config


def snapshot(store, t: datetime, meta: dict, cal: dict) -> pl.DataFrame:
    """Строки признаков всех объектов на момент t из журнала сервиса."""
    store.ev_view(t)
    import retro
    return retro.snapshot(store.con, t, meta, cal)


def seq(store, t: datetime, meta: dict) -> dict | None:
    """Вход сети (одно 168-часовое окно на объект) или None, если сетей в выгрузке нет."""
    if not meta.get('seq', {}).get('present'):
        return None
    store.ev_view(t)
    import retro
    return retro.seq_pack(store.con, t, meta)


# семейства датчиков, на которые держится каждый тип (INTEGRATION §7: молчание семьи бьёт по типу)
STYPE_PAT = {'smoke': 'Дым', 'heat': 'Теплово', 'temp': 'Температур', 'gas': 'Газов',
             'flood': 'Затоплен', 'pump': 'Насос', 'phase': 'Фаз', 'ups': 'ИБП',
             'fan': 'Вентилятор', 'door': 'Дверь', 'motion': 'Движение', 'hatch': 'Люк'}
FAMILIES = {'fire': ('smoke', 'heat', 'temp'), 'gas': ('gas',),
            'flood': ('flood', 'pump'), 'equipment': ('phase', 'ups', 'fan', 'pump'),
            'sensor': ('smoke', 'temp'), 'intrusion': ('door', 'motion', 'hatch')}


def freshness(store, t: datetime, back_days: int = 30) -> dict:
    """Сколько часов назад по каждому типу был последний признак его семейств.

    Мера грубая (наблюдение M10): последнее событие семейства по подстроке stype, без учёта
    окна признаков. Точный счёт не нужен — флажок несвежести использует только порядок величины.
    """
    cutoff = t - timedelta(days=back_days)
    raws = store.con.sql(
        f"""SELECT stype, max(ts) AS last FROM readings WHERE ts < TIMESTAMP '{t}'
            AND ts >= TIMESTAMP '{cutoff}' GROUP BY stype""").fetchall()
    oldest = datetime.min
    lasts = {}
    for s, last in raws:
        for fam, pat in STYPE_PAT.items():
            if pat in s:
                lasts[fam] = max(lasts.get(fam, oldest), last)
    hours = 24.0 * back_days
    out = {}
    for tp, fams in FAMILIES.items():
        ts = max((lasts.get(f, oldest) for f in fams), default=oldest)
        out[tp] = hours if ts == oldest else (t - ts).total_seconds() / 3600.0
    return out