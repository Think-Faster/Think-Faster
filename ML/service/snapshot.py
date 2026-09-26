"""Онлайн-сборка признаков (M3): строка на каждый объект в момент прогноза.

Повторяет проверенный ретропрогон — `retro.snapshot()` из main — на кольцевом журнале сервиса
(INTEGRATION §1.2). Никакой своей копии: retro.snapshot сам пересоздаёт вид `ev` (окно 100 сут +
вся история охраны из `ev_all`), строит разметку labels.build и считает признаки; он же проверен
на отсутствие заглядывания в будущее. Сервис даёт ему ровно те таблицы, что он ждёт: obj, ch,
ev_all. Пересбор всего парка — секунды на границе часа (требование ≤60 с).

Рядом считается свежесть данных по типам (для флага «данные несвежие», INTEGRATION §7) — для
наблюдения M10, а не для остановки выдачи.
"""
from datetime import datetime, timedelta

import polars as pl

import svc as config


def snapshot(store, t: datetime, meta: dict, cal: dict, seq: bool = False, holes=()):
    """Строки признаков всех объектов на момент t (+ вход сети, если seq).

    Возврат — как в retro.snapshot(): (df, seq_pack) — один прогон labels.build на такт. `holes` —
    игнорируемые периоды главного диспетчера (settings.gaps): их события не идут в признаки.
    """
    import retro
    return retro.snapshot(store.con, t, meta, cal, seq=seq, holes=holes)


# семейства датчиков, на которые держится каждый тип (INTEGRATION §7: молчание семьи бьёт по типу).
# Основа слова в нижнем регистре ищется в stype без регистра: в справочнике «Датчик дыма»,
# «Состояние насоса», «Датчик движения» — с заглавной основы не находились, и флаг горел всегда.
STYPE_PAT = {'smoke': 'дым', 'heat': 'теплов', 'temp': 'температур', 'gas': 'газов',
             'flood': 'затоплен', 'pump': 'насос', 'phase': 'фаз', 'ups': 'ибп',
             'fan': 'вентилятор', 'door': 'двер', 'motion': 'движени', 'hatch': 'люк'}
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
        f"""SELECT stype, max(ts) AS last FROM ev_all WHERE ts < TIMESTAMP '{t}'
            AND ts >= TIMESTAMP '{cutoff}' GROUP BY stype""").fetchall()
    oldest = datetime.min
    lasts = {}
    for s, last in raws:
        for fam, pat in STYPE_PAT.items():
            if pat in s.lower():
                lasts[fam] = max(lasts.get(fam, oldest), last)
    hours = 24.0 * back_days
    out = {}
    for tp, fams in FAMILIES.items():
        ts = max((lasts.get(f, oldest) for f in fams), default=oldest)
        out[tp] = hours if ts == oldest else (t - ts).total_seconds() / 3600.0
    return out