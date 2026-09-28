"""Рекомендации по ТО и ремонту (ТЗ §4, §6, §8; раздел 61): словарь мер, стадия рецидива, поправки.

Модель говорит, где и что может случиться. Рекомендация говорит, что с этим делать. Она собирается
из блоков, каждый из которых записан в `settings/recommendations.csv` строкой с `rule_id`:

1. **Мера по признаку** (`вид = мера`). Признак — какой датчик и в каком состоянии держит эпизод:
   ИБП на батареях, насос «Неисправен», дым, дверь под охраной... У каждого признака два-три
   варианта: A — основной, B — другая гипотеза причины, C — минимальная мера, если на A нет людей.
   Признаков у эпизода может быть несколько (каждый — от 20% сработок): у главного все варианты,
   у остальных основной. По прогнозу без эпизода признаки берутся из всех оснований модели.
2. **Паттерн** (`вид = паттерн`): как вёл себя эпизод — разом на многих каналах, по очереди,
   дребезг, одиночный всплеск, держится полчаса и дольше; у прогноза — тревога стоит неделю.
   Паттерн разводит гипотезу причины: общая причина, каскад от первого канала, наладка канала,
   замена датчика. Интенсивность выводится числами (каналов, сработок, длительность).
3. **Сочетание** (`вид = сочетание`): на объекте в пределах часа есть другой тип (пожар с отказами
   датчиков, газ с отказом оборудования, вода с отказом насоса).
4. **Стадия рецидива** (`вид = повтор`). Сколько эпизодов того же типа было на объекте за 30 суток:
   R0 — ни одного, R1 — 1–2, R2 — 3–9 или повтор после выезда бригады, R3 — 10 и больше. Стадии
   взяты из данных: вероятность повтора в 7 суток растёт по ним у всех типов (`stats`). На R2 мера
   меняется: тот же датчик повторяет — менять датчик, разные датчики — искать общую причину.
5. **Поправки по обстановке** (`вид = поправка`): отказ разом на нескольких объектах коллектора,
   насос-«мигалка», весна, холод, персонал на объекте, перезапуск после питания, ППР
   газовых датчиков.

Вариант `A` у поправки, паттерна и сочетания — гипотеза, она встаёт первой; `+` — пометка к мерам.

У каждой меры есть `состав` — кого и сколько посылать: без выезда (диспетчер, план, разбор),
специалист (энергетик, связь и автоматика — один человек у щита или шкафа), звено из 3 (спуск в
коллектор: один внутри, наблюдающий и страхующий наверху — Правила 902н и 758н), бригада из 4
(газоопасные работы, откачка, подъём агрегата: двое внутри, двое наверху, не меньше трёх по правилам
газоопасных работ). Состав зависит от вида работ, а не от масштаба сработки: когда горят несколько
объектов коллектора, выезд короче, а не шире — это общая причина в одной точке (раздел 61).
Основные оперативные меры с выездом собираются в один выезд (`visit`): самый большой состав из
нужных, всё остальное делает тот же выезд.

Несколько типов на объекте разом (несколько тревог прогноза) собираются в составную рекомендацию
(`compose`): оперативные меры всех типов по сроку, у каждой — свой тип и правило.

Вывод отделён от вывода модели: `produced_by = rules`, версия — хеш словаря. Модель в правилах не
участвует, кроме оснований прогноза.

    python recommend.py stats                    # повторы по стадиям → settings/recurrence.csv
    python recommend.py run 5122 equipment "2026-03-01 10:00"
    python recommend.py run 5122 equipment "2026-03-01 10:00" pump_fault_168h,n_events_24h   # по прогнозу
    python recommend.py run 5122 equipment,flood "2026-03-10 10:00" equipment:pump_fault_168h,flood:pump_all_24h equipment=200
                                                 # составная: два типа, основания по типам, тревога стоит 200 ч
    python recommend.py retro                    # 2026: правила против того, что было на деле (на 10-й минуте)
    python recommend.py retro full               # то же задним числом по всему эпизоду
"""
import csv
import hashlib
import json
import sys
from datetime import datetime

import polars as pl

import config

RULES = config.ML / 'settings' / 'recommendations.csv'
RECUR = config.ML / 'settings' / 'recurrence.csv'
STAGES = ('R0', 'R1', 'R2', 'R3')
FLASH = 40          # переключений насоса за час — «мигалка» (раздел 61)
LINE_MIN = 30       # мин: окно «одновременно на другом объекте коллектора»
RESTART_MIN = 10    # мин: эпизод после подачи питания — перезапуск (раздел 45)
OPER_H = 4          # ч: по факту мера со сроком до 4 ч — оперативная, дальше — ТО и ремонт
HORIZON_H = 24      # ч: по прогнозу — в горизонт прогноза, дальше — ТО
DECIDE_MIN = 10     # мин: момент решения по факту (правило `silent` в factalert.py)
SUSTAIN_MIN = 5     # мин: сработки на 1–2 каналах идут дольше — устойчивый эпизод
NEVER = "TIMESTAMP '9999-01-01'"
SECOND = 0.2        # доля сработок эпизода, с которой датчик и состояние — ещё один признак эпизода
BURST_MIN = 2       # мин: каналы, сработавшие в первые 2 мин, — «разом»
CO_MIN = 60         # мин: другой тип на объекте в пределах часа — сочетание
STANDING_H = 168    # ч: тревога прогноза стоит неделю — стоящая (ISA-18.2, раздел 60)
# Состав выезда → человек (раздел 61): звено — 1 в коллекторе + наблюдающий + страхующий (902н, 758н),
# бригада — 2 в коллекторе + 2 наверху (газоопасные работы: не меньше 3, внутри не больше 2)
CREW = {'без выезда': 0, 'специалист': 1, 'звено': 3, 'бригада': 4}

# Датчик и состояние эпизода → признак словаря
TRIGGER = {
    ('Состояние насоса', 'Неисправен'): 'pump_fault',
    ('Состояние насоса', 'Обесточен'): 'pump_power',
    ('Состояние насоса', 'Отключено устройство'): 'pump_power',
    ('Состояние вентилятора', 'Неисправен'): 'fan_fault',
    ('Состояние вентилятора', 'Обесточен'): 'fan_power',
    ('Состояние вентилятора', 'Отключено устройство'): 'fan_power',
    ('ИБП', 'Питание от батарей'): 'ups_batt',
    ('Переключатель', 'Отключено устройство'): 'switch_off',
    ('Датчик дыма', 'Обнаружен дым'): 'smoke',
    ('Тепловой датчик', 'Не замкнут'): 'heat',
    ('Ручной извещатель', 'Не замкнут'): 'manual',
    ('Ручной извещатель', 'Рычаг сдернут'): 'manual',
    ('Состояние насоса', 'Работают все насосы в АНС'): 'pump_all',
    ('Состояние насоса', 'Затоплен'): 'pump_flooded',
    ('Датчик затопления', 'Не замкнут'): 'flood_sensor',
    ('Газовый датчик', 'Обнаружен газ'): 'gas',
}
BY_STYPE = {  # остальное — по типу датчика
    ('equipment', 'ИБП'): 'ups_fault', ('equipment', 'Состояние фазы'): 'phase',
    ('fire', 'Датчик температуры'): 'temp_hi',
    ('intrusion', 'КД Дверь'): 'door', ('intrusion', 'КД Люк'): 'hatch', ('intrusion', 'Стекло'): 'hatch',
    ('sensor', 'Датчик дыма'): 'smoke_fault', ('sensor', 'Датчик температуры'): 'temp_fault',
    ('sensor', 'Газовый датчик'): 'gas_fault', ('sensor', 'Ручной извещатель'): 'manual_fault',
    ('sensor', 'Тепловой датчик'): 'manual_fault', ('sensor', 'Датчик движения'): 'motion_fault',
    ('sensor', 'КД Дверь'): 'door_fault', ('sensor', 'КД Люк'): 'door_fault', ('sensor', 'КД АВ'): 'door_fault',
    ('sensor', '9-секционный люк'): 'door_fault', ('sensor', 'Стекло'): 'door_fault',
    ('sensor', 'Состояние УИР-Р'): 'uir_fault',
}
# Основание прогноза (признак витрины без окна) → признак словаря
REASON = {'smoke': 'smoke', 'heat': 'heat', 'manual': 'manual', 'temp_hi': 'temp_hi', 'gas': 'gas',
          'pump_fault': 'pump_fault', 'pump_all': 'pump_all', 'pump_flooded': 'pump_flooded',
          'flood_sensor': 'flood_sensor', 'fan_fault': 'fan_fault', 'phase_off': 'phase', 'phase_fault': 'phase',
          'ups_batt': 'ups_batt', 'ups_fault': 'ups_fault', 'smoke_fault': 'smoke_fault',
          'temp_fault': 'temp_fault', 'gas_fault': 'gas_fault', 'door': 'door', 'hatch': 'hatch'}


def trigger(tp: str, stype: str | None, what: str | None) -> str:
    if stype is None:
        return 'forecast'
    return TRIGGER.get((stype, what)) or BY_STYPE.get((tp, stype)) or ('other_fault' if tp == 'sensor' else 'forecast')


def stage(k30: int, after_visit: bool) -> str:
    if k30 >= 10:
        return 'R3'
    if k30 >= 3 or (k30 >= 1 and after_visit):
        return 'R2'
    return 'R1' if k30 >= 1 else 'R0'


def load_rules(path=RULES) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode('utf-8').splitlines(), delimiter=';'))
    for r in rows:
        assert len(r) == 15 and None not in r, f"{path.name}: {r['rule_id']} — не 15 полей"
        r['срок_ч'] = int(r['срок_ч']) if r['срок_ч'] else None
    return rows, hashlib.sha1(raw).hexdigest()[:10]


def load_recurrence(path=RECUR) -> dict:
    if not path.exists():
        return {}
    rows = csv.DictReader(path.read_text(encoding='utf-8').splitlines(), delimiter=';')
    return {(r['тип'], r['стадия']): r for r in rows}


# --- контекст эпизода ----------------------------------------------------------------------------

def context(con, where: str = 'true', upto: str = NEVER) -> pl.DataFrame:
    """Всё, что нужно правилам, по каждому чистому эпизоду `inc`, отобранному условием `where`
    (по колонкам `e.*`). История и соседи берутся по всем эпизодам, а не только по отобранным.

    `upto` — момент решения, SQL-выражение с `{a}` вместо псевдонима эпизода: по факту решение
    принимается, пока эпизод идёт, и видно только то, что пришло до этого момента (сработки,
    соседи по коллектору, другие типы на объекте). По умолчанию — весь эпизод, задним числом."""
    U = lambda a: upto.format(a=a)
    con.sql("""CREATE OR REPLACE TEMP TABLE e AS
        SELECT row_number() OVER (ORDER BY object_id, type, t0) id, * FROM inc WHERE noise IS NULL""")
    con.sql(f"""CREATE OR REPLACE TEMP TABLE dom AS
        SELECT id, channel_id, stype, what FROM (
          SELECT e.id, t.channel_id, t.stype, t.what, count(*) n, min(t.ts) f FROM e JOIN trig t
            ON t.object_id = e.object_id AND t.type = e.type AND t.ts BETWEEN e.t0 AND least(e.t1, {U('e')})
          GROUP BY ALL)
        QUALIFY row_number() OVER (PARTITION BY id ORDER BY n DESC, f) = 1""")
    con.sql("""CREATE OR REPLACE TEMP TABLE hist AS
        SELECT e.*, d.channel_id, d.stype, d.what,
          count(*) OVER (PARTITION BY e.object_id, e.type ORDER BY e.t0
                         RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING) k30,
          lag(e.t0) OVER w prev_t0, lag(e.confirmed) OVER w prev_confirmed, lag(d.channel_id) OVER w prev_channel,
          lead(e.t0) OVER w next_t0, lead(d.channel_id) OVER w next_channel
        FROM e LEFT JOIN dom d USING (id)
        WINDOW w AS (PARTITION BY e.object_id, e.type ORDER BY e.t0)""")
    # Паттерн эпизода: сколько каналов, как быстро подключались, дребезг, длительность (раздел 61)
    con.sql(f"""CREATE OR REPLACE TEMP TABLE chn AS
        SELECT e.id, t.channel_id, t.stype, t.what, date_trunc('minute', t.ts) m, count(*) n, min(t.ts) f, max(t.ts) l
        FROM e JOIN trig t ON t.object_id = e.object_id AND t.type = e.type AND t.ts BETWEEN e.t0 AND least(e.t1, {U('e')})
        GROUP BY ALL""")
    con.sql(f"""CREATE OR REPLACE TEMP TABLE pt AS
        WITH c AS (SELECT id, channel_id, sum(n) n, min(f) f, max(l) l, max(n) mx FROM chn GROUP BY ALL),
        w AS (SELECT id, stype, what, sum(n) n FROM chn GROUP BY ALL),
        wl AS (SELECT id, list(struct_pack(stype := stype, what := what) ORDER BY n DESC) trgs FROM (
                 SELECT *, n / sum(n) OVER (PARTITION BY id) sh FROM w) WHERE sh >= {SECOND} GROUP BY id)
        SELECT id, count(*)::INT nch, sum(n)::INT ntr, max(mx) mx, arg_min(channel_id, f) first_channel,
               date_diff('second', min(f), max(f)) / 60.0 spread_min, date_diff('second', min(f), max(l)) / 60.0 dur_min,
               count(*) FILTER (WHERE f <= (SELECT min(f) FROM c c2 WHERE c2.id = c.id) + INTERVAL {BURST_MIN} MINUTE) burst,
               any_value(wl.trgs) trgs
        FROM c LEFT JOIN wl USING (id) GROUP BY id""")
    con.sql(f"""CREATE OR REPLACE TEMP TABLE sel AS
        SELECT e.*, pt.* EXCLUDE (id),
          (SELECT list(DISTINCT b.type) FROM e b WHERE b.object_id = e.object_id AND b.type <> e.type
             AND b.t0 BETWEEN e.t0 - INTERVAL {CO_MIN} MINUTE AND least(e.t0 + INTERVAL {CO_MIN} MINUTE, {U('e')})) co_types
        FROM hist e LEFT JOIN pt USING (id) WHERE {where}""")
    power = """SELECT object_id, ts FROM ev WHERE state IN ('Есть питание', 'Питание от сети')"""
    return con.sql(f"""
        WITH gd AS (SELECT object_id, ts, armed FROM guard),
        pw AS ({power} AND object_id IN (SELECT object_id FROM sel)),
        ph AS (SELECT object_id, date_trunc('hour', ts) h, count(*) n FROM ev
               WHERE stype = 'Состояние насоса' AND state IN ('Включен', 'Выключен')
                 AND object_id IN (SELECT object_id FROM sel) GROUP BY ALL),
        lo AS (SELECT DISTINCT object_id, date_trunc('hour', ts) h FROM ev
               WHERE stype = 'Датчик температуры' AND state = 'Температура ниже 3ºC'
                 AND object_id IN (SELECT object_id FROM sel))
        SELECT s.*,
          (SELECT count(DISTINCT b.object_id) FROM e b WHERE b.collector_id = s.collector_id
             AND b.type = s.type AND b.object_id <> s.object_id
             AND b.t0 BETWEEN s.t0 - INTERVAL {LINE_MIN} MINUTE AND least(s.t0 + INTERVAL {LINE_MIN} MINUTE, {U('s')})) line_n,
          (SELECT max(n) FROM ph WHERE ph.object_id = s.object_id
             AND ph.h BETWEEN s.t0 - INTERVAL 24 HOUR AND s.t0) flash_n,
          (SELECT count(*) FROM lo WHERE lo.object_id = s.object_id
             AND lo.h BETWEEN s.t0 - INTERVAL 24 HOUR AND s.t0) > 0 AS cold,
          g.armed, date_diff('minute', g.ts, s.t0) AS guard_min,
          date_diff('minute', p.ts, s.t0) <= {RESTART_MIN} AS restart
        FROM sel s
        ASOF LEFT JOIN gd g ON g.object_id = s.object_id AND g.ts <= s.t0
        ASOF LEFT JOIN pw p ON p.object_id = s.object_id AND p.ts <= s.t0
        ORDER BY s.t0""").pl()


def flags(r: dict) -> dict:
    """Какие поправки и условия стадии выполняются для строки контекста."""
    tp, t0 = r['type'], r['t0']
    recent = r['prev_t0'] is not None and (t0 - r['prev_t0']).days < 30
    f = {
        'same_channel': recent and r['channel_id'] is not None and r['channel_id'] == r['prev_channel'],
        'other_channel': recent and r['channel_id'] is not None and r['prev_channel'] is not None
                         and r['channel_id'] != r['prev_channel'],
        'after_visit': recent and bool(r['prev_confirmed']),
        'line_wide': tp == 'equipment' and (r['line_n'] or 0) >= 1,
        'pump_flasher': tp == 'equipment' and (r['flash_n'] or 0) >= FLASH,
        'spring': tp == 'flood' and t0.month in (4, 5, 6),
        'cold': tp in ('fire', 'sensor', 'flood') and bool(r['cold']),
        'restart': tp in ('gas', 'equipment', 'flood') and bool(r['restart']),
        'disconnected': r['what'] == 'Отключено устройство',
    }
    off = r['armed'] is False
    workday = t0.weekday() < 5 and 8 <= t0.hour < 17
    f['planned_check'] = tp == 'gas' and off and workday
    f['guard_on'] = tp == 'gas' and r['armed'] is True
    f['staff_on_site'] = tp not in ('intrusion', 'gas') and off
    return f


# --- сборка рекомендации -------------------------------------------------------------------------

def _match(rule: dict, **kw) -> bool:
    return all(rule[k] in (v, '*') for k, v in kw.items())


def pattern(r: dict) -> str | None:
    """Как вёл себя эпизод (раздел 61, названия по ISA-18.2). По прогнозу без эпизода — только «стоящая»."""
    if (r.get('since_hours') or 0) >= STANDING_H:
        return 'standing'
    if r.get('nch') is None:
        return None
    if r['nch'] >= 3 and r['burst'] >= 3:
        return 'mass'           # 3+ канала за 2 мин — общая причина
    if r['nch'] >= 2 and r['spread_min'] >= 5:
        return 'spreading'      # каналы подключались по очереди
    if r['mx'] >= 3:
        return 'chatter'        # 3+ смены состояния канала за минуту
    if r['dur_min'] < 1 and r['nch'] == 1 and r['ntr'] <= 2:
        return 'fleeting'       # одиночная меньше минуты
    return 'sustained' if r['dur_min'] >= SUSTAIN_MIN else 'short'


def triggers(r: dict) -> list[str]:
    """Признаки эпизода по убыванию доли сработок (каждый от SECOND), по прогнозу — из оснований модели."""
    tp = r['type']
    if r.get('reason_triggers'):
        out = list(r['reason_triggers'])
    elif r.get('trgs'):
        out = [trigger(tp, x['stype'], x['what']) for x in r['trgs']]
    else:
        out = [trigger(tp, r.get('stype'), r.get('what'))]
    return list(dict.fromkeys(out))


KEEP = ('rule_id', 'вариант', 'режим', 'признак', 'мера', 'вид_работ', 'срок_ч', 'исполнитель', 'состав', 'обоснование', 'источник',
        'сверка_ртэк')


def item(x: dict, tp: str, mode: str) -> dict:
    """Строка словаря в выход: состав и число людей. Пока по факту горит газ, любой спуск на этот объект —
    газоопасная работа, и звено поднимается до бригады."""
    m = {k: x[k] for k in KEEP}
    if tp == 'gas' and mode == 'факт' and m['состав'] == 'звено':
        m['состав'] = 'бригада'
    m['людей'] = CREW.get(m['состав'])
    return m


def visit(now: list[dict]) -> dict | None:
    """Один выезд на основные оперативные меры (вариант A): самый большой состав, срок — самый ранний.
    Запасные варианты B и C несут свой состав в строке меры, но выезд не раздувают. `после_проверки` —
    раньше выезда стоит мера без выезда (камера, связь, перезапуск): выезд, только если она не сняла тревогу."""
    go = [m for m in now if (m.get('людей') or 0) > 0 and m['вариант'] == 'A']
    if not go:
        return None
    big = max(go, key=lambda m: m['людей'])
    first = min(m['срок_ч'] for m in go)
    return {'состав': big['состав'], 'людей': big['людей'], 'срок_ч': first,
            'исполнители': sorted({m['исполнитель'] for m in go}), 'меры': [m['rule_id'] for m in go],
            'после_проверки': any(m['людей'] == 0 and m['вариант'] == 'A' and m['срок_ч'] < first for m in now)}


def recommend(r: dict, rules: list[dict], version: str, recur: dict) -> dict:
    """Рекомендация по одному типу на объекте: меры признаков + паттерн + сочетание + рецидив + поправки.

    Режим «факт» — происшествие уже идёт (эпизод есть): реагирование — подтвердить, локализовать,
    устранить, а меры ТО — после. Режим «прогноз» — эпизода нет, есть тревога модели на 24 ч:
    профилактика в пределах горизонта, мер «немедленно» нет. У каждой строки словаря свой `режим`."""
    tp = r['type']
    mode = 'факт' if r.get('stype') is not None else 'прогноз'
    rules = [x for x in rules if x['режим'] in (mode, 'оба')]
    trigs = triggers(r)
    f = flags(r)
    st = stage(r['k30'], f['after_visit'])
    on = {k for k, v in f.items() if v}
    pat = pattern(r)
    co = set(r.get('co_types') or ())

    # Меры признаков: у главного все варианты, у остальных — только основной
    base = []
    for i, t in enumerate(trigs):
        m = [x for x in rules if x['вид'] == 'мера' and x['тип'] == tp and x['признак'] == t]
        base += sorted(m, key=lambda x: x['вариант']) if not base else [x for x in m if x['вариант'] == 'A']
    if not base:  # у признаков нет своих мер — прогнозная мера типа
        base = [x for x in rules if x['вид'] == 'мера' and x['тип'] == tp and x['признак'] == 'forecast']
    mods = [x for x in rules if x['вид'] == 'поправка' and x['признак'] in on and _match(x, тип=tp)]
    pats = [x for x in rules if x['вид'] == 'паттерн' and x['признак'] == pat and _match(x, тип=tp)]
    combo = [x for x in rules if x['вид'] == 'сочетание' and x['тип'] == tp and x['признак'] in co]
    rec = [x for x in rules if x['вид'] == 'повтор' and _match(x, тип=tp, стадия=st)
           and (x['признак'] == '*' or x['признак'] in on)]
    # правило своего типа вытесняет общее с тем же признаком и стадией (RC-03 вместо RC-02)
    own = {(x['признак'], x['стадия']) for x in rec if x['тип'] == tp}
    rec = [x for x in rec if x['тип'] == tp or (x['признак'], x['стадия']) not in own]

    # Порядок: гипотезы (поправка, паттерн, сочетание с вариантом A) → мера против рецидива → меры
    # признаков. Затем всё делится по сроку на оперативный блок (до OPER_H часов) и ТО, и внутри
    # оперативного — по сроку: мера «немедленно» всегда первой, какая бы гипотеза её ни обогнала.
    hyp = [x for x in mods + pats + combo if x['вариант'] == 'A']
    order = hyp + [x for x in rec if x['вариант'] == 'A'] + [x for x in rec if x['вариант'] not in ('A', '+')] + base
    notes = [x for x in mods + rec + pats + combo if x['вариант'] == '+' and x['мера'] and x['срок_ч'] is not None]
    blocks = {'оперативно': [], 'ТО': []}
    for x in order:
        blocks['оперативно' if x['срок_ч'] <= (OPER_H if mode == 'факт' else HORIZON_H) else 'ТО'].append(
            item(x, tp, mode))
    blocks['оперативно'].sort(key=lambda x: x['срок_ч'])
    ref = recur.get((tp, st), {})
    first = {b: (v[0] if v else None) for b, v in blocks.items()}
    num = lambda k: None if r.get(k) is None else round(float(r[k]), 1)
    return {
        'object_id': int(r['object_id']), 'type': tp, 'mode': mode, 'episode_t0': str(r['t0']), 'trigger': trigs[0],
        'triggers': trigs, 'channel_id': r.get('channel_id'), 'first_channel': r.get('first_channel'),
        'pattern': pat, 'intensity': {'channels': r.get('nch'), 'triggers': r.get('ntr'),
                                      'duration_min': num('dur_min'), 'since_hours': r.get('since_hours')},
        'with_types': sorted(co), 'stage': st, 'k30': int(r['k30']),
        'repeat_7d': float(ref['p7']) if ref else None, 'repeat_30d': float(ref['p30']) if ref else None,
        'context': sorted(on), 'hypothesis': [x['rule_id'] for x in hyp],
        'now': blocks['оперативно'], 'maintenance': blocks['ТО'], 'visit': visit(blocks['оперативно']),
        'notes': [item(x, tp, mode) for x in notes],
        'text': ' / '.join(v['мера'] for v in first.values() if v),
        'produced_by': 'rules', 'version': version,
    }


def compose(recs: list[dict]) -> dict:
    """Составная рекомендация по объекту из рекомендаций нескольких типов (несколько тревог разом):
    оперативные меры всех типов по сроку, ТО без повторов одного правила, у каждой меры — свой тип."""
    now, to, seen = [], [], set()
    for x in recs:
        for blk, out in (('now', now), ('maintenance', to)):
            for m in x[blk]:
                key = (m['rule_id'], m['мера'])
                if key not in seen:
                    seen.add(key)
                    out.append({'type': x['type'], **m})
    now.sort(key=lambda m: m['срок_ч'])
    return {'object_id': recs[0]['object_id'], 'types': [x['type'] for x in recs], 'now': now, 'maintenance': to,
            'visit': visit(now),
            'parts': recs, 'produced_by': 'rules', 'version': recs[0]['version']}


# --- режимы --------------------------------------------------------------------------------------

def bucket_sql(col='k30'):
    return f"CASE WHEN {col} = 0 THEN 'R0' WHEN {col} <= 2 THEN 'R1' WHEN {col} <= 9 THEN 'R2' ELSE 'R3' END"


def stats(con) -> None:
    """Повтор того же типа на объекте в 7 и 30 суток по стадии: подбор 2019–2024, тест 2026.
    R2 здесь — только по счёту 3–9; повтор после выезда отдельно в `retro`."""
    context(con, "e.t0 < TIMESTAMP '1900-01-01'")   # только таблица hist
    q = f"""SELECT type, {bucket_sql()} st, count(*) n,
              avg((next_t0 <= t0 + INTERVAL 7 DAY)::int) p7, avg((next_t0 <= t0 + INTERVAL 30 DAY)::int) p30,
              quantile_cont(date_diff('minute', t0, next_t0) / 1440.0, 0.5) med
            FROM hist WHERE year(t0) <> 2021 AND {{w}} GROUP BY ALL ORDER BY 1, 2"""
    fit = con.sql(q.format(w=f"t0 < TIMESTAMP '{config.TRAIN_END}'")).fetchall()
    test = {(a, b): (n, p7, p30) for a, b, n, p7, p30, _ in
            con.sql(q.format(w=f"t0 >= TIMESTAMP '{config.VAL_END}' AND t0 < TIMESTAMP '2026-06-01'")).fetchall()}
    with RECUR.open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, delimiter=';', lineterminator='\n')
        w.writerow(['тип', 'стадия', 'эпизодов', 'p7', 'p30', 'медиана_до_следующего_сут', 'период'])
        for tp, st, n, p7, p30, med in fit:
            w.writerow([tp, st, n, f'{p7:.3f}', f'{p30:.3f}', f'{med:.1f}' if med is not None else '', '2019–2024'])
    print('| тип | стадия | эпизодов 2019–2024 | повтор 7 сут | 30 сут | медиана до след., сут '
          '| эпизодов 2026 | повтор 7 сут 2026 | 30 сут 2026 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tp, st, n, p7, p30, med in fit:
        tn, t7, t30 = test.get((tp, st), (0, None, None))
        fmt = lambda x: '—' if x is None else f'{x:.0%}'
        print(f'| {tp} | {st} | {n} | {p7:.0%} | {p30:.0%} | {med:.1f} | {tn} | {fmt(t7)} | {fmt(t30)} |')


def reason_triggers(tp: str, reasons: list[str], rules: list[dict]) -> list[str]:
    """Основания прогноза, у которых в словаре есть меры этого типа, в порядке модели:
    `pump_fault_168h` → `pump_fault`. Основание вида `fire:smoke_24h` относится только к своему типу."""
    have = {x['признак'] for x in rules if x['вид'] == 'мера' and x['тип'] == tp}
    out = []
    for f in reasons:
        if ':' in f:
            t, f = f.split(':', 1)
            if t != tp:
                continue
        key = REASON.get(f.rsplit('_', 1)[0] if f.rsplit('_', 1)[-1].rstrip('h').isdigit() else f)
        if key in have and key not in out:
            out.append(key)
    return out


def ambient(con, obj: int, at) -> dict:
    """Обстановка на объекте за сутки до `at`, которая годится и для прогноза: мигалка насоса, холод."""
    flash = con.sql(f"""SELECT max(n) FROM (SELECT date_trunc('hour', ts) h, count(*) n FROM ev
                        WHERE object_id = {obj} AND stype = 'Состояние насоса' AND state IN ('Включен', 'Выключен')
                          AND ts BETWEEN TIMESTAMP '{at}' - INTERVAL 24 HOUR AND TIMESTAMP '{at}' GROUP BY 1)""").fetchone()[0]
    cold = con.sql(f"""SELECT count(*) FROM ev WHERE object_id = {obj} AND stype = 'Датчик температуры'
                       AND state = 'Температура ниже 3ºC'
                       AND ts BETWEEN TIMESTAMP '{at}' - INTERVAL 24 HOUR AND TIMESTAMP '{at}'""").fetchone()[0]
    return {'flash_n': flash or 0, 'cold': cold > 0}


def build(con, obj: int, types: list[str], at, reasons=(), since: dict | None = None,
          book: tuple | None = None, ctx: pl.DataFrame | None = None) -> dict:
    """Рекомендация по объекту на момент `at` по каждому типу из `types`: по последнему эпизоду за сутки,
    иначе по прогнозу (`reasons` — основания из `tf.forecast.results`, `since` — сколько часов тревога уже
    стоит, §2.3 INTEGRATION.md). Если типов несколько — составная рекомендация по объекту.
    `book` — (правила, версия, повторы), `ctx` — готовый `context()` на этот момент: сервис строит их
    один раз на такт для всех объектов с тревогой (`batch`)."""
    rules, ver, recur = book or (*load_rules(RULES), load_recurrence(RECUR))
    at = at if isinstance(at, datetime) else datetime.fromisoformat(at)
    since = since or {}
    if ctx is None:
        tl = ', '.join(f"'{t}'" for t in types)
        ctx = context(con, f"e.object_id = {obj} AND e.type IN ({tl}) AND e.t0 <= TIMESTAMP '{at}' "
                           f"AND e.t0 >= TIMESTAMP '{at}' - INTERVAL 24 HOUR", upto=f"TIMESTAMP '{at}'")
    out = []
    for tp in types:
        ep = ctx.filter((pl.col('object_id') == obj) & (pl.col('type') == tp))
        if len(ep):
            r = ep.row(-1, named=True)
            r['co_types'] = sorted(set(r['co_types'] or ()) | {t for t in types if t != tp})
        else:  # эпизода нет — прогнозная рекомендация с историей объекта
            k30 = con.sql(f"""SELECT count(*), max(t0) FROM hist WHERE object_id = {obj} AND type = '{tp}'
                              AND t0 < TIMESTAMP '{at}' AND t0 >= TIMESTAMP '{at}' - INTERVAL 30 DAY""").fetchone()
            last = con.sql(f"""SELECT confirmed, channel_id FROM hist WHERE object_id = {obj} AND type = '{tp}'
                               AND t0 < TIMESTAMP '{at}' ORDER BY t0 DESC LIMIT 1""").fetchone() or (None, None)
            r = {'object_id': obj, 'type': tp, 't0': at, 'stype': None, 'what': None, 'channel_id': None,
                 'k30': k30[0], 'prev_t0': k30[1], 'prev_confirmed': last[0], 'prev_channel': last[1],
                 'line_n': 0, 'armed': None, 'restart': False, **ambient(con, obj, at),
                 'reason_triggers': reason_triggers(tp, list(reasons), rules), 'since_hours': since.get(tp),
                 'co_types': [t for t in types if t != tp]}
        out.append(recommend(r, rules, ver, recur))
    return out[0] if len(out) == 1 else compose(out)


def batch(con, at: datetime, wanted: dict) -> dict:
    """Рекомендации такта (M13): `wanted` — объект → (типы с тревогой, основания, since по типу).
    Контекст эпизодов за сутки до `at` строится один раз на все объекты; ответ — объект → рекомендация
    (у одного типа — по типу, у нескольких — составная)."""
    if not wanted:
        return {}
    book = (*load_rules(RULES), load_recurrence(RECUR))   # сервис переназначает RULES/RECUR на свою папку
    objs = ', '.join(str(int(o)) for o in wanted)
    ctx = context(con, f"e.object_id IN ({objs}) AND e.t0 <= TIMESTAMP '{at}' "
                       f"AND e.t0 >= TIMESTAMP '{at}' - INTERVAL 24 HOUR", upto=f"TIMESTAMP '{at}'")
    return {o: build(con, int(o), types, at, reasons, since, book, ctx)
            for o, (types, reasons, since) in wanted.items()}


def run(con, obj: int, types: list[str], at: str, reasons: list[str] = (), since: dict | None = None) -> None:
    print(json.dumps(build(con, obj, types, at, reasons, since), ensure_ascii=False, indent=2, default=str))


def retro(con, full: bool = False) -> None:
    """Тест 2026: сколько раз сработало каждое правило и сходится ли рекомендация с тем, что было потом.
    Рекомендация по факту строится на момент решения — DECIDE_MIN минут от начала эпизода, как её увидел
    бы диспетчер; `full` — по всему эпизоду задним числом, для сравнения."""
    rules, ver = load_rules()
    recur = load_recurrence()
    upto = NEVER if full else f"{{a}}.t0 + INTERVAL {DECIDE_MIN} MINUTE"
    ctx = context(con, f"e.t0 >= TIMESTAMP '{config.VAL_END}' AND e.t0 < TIMESTAMP '2026-06-01'", upto)  # июнь на исход
    recs = [(r, recommend(r, rules, ver, recur)) for r in ctx.iter_rows(named=True)]
    print(f'эпизодов 2026 (янв–май, июнь — на исход): {len(recs)}, словарь {ver}, правил {len({x["rule_id"] for x in rules})} (строк с вариантами {len(rules)}), '
          f'момент решения: {"весь эпизод" if full else f"{DECIDE_MIN} мин от начала"}')

    # 1. что встаёт первым в каждом блоке
    first = lambda xs: xs[0]['rule_id'] if xs else '—'
    top = pl.DataFrame([{'type': r['type'], 'stage': x['stage'], 'now': first(x['now']),
                         'to': first(x['maintenance'])} for r, x in recs])
    print('\n**Первая мера в каждом блоке по типам** (правило: эпизодов)\n')
    print('| тип | эпизодов | R0 / R1 / R2 / R3 | оперативно | ТО и ремонт |')
    print('|---|---:|---|---|---|')
    for tp in config.TYPES:
        d = top.filter(pl.col('type') == tp)
        st = ' / '.join(str(d.filter(pl.col('stage') == s).height) for s in STAGES)
        cell = lambda c: ', '.join(f"{a}: {b}" for a, b in d.group_by(c).len().sort('len', descending=True).head(4).iter_rows())
        print(f"| {tp} | {d.height} | {st} | {cell('now')} | {cell('to')} |")

    # 2. поправки: сколько раз и что за ними на деле
    print('\n**Поправки и условия стадии** (эпизодов 2026 / доля с выездом бригады в 2 ч / повтор 7 сут)\n')
    print('| условие | эпизодов | выезд | повтор в 7 сут | без условия: выезд | повтор 7 сут |')
    print('|---|---:|---:|---:|---:|---:|')
    rep7 = lambda r: r['next_t0'] is not None and (r['next_t0'] - r['t0']).days < 7
    for k in ('same_channel', 'other_channel', 'after_visit', 'line_wide', 'pump_flasher', 'spring', 'cold',
              'restart', 'planned_check', 'guard_on', 'staff_on_site', 'disconnected'):
        a = [r for r, x in recs if k in x['context']]
        tps = {r['type'] for r in a}
        b = [r for r, x in recs if k not in x['context'] and r['type'] in tps]
        if not a:
            print(f'| {k} | 0 | | | | |')
            continue
        m = lambda xs, fn: f'{sum(map(fn, xs)) / len(xs):.0%}' if xs else '—'
        print(f"| {k} | {len(a)} | {m(a, lambda r: bool(r['confirmed']))} | {m(a, rep7)} "
              f"| {m(b, lambda r: bool(r['confirmed']))} | {m(b, rep7)} |")

    # 3. стадия против повтора на деле
    print('\n**Стадия против повтора на деле** (2026, с учётом «повтор после выезда» в R2)\n')
    print('| тип | ' + ' | '.join(f'{s}: эпизодов / повтор 7 сут' for s in STAGES) + ' |')
    print('|---|' + '---|' * len(STAGES))
    for tp in config.TYPES:
        cells = []
        for s in STAGES:
            a = [r for r, x in recs if r['type'] == tp and x['stage'] == s]
            cells.append(f"{len(a)} / {sum(map(rep7, a)) / len(a):.0%}" if a else '0')
        print(f'| {tp} | ' + ' | '.join(cells) + ' |')

    # 4. тот же датчик: после рекомендации «менять датчик» повторяет ли снова он же
    print('\n**Следующий эпизод с того же канала** (в 30 сут; признак, по которому R2 выбирает «менять датчик»)\n')
    print('| тип | тот же канал сейчас: эпизодов / следующий снова с него | другой канал: эпизодов / следующий с этого же |')
    print('|---|---|---|')
    nxt = lambda r: (r['next_t0'] is not None and (r['next_t0'] - r['t0']).days < 30
                     and r['next_channel'] == r['channel_id'])
    for tp in config.TYPES:
        a = [r for r, x in recs if r['type'] == tp and 'same_channel' in x['context']]
        b = [r for r, x in recs if r['type'] == tp and 'other_channel' in x['context']]
        fmt = lambda xs: f"{len(xs)} / {sum(map(nxt, xs)) / len(xs):.0%}" if xs else '0'
        print(f'| {tp} | {fmt(a)} | {fmt(b)} |')

    # 5. срочность первой оперативной меры против выезда на деле
    print('\n**Срок первой оперативной меры против выезда бригады** (след выезда в 2 ч, 2026)\n')
    print('| срок | эпизодов | выезд был |')
    print('|---|---:|---:|')
    lead = lambda x: x['now'][0]['срок_ч'] if x['now'] else None
    for lo, hi, name in ((0, 0, 'немедленно'), (1, 1, '1 ч'), (2, 4, '2–4 ч')):
        a = [r for r, x in recs if lead(x) is not None and lo <= lead(x) <= hi]
        if a:
            print(f"| {name} | {len(a)} | {sum(bool(r['confirmed']) for r in a) / len(a):.0%} |")
    a = [r for r, x in recs if lead(x) is None]
    if a:
        print(f"| оперативной меры нет, только ТО | {len(a)} | {sum(bool(r['confirmed']) for r in a) / len(a):.0%} |")

    # 5б. кого посылать: состав выезда рекомендации против выезда на деле
    print('\n**Состав выезда в рекомендации против выезда на деле** (след выезда в 2 ч, 2026)\n')
    print('| состав | эпизодов | доля | выезд был |')
    print('|---|---:|---:|---:|')
    crew = lambda x: x['visit']['состав'] if x['visit'] else 'без выезда'
    for c in CREW:
        a = [r for r, x in recs if crew(x) == c]
        if a:
            print(f"| {c} | {len(a)} | {len(a) / len(recs):.0%} | {sum(bool(r['confirmed']) for r in a) / len(a):.0%} |")

    # 6. паттерн эпизода: разводит ли он выезд и повтор того же канала
    pats = ('mass', 'spreading', 'chatter', 'fleeting', 'sustained', 'short')
    print('\n**Паттерн эпизода** (2026: эпизодов / выезд в 2 ч / следующий эпизод с того же канала в 30 сут)\n')
    print('| тип | ' + ' | '.join(pats) + ' |')
    print('|---|' + '---|' * len(pats))
    for tp in config.TYPES:
        cells = []
        for p in pats:
            a = [r for r, x in recs if r['type'] == tp and x['pattern'] == p]
            cells.append(f"{len(a)} / {sum(bool(r['confirmed']) for r in a) / len(a):.0%} / "
                         f"{sum(map(nxt, a)) / len(a):.0%}" if a else '0')
        print(f'| {tp} | ' + ' | '.join(cells) + ' |')

    # 7. сочетания типов на объекте в пределах часа
    print(f'\n**Сочетания типов** (другой тип на объекте от −{CO_MIN} мин до момента решения; эпизодов / выезд)\n')
    print('| тип | с каким | эпизодов | выезд | выезд без сочетания |')
    print('|---|---|---:|---:|---:|')
    for tp in config.TYPES:
        own = [(r, x) for r, x in recs if r['type'] == tp]
        for o in config.TYPES:
            a = [r for r, x in own if o in x['with_types']]
            b = [r for r, x in own if not x['with_types']]
            if len(a) >= 10 and b:
                print(f"| {tp} | {o} | {len(a)} | {sum(bool(r['confirmed']) for r in a) / len(a):.0%} "
                      f"| {sum(bool(r['confirmed']) for r in b) / len(b):.0%} |")

    # 8. сколько признаков и гипотез в одной рекомендации
    k = pl.DataFrame([{'type': r['type'], 'trig': len(x['triggers']), 'hyp': len(x['hypothesis']),
                       'now': len(x['now']), 'to': len(x['maintenance'])} for r, x in recs])
    print('\n**Состав рекомендации** (среднее на эпизод 2026)\n')
    print('| тип | признаков | гипотез | мер оперативно | мер ТО | эпизодов с 2+ признаками |')
    print('|---|---:|---:|---:|---:|---:|')
    for tp in config.TYPES:
        d = k.filter(pl.col('type') == tp)
        print(f"| {tp} | {d['trig'].mean():.2f} | {d['hyp'].mean():.2f} | {d['now'].mean():.1f} | {d['to'].mean():.1f} "
              f"| {(d['trig'] >= 2).mean():.0%} |")

    out = config.WORK / ('recommend_2026_full.jsonl' if full else 'recommend_2026.jsonl')
    with out.open('w', encoding='utf-8') as fh:
        for _, x in recs:
            fh.write(json.dumps(x, ensure_ascii=False, default=str) + '\n')
    print(f'\nвсе рекомендации: {out}')


if __name__ == '__main__':
    mode = sys.argv[1]
    con = config.connect(read_only=True)
    if mode == 'stats':
        stats(con)
    elif mode == 'run':
        # run <объект> <тип[,тип...]> <момент> [основания через запятую] [тип=часов тревоги,...]
        arg = sys.argv[2:] + [''] * 2
        since = {k: int(v) for k, v in (x.split('=') for x in arg[4].split(',') if x)}
        run(con, int(arg[0]), arg[1].split(','), arg[2], [x for x in arg[3].split(',') if x], since)
    else:
        retro(con, full='full' in sys.argv[2:])
