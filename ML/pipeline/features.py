"""Шаг 3. Витрина признаков объект × час без заглядывания в будущее.

Момент прогноза — конец часа h. Признаки видят только часы ≤ h, цель — начало эпизода в часах
h+1 … h+горизонт. Вместо метки храним «через сколько часов следующий эпизод» по каждому типу:
горизонт тогда настраивается без пересборки витрины (ТЗ §6, Ф5-3).

Часть разметки смотрит немного вперёд: шум Н8 — на 10 мин, персонал при проникновении — на 15 мин,
след выезда — на 30 мин. Такие события попадают в признаки по моменту, когда они становятся известны
(KNOWN), а не по моменту начала.

Что лежит в витрине:
- счётчики состояний по ключам KEYS в окнах 1 ч / 6 ч / 24 ч / 7 сут;
- газ и температура: максимум, среднее, тренд; сколько часов нет показаний;
- эпизоды и триггеры по типам, шум Н7 и Н8 — на объекте и во всём коллекторе; часы с последнего эпизода;
- выезды бригады в коллекторе, вход персонала под охраной, режим охраны;
- календарь: час, день недели, сезон, праздники и 9 мая (ответ 23);
- состав объекта: число каналов по типам датчиков.

Выход: work/features/<год>.parquet, work/features/meta.json и work/seq.npz — часовые ряды для нейросети.

    python features.py
"""
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config

T0 = datetime(2019, 1, 1)
NH = int((config.DATA_END - T0).total_seconds() // 3600)
CAP = 2160       # ч: «давно» — 90 суток; дальше счётчик не растёт
NEXT_CAP = 721   # ч: до следующего эпизода «больше месяца»
FUTURE = 72      # ч: строка попадает в витрину, только если за ней 72 ч данных — горизонт до 72 ч
# Все дыры в данных: потери и 2021 год
HOLES = config.GAPS + [(datetime(2021, 1, 1), datetime(2022, 1, 1))]

KEYS = [
    ('smoke', "stype = 'Датчик дыма' AND state = 'Обнаружен дым'"),
    ('smoke_clear', "stype = 'Датчик дыма' AND state = 'Дыма нет'"),
    ('smoke_fault', "stype = 'Датчик дыма' AND state = 'Неисправен'"),
    ('heat', "stype = 'Тепловой датчик' AND state = 'Не замкнут'"),
    ('manual', "stype = 'Ручной извещатель' AND state IN ('Не замкнут', 'Рычаг сдернут')"),
    ('temp_hi', "stype = 'Датчик температуры' AND state = 'Температура выше 40ºC'"),
    ('temp_lo', "stype = 'Датчик температуры' AND state = 'Температура ниже 3ºC'"),
    ('temp_fault', "stype = 'Датчик температуры' AND state IN ('Неисправен', 'Не определено')"),
    ('gas', "stype = 'Газовый датчик' AND state = 'Обнаружен газ'"),
    ('gas_fault', "stype = 'Газовый датчик' AND state = 'Неисправен'"),
    ('pump_on', "stype = 'Состояние насоса' AND state = 'Включен'"),
    ('pump_off', "stype = 'Состояние насоса' AND state = 'Выключен'"),
    ('pump_flooded', "stype = 'Состояние насоса' AND state = 'Затоплен'"),
    ('pump_all', "stype = 'Состояние насоса' AND state = 'Работают все насосы в АНС'"),
    ('pump_fault', "stype = 'Состояние насоса' AND state IN ('Неисправен', 'Обесточен')"),
    ('fan_on', "stype = 'Состояние вентилятора' AND state = 'Включен'"),
    ('fan_off', "stype = 'Состояние вентилятора' AND state = 'Выключен'"),
    ('fan_fault', "stype = 'Состояние вентилятора' AND state IN ('Неисправен', 'Обесточен')"),
    ('phase_off', "stype = 'Состояние фазы' AND state = 'Обесточен'"),
    ('phase_on', "stype = 'Состояние фазы' AND state = 'Есть питание'"),
    ('phase_fault', "stype = 'Состояние фазы' AND state = 'Неисправен'"),
    ('ups_batt', "stype = 'ИБП' AND state = 'Питание от батарей'"),
    ('ups_mains', "stype = 'ИБП' AND state = 'Питание от сети'"),
    ('ups_fault', "stype = 'ИБП' AND state IN ('Неисправен', 'Батарея неисправна', 'Батарея разряжена')"),
    ('switch', "stype = 'Переключатель' AND state IN ('Включен', 'Выключен')"),
    ('door', "stype = 'КД Дверь' AND state = 'Не замкнут'"),
    ('hatch', "stype IN ('КД Люк', '9-секционный люк', 'Стекло') AND state = 'Не замкнут'"),
    ('av', "stype = 'КД АВ' AND state = 'Не замкнут'"),
    ('motion', "state = 'Обнаружено движение'"),
    ('uir_call', "stype = 'Состояние УИР-Р' AND state IN ('Вызов', 'Разговор')"),
    ('uir_lever', "stype = 'Состояние УИР-Р' AND state = 'Рычаг сдернут'"),
    ('arm', "stype = 'Состояние охраны' AND state = 'На охране'"),
    ('disarm', "stype = 'Состояние охраны' AND state = 'Снято с охраны'"),
    ('guard_faulty', "stype = 'Состояние охраны' AND state = 'Много неисправных устройств'"),
    ('flood_sensor', "stype = 'Датчик затопления' AND state = 'Не замкнут'"),
    ('fault_other', "state = 'Неисправен'"),
    ('disconnected', "state = 'Отключено устройство'"),
    ('undefined', "state IN ('Неопределен', 'Не определено')"),
    ('off', "state = 'Выключен'"),
    ('norm', "state = 'Норма'"),
]
NUMERIC = ['gas_max', 'gas_sum', 'gas_n', 'temp_sum', 'temp_n', 'temp_max', 'temp_min', 'n_events', 'n_channels',
           'guard_last']
ONSETS = [f'onset_{t}' for t in config.TYPES] + ['noise_fire', 'noise_flood']
TRIGS = [f'trig_{t}' for t in config.TYPES]
OTHER = ['arrival', 'visit']
BASE = [k for k, _ in KEYS] + NUMERIC + ONSETS + TRIGS + OTHER
IDX = {k: i for i, k in enumerate(BASE)}
NAN_INIT = ['gas_max', 'temp_max', 'temp_min', 'guard_last']
# Через сколько после начала событие становится известно (см. labels.py)
KNOWN = {'fire': '10 minutes', 'intrusion': '15 minutes', 'noise_fire': '10 minutes',
         'arrival': '15 minutes', 'visit': '30 minutes'}

# Нерабочие праздничные дни РФ (без переносов) и длинные каникулы
HOLIDAYS = [(1, d) for d in range(1, 9)] + [(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)]
LONG = [(1, 1, 1, 10), (5, 1, 5, 11)]


def hour_of(col: str, known: str | None = None) -> str:
    ts = f"{col} + INTERVAL '{known}'" if known else col
    return f"date_diff('hour', TIMESTAMP '{T0}', {ts})"


def dense(x) -> np.ndarray:
    """Столбец из fetchnumpy: NULL → NaN."""
    if np.ma.isMaskedArray(x):
        return x.astype(np.float64).filled(np.nan)
    return np.asarray(x)


def load_base(con, lo: int = 0, hi: int = NH) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """Часовые ряды объектов за часы lo … hi-1 (по умолчанию — все данные)."""
    objects = [r[0] for r in con.sql('SELECT object_id FROM obj3 ORDER BY object_id').fetchall()]
    oi = {o: i for i, o in enumerate(objects)}
    base = np.zeros((len(objects), hi - lo, len(BASE)), np.float32)
    for c in NAN_INIT:
        base[:, :, IDX[c]] = np.nan

    def put(sql: str, columns: list[str]) -> None:
        d = con.sql(f"""SELECT * FROM ({sql}) WHERE h >= {lo} AND h < {hi}
                        AND object_id IN (SELECT object_id FROM obj3)""").fetchnumpy()
        o = np.array([oi[x] for x in d['object_id']], dtype=np.int64)
        h = dense(d['h']).astype(np.int64) - lo
        for c in columns:
            base[o, h, IDX[c]] = dense(d[c])

    case = 'CASE ' + ' '.join(f"WHEN {cond} THEN '{k}'" for k, cond in KEYS) + ' END'
    d = con.sql(f"""SELECT object_id, h, k, count(*) AS n FROM (
                        SELECT object_id, {hour_of('ts')} AS h, {case} AS k FROM ev WHERE state IS NOT NULL)
                    WHERE k IS NOT NULL AND h >= {lo} AND h < {hi} AND object_id IN (SELECT object_id FROM obj3)
                    GROUP BY ALL""").fetchnumpy()
    base[np.array([oi[x] for x in d['object_id']]), dense(d['h']).astype(np.int64) - lo,
         np.array([IDX[k] for k in d['k']])] = dense(d['n'])
    print('  счётчики состояний', flush=True)
    gas = "stype = 'Газовый датчик' AND num >= 0 AND num < 327.68"
    temp = "stype = 'Датчик температуры' AND num BETWEEN -40 AND 80"
    put(f"""SELECT object_id, {hour_of('ts')} AS h, count(*) AS n_events, count(DISTINCT channel_id) AS n_channels,
                   max(num) FILTER (WHERE {gas}) AS gas_max, coalesce(sum(num) FILTER (WHERE {gas}), 0) AS gas_sum,
                   count(num) FILTER (WHERE {gas}) AS gas_n,
                   coalesce(sum(num) FILTER (WHERE {temp}), 0) AS temp_sum, count(num) FILTER (WHERE {temp}) AS temp_n,
                   max(num) FILTER (WHERE {temp}) AS temp_max, min(num) FILTER (WHERE {temp}) AS temp_min
            FROM ev GROUP BY ALL""", NUMERIC[:-1])
    print('  показания', flush=True)
    put(f"""SELECT object_id, {hour_of('ts')} AS h, arg_max(armed::INT, ts) AS guard_last FROM guard GROUP BY ALL""",
        ['guard_last'])
    for t in config.TYPES:
        put(f"""SELECT object_id, {hour_of('t0', KNOWN.get(t))} AS h, count(*) AS onset_{t} FROM inc
                WHERE type = '{t}' AND noise IS NULL GROUP BY ALL""", [f'onset_{t}'])
        put(f"""SELECT object_id, {hour_of('ts', KNOWN.get(t))} AS h, count(*) AS trig_{t} FROM trig
                WHERE type = '{t}' GROUP BY ALL""", [f'trig_{t}'])
    put(f"""SELECT object_id, {hour_of('t0', KNOWN['noise_fire'])} AS h, count(*) AS noise_fire FROM inc
            WHERE noise = 'Н8' GROUP BY ALL""", ['noise_fire'])
    put(f"""SELECT object_id, {hour_of('t0')} AS h, count(*) AS noise_flood FROM inc WHERE noise = 'Н7' GROUP BY ALL""",
        ['noise_flood'])
    put(f"""SELECT object_id, {hour_of('ts', KNOWN['arrival'])} AS h, count(*) AS arrival FROM armed_open
            WHERE arrival GROUP BY ALL""", ['arrival'])
    put(f"""SELECT o.object_id, {hour_of('v.t0', KNOWN['visit'])} AS h, count(*) AS visit
            FROM visit v JOIN obj3 o USING (collector_id) GROUP BY ALL""", ['visit'])
    return base, objects, oi


def window_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Сумма по часам h-w+1 … h вдоль оси 0. Накопление в float64: у газа миллиарды показаний."""
    cs = np.cumsum(x, axis=0, dtype=np.float64)
    out = cs.copy()
    out[w:] -= cs[:-w]
    return out.astype(np.float32)


def window_ext(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Скользящий максимум или минимум по часам h-w+1 … h, NaN — показаний не было."""
    fill = -np.inf if fn is np.max else np.inf
    padded = np.concatenate([np.full(w - 1, fill, np.float32), np.where(np.isnan(x), fill, x)])
    out = fn(np.lib.stride_tricks.sliding_window_view(padded, w), axis=-1)
    return np.where(np.isinf(out), np.nan, out).astype(np.float32)


def since_last(x: np.ndarray) -> np.ndarray:
    """Сколько часов назад было последнее ненулевое значение (0 — в этом часе), не больше CAP."""
    idx = np.arange(len(x))
    last = np.maximum.accumulate(np.where(x > 0, idx, -CAP - 1))
    return np.minimum(idx - last, CAP).astype(np.float32)


def ffill(x: np.ndarray) -> np.ndarray:
    idx = np.maximum.accumulate(np.where(np.isnan(x), 0, np.arange(len(x))))
    return x[idx]


def hours_to_next(x: np.ndarray) -> np.ndarray:
    """Через сколько часов после часа h следующее ненулевое значение (≥ 1), не больше NEXT_CAP."""
    idx = np.arange(len(x))
    first = np.minimum.accumulate(np.where(x > 0, idx, 10 ** 9)[::-1])[::-1]
    after = np.append(first[1:], 10 ** 9)
    return np.minimum(after - idx, NEXT_CAP).astype(np.int16)


def calendar() -> dict[str, np.ndarray]:
    hours = np.arange(NH)
    days = [T0 + timedelta(days=d) for d in range(NH // 24 + 1)]
    holiday = np.array([(d.month, d.day) in HOLIDAYS for d in days])
    long_ = np.array([any((m1, d1) <= (d.month, d.day) <= (m2, d2) for m1, d1, m2, d2 in LONG) for d in days])
    to_holiday = np.zeros(len(days), np.float32)
    gap = 60
    for i in range(len(days) - 1, -1, -1):
        gap = 0 if holiday[i] else min(gap + 1, 60)
        to_holiday[i] = gap
    day = hours // 24
    doy = np.array([d.timetuple().tm_yday for d in days])[day]
    return {
        'hour': (hours % 24).astype(np.float32),
        'dow': np.array([d.weekday() for d in days], np.float32)[day],
        'month': np.array([d.month for d in days], np.float32)[day],
        'doy_sin': np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
        'doy_cos': np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
        'holiday': holiday[day].astype(np.float32),
        'long_holiday': long_[day].astype(np.float32),
        'may9': np.array([(d.month, d.day) == (5, 9) for d in days], np.float32)[day],
        'days_to_holiday': to_holiday[day],
    }


def object_features(b: np.ndarray, coll: np.ndarray) -> dict[str, np.ndarray]:
    """b — часовые ряды объекта (часы × BASE), coll — сумма ONSETS + TRIGS по коллектору."""
    f: dict[str, np.ndarray] = {}
    for k, _ in KEYS:
        for w in config.WINDOWS:
            f[f'{k}_{w}h'] = window_sum(b[:, IDX[k]], w)
    for w in config.WINDOWS:
        f[f'events_{w}h'] = window_sum(b[:, IDX['n_events']], w)
        f[f'channels_{w}h'] = window_sum(b[:, IDX['n_channels']], w) / w
        f[f'gas_max_{w}h'] = window_ext(b[:, IDX['gas_max']], w, np.max)
    for w in (1, 24, 168):
        n = window_sum(b[:, IDX['gas_n']], w)
        f[f'gas_mean_{w}h'] = np.where(n > 0, window_sum(b[:, IDX['gas_sum']], w) / np.maximum(n, 1), np.nan)
        n = window_sum(b[:, IDX['temp_n']], w)
        f[f'temp_mean_{w}h'] = np.where(n > 0, window_sum(b[:, IDX['temp_sum']], w) / np.maximum(n, 1), np.nan)
    for w in (24, 168):
        f[f'temp_max_{w}h'] = window_ext(b[:, IDX['temp_max']], w, np.max)
        f[f'temp_min_{w}h'] = window_ext(b[:, IDX['temp_min']], w, np.min)
    f['temp_trend'] = f['temp_mean_24h'] - f['temp_mean_168h']
    f['gas_trend'] = f['gas_mean_24h'] - f['gas_mean_168h']
    f['since_event'] = since_last(b[:, IDX['n_events']])
    f['since_gas'] = since_last(b[:, IDX['gas_n']])
    f['since_temp'] = since_last(b[:, IDX['temp_n']])
    f['armed'] = ffill(b[:, IDX['guard_last']])
    f['since_guard'] = since_last(b[:, IDX['arm']] + b[:, IDX['disarm']])
    for k in ONSETS:
        for w in (24, 168, 720, CAP):
            f[f'{k}_{w}h'] = window_sum(b[:, IDX[k]], w)
    for t in config.TYPES:
        f[f'since_{t}'] = since_last(b[:, IDX[f'onset_{t}']])
    for k in TRIGS:
        for w in (1, 6, 24):
            f[f'{k}_{w}h'] = window_sum(b[:, IDX[k]], w)
    for k in OTHER:
        for w in (24, 168):
            f[f'{k}_{w}h'] = window_sum(b[:, IDX[k]], w)
    f['since_visit'] = since_last(b[:, IDX['visit']])
    # весь коллектор: подсистемы одного коллектора висят на разных объектах
    for j, k in enumerate(ONSETS + TRIGS):
        for w in (24, 168):
            f[f'coll_{k}_{w}h'] = window_sum(coll[:, j], w)
    return f


def valid_hours(n_events: np.ndarray) -> np.ndarray:
    """Часы, где окно 7 сут назад и FUTURE часов вперёд целиком внутри данных объекта."""
    hours = np.arange(NH)
    ok = hours + FUTURE < NH
    for a, b in HOLES:
        ha, hb = (int((x - T0).total_seconds() // 3600) for x in (a, b))
        ok &= ~((hours + FUTURE >= ha) & (hours - config.WARMUP < hb))
    first = np.argmax(n_events > 0) if (n_events > 0).any() else NH
    ok &= hours >= first + config.WARMUP
    return ok


def main() -> None:
    t = time.time()
    con = config.connect(read_only=True)
    base, objects, oi = load_base(con)
    print('часовые ряды', base.shape, round(time.time() - t), 'с', flush=True)

    meta = con.sql('SELECT object_id, collector_id, kind FROM obj3 ORDER BY object_id').fetchnumpy()
    collectors = dense(meta['collector_id']).astype(np.int64)
    stypes = [r[0] for r in con.sql('SELECT DISTINCT stype FROM ch ORDER BY 1').fetchall()]
    composition = np.zeros((len(objects), len(stypes)), np.float32)
    for o, s, n in con.sql('SELECT object_id, stype, count(*) FROM ch WHERE object_id IN '
                           '(SELECT object_id FROM obj3) GROUP BY ALL').fetchall():
        composition[oi[o], stypes.index(s)] = n
    targets = {}
    for tp in config.TYPES:
        for suffix, cond in (('', ''), ('_conf', 'AND confirmed'), ('_prim', 'AND primary_')):
            d = con.sql(f"""SELECT object_id, {hour_of('t0')} AS h, count(*) AS n FROM inc
                            WHERE type = '{tp}' AND noise IS NULL {cond}
                              AND object_id IN (SELECT object_id FROM obj3) GROUP BY ALL""").fetchnumpy()
            arr = np.zeros((len(objects), NH), np.float32)
            h = dense(d['h']).astype(np.int64)
            keep = (h >= 0) & (h < NH)
            arr[np.array([oi[x] for x in d['object_id']])[keep], h[keep]] = dense(d['n'])[keep]
            targets[f'next_{tp}{suffix}'] = np.stack([hours_to_next(a) for a in arr])
    con.close()

    onset_cols = [IDX[k] for k in ONSETS + TRIGS]
    coll_total = {c: base[collectors == c][:, :, onset_cols].sum(axis=0) for c in np.unique(collectors)}
    cal = calendar()
    hours = np.arange(NH)
    stamp = np.datetime64(T0, 's') + (hours + 1) * np.timedelta64(1, 'h')  # момент прогноза — конец часа
    year = stamp.astype('datetime64[Y]').astype(int) + 1970
    ok = np.zeros((len(objects), NH), bool)

    out = config.WORK / 'features'
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob('*.parquet'):
        p.unlink()
    writers: dict[int, pq.ParquetWriter] = {}
    rows = 0
    for o, object_id in enumerate(objects):
        b = base[o]
        ok[o] = valid_hours(b[:, IDX['n_events']])
        f = object_features(b, coll_total[collectors[o]])
        keep = np.flatnonzero(ok[o] & (f['events_168h'] > 0))
        if not len(keep):
            continue
        cols = {'object_id': np.full(len(keep), object_id, np.int32),
                'collector_id': np.full(len(keep), collectors[o], np.int32),
                'ts': stamp[keep], 'h': hours[keep].astype(np.int32),
                'guard_object': np.full(len(keep), meta['kind'][o] == 'guardObject', np.float32)}
        cols.update({k: v[keep] for k, v in cal.items()})
        cols.update({f'comp_{i}': np.full(len(keep), composition[o, i], np.float32) for i in range(len(stypes))})
        cols.update({k: v[keep] for k, v in f.items()})
        cols.update({k: v[o][keep] for k, v in targets.items()})
        table = pa.table(cols)
        for y in np.unique(year[keep]):
            part = table.filter(pc.equal(pc.year(table['ts']), int(y)))
            if int(y) not in writers:
                writers[int(y)] = pq.ParquetWriter(out / f'{int(y)}.parquet', table.schema, compression='zstd')
            writers[int(y)].write_table(part)
        rows += len(keep)
        print(f'  объект {object_id}: {len(keep)} ч', flush=True)
    for w in writers.values():
        w.close()
    features = [c for c in cols if not c.startswith('next_') and c not in ('object_id', 'collector_id', 'ts', 'h')]
    (out / 'meta.json').write_text(json.dumps(
        {'features': features, 'targets': list(targets), 'stypes': stypes, 'keys': BASE,
         'next_cap': NEXT_CAP, 'future': FUTURE, 'known': KNOWN}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('витрина', rows, 'строк,', len(features), 'признаков за', round(time.time() - t), 'с', flush=True)

    # Часовые ряды для нейросети: знаковый log1p, float16; по объекту, чтобы не держать копии base
    seq = np.empty(base.shape, np.float16)
    for o in range(len(objects)):
        x = np.nan_to_num(base[o], nan=0.0)
        seq[o] = np.sign(x) * np.log1p(np.abs(x))
    del base
    np.savez(config.WORK / 'seq.npz', base=seq, objects=np.array(objects), collectors=collectors,
             composition=composition, ok=ok, **targets)
    print('seq.npz', seq.shape, round(time.time() - t), 'с')


if __name__ == '__main__':
    main()
