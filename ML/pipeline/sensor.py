"""Шаг 7. Отдельная модель отказа датчика (Ф3-3): канал × сутки.

Вопрос модели — какой из исправных сейчас датчиков откажет в ближайшие H суток (по умолчанию 7:
этого хватает, чтобы поставить замену в план ТО). Прогноз делается в конце суток d по событиям
до конца d включительно. Отказ — триггер типа sensor на канале (labels.py: «Неисправен»,
«Отключено устройство», «Не определено», заглушки вместо показаний газа и температуры).

Отказы бывают двух видов, и цель — только первый:
- одиночный — в эти сутки на объекте отказали меньше MASS каналов. Это износ или поломка самого датчика,
  его и нужно менять по плану ТО;
- массовый — в одни сутки отказывают десятки и сотни каналов объекта (рекорд — 517). Это шлейф,
  контроллер или питание. На такие сутки приходится 85–90% канало-суток с отказом. Их видит
  основная модель (объект × час, тип sensor), здесь они — признак объекта, а не цель.

Строка входит в выборку, если канал уже появлялся в журнале, подавал события в последний год и
не отказывал последние HEALTHY суток. Без последнего условия задача вырождается в «сломан вчера —
сломан завтра»: у газовых датчиков заглушки идут неделями подряд.

Признаки канала: события за 1/7/30 суток и их изменение, дребезг (смены состояния), «Неопределен»
и «Выключен», тишина, срабатывания; показания газа и температуры — среднее, разброс, тренд,
минимум/максимум, доля нулей, частота показаний; история отказов — число за 30/90/365 суток,
давность, возраст канала. Соседи: отказы других датчиков объекта и коллектора, отказы
оборудования объекта (питание), массовые отказы объекта. Календарь: месяц и день недели. День года
не берём: по нему дерево запоминает даты массовых отказов обучающих лет.

Разбиение то же, что у основной модели: обучение 2022–2024, проверка 2025, тест 2026.
В обучение идут все положительные строки и каждые NEG_STEP-е сутки отрицательных с весом NEG_STEP —
так ожидаемая функция потерь та же, что на полной выборке, а памяти нужно в разы меньше.
Базовые уровни: история отказов канала, давность отказа, частота отказов по типу датчика.

    python sensor.py build
    python sensor.py train --horizon 7                  # одиночные отказы
    python sensor.py train --horizon 7 --target fault   # любые отказы, для сравнения
"""
import argparse
import json
import time
from datetime import date, datetime

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

import config
import metrics
from labels import EQUIPMENT

T0 = datetime(2019, 1, 1)
ND = (config.DATA_END - T0).days
FUTURE = 7          # сут: цель должна быть наблюдаема столько вперёд
NEXT_CAP = 91       # сут: дальше «следующего отказа» не считаем
CAP = 730           # сут: предел давности и возраста
HEALTHY = 7         # сут без отказа, чтобы канал считался исправным
MASS = 3            # столько каналов объекта с отказом в одни сутки — массовый отказ
WARM = 30           # сут после начала данных или дыры, пока окна не заполнены
CHUNK = 500         # каналов за раз: плотные массивы канал × сутки
NEG_STEP = 5
BUDGETS = (0.005, 0.02)  # доля исправных датчиков, которую бригада проверяет в сутки
OUT = config.WORK / 'sensor'
SENSOR = f"stype NOT IN {EQUIPMENT} AND stype <> 'Состояние охраны'"
DAY = "date_diff('day', TIMESTAMP '2019-01-01', ts)"
STATE_FAULTS = "('Неисправен', 'Отключено устройство', 'Не определено')"
GROUPS = {'Датчик дыма': 'дым', 'Датчик температуры': 'температура', 'Газовый датчик': 'газ'}
SPLITS = {'train': [2022, 2023, 2024], 'val': [2025], 'test': [2026]}
DAILY = ['ev', 'undef', 'off', 'nv', 'sv', 'sv2', 'minv', 'maxv', 'zero', 'flips', 'alarm', 'fault', 'fault_state',
         'fault_rows']
YEAR = np.array([date.fromordinal(T0.toordinal() + d).year for d in range(ND)], np.int32)


def wsum(x: np.ndarray, w: int) -> np.ndarray:
    """Сумма по суткам d-w+1 … d вдоль оси 1."""
    cs = np.cumsum(x, axis=1, dtype=np.float64)
    out = cs.copy()
    out[:, w:] -= cs[:, :-w]
    return out.astype(np.float32)


def wext(x: np.ndarray, w: int, fn) -> np.ndarray:
    """Скользящий максимум или минимум по суткам d-w+1 … d, NaN — показаний не было."""
    fill = -np.inf if fn is np.max else np.inf
    padded = np.concatenate([np.full((len(x), w - 1), fill, np.float32), np.where(np.isnan(x), fill, x)], 1)
    out = fn(sliding_window_view(padded, w, axis=1), axis=-1)
    return np.where(np.isinf(out), np.nan, out).astype(np.float32)


def since(x: np.ndarray, cap: int) -> np.ndarray:
    """Сколько суток назад было последнее ненулевое значение (0 — в эти сутки), не больше cap."""
    idx = np.arange(x.shape[1])
    last = np.maximum.accumulate(np.where(x > 0, idx, -cap - 1), axis=1)
    return np.minimum(idx - last, cap).astype(np.float32)


def to_next(x: np.ndarray, cap: int) -> np.ndarray:
    """Через сколько суток после d следующее ненулевое значение (≥ 1), не больше cap."""
    idx = np.arange(x.shape[1])
    first = np.minimum.accumulate(np.where(x > 0, idx, 10 ** 9)[:, ::-1], axis=1)[:, ::-1]
    after = np.concatenate([first[:, 1:], np.full((len(x), 1), 10 ** 9)], 1)
    return np.minimum(after - idx, cap).astype(np.int16)


def days() -> tuple[np.ndarray, np.ndarray]:
    """Годные сутки прогноза и сколько суток прошло с начала данных или конца последней дыры."""
    bad = np.zeros(ND, bool)
    for a, b in [(datetime(2021, 1, 1), datetime(2022, 1, 1))] + config.GAPS:
        bad[(a - T0).days:(b - T0).days] = True
    idx = np.arange(ND)
    fresh = idx - np.maximum.accumulate(np.where(bad, idx + 1, 0))
    cb = np.concatenate([[0], np.cumsum(bad)])
    ahead = cb[np.minimum(idx + FUTURE + 1, ND)] - cb[np.minimum(idx + 1, ND)]
    ok = ~bad & (fresh >= WARM) & (ahead == 0) & (idx + FUTURE < ND)
    return ok, fresh.astype(np.float32)


def calendar() -> dict[str, np.ndarray]:
    dates = [date.fromordinal(T0.toordinal() + d) for d in range(ND)]
    return {'month': np.array([x.month for x in dates], np.float32),
            'dow': np.array([x.weekday() for x in dates], np.float32)}


def load_daily(con) -> tuple[dict, dict, dict]:
    """Каналы-датчики, их суточные агрегаты (разреженно) и отказы оборудования по объектам."""
    con.sql(f"""CREATE TEMP TABLE sch AS
                SELECT c.channel_id, c.stype, o.object_id, o.collector_id
                FROM ch c JOIN obj3 o ON o.object_id = c.object_id WHERE {SENSOR}""")
    con.sql(f"""
        CREATE TEMP TABLE daily AS
        WITH e AS (
            SELECT channel_id, {DAY} AS d, state,
                   CASE WHEN stype = 'Газовый датчик' AND num >= 0 AND num < 327.68 THEN num
                        WHEN stype = 'Датчик температуры' AND num BETWEEN -40 AND 80 THEN num END AS v
            FROM ev WHERE {SENSOR}
        ), agg AS (
            SELECT channel_id, d, count(*) AS ev, count(*) FILTER (WHERE state = 'Неопределен') AS undef,
                   count(*) FILTER (WHERE state = 'Выключен') AS off, count(v) AS nv,
                   coalesce(sum(v), 0) AS sv, coalesce(sum(v * v), 0) AS sv2,
                   coalesce(min(v), 0) AS minv, coalesce(max(v), 0) AS maxv,
                   count(*) FILTER (WHERE v = 0) AS zero
            FROM e GROUP BY 1, 2
        ), flips AS (
            SELECT channel_id, d, count(*) AS flips FROM (
                SELECT channel_id, {DAY} AS d, state <> lag(state) OVER (PARTITION BY channel_id ORDER BY ts) AS flip
                FROM ev WHERE {SENSOR} AND state IS NOT NULL) WHERE flip GROUP BY 1, 2
        ), tr AS (
            SELECT channel_id, {DAY} AS d, count(*) FILTER (WHERE type <> 'sensor') AS alarm,
                   (count(*) FILTER (WHERE type = 'sensor') > 0)::INT AS fault,
                   (count(*) FILTER (WHERE type = 'sensor' AND what IN {STATE_FAULTS}) > 0)::INT AS fault_state,
                   count(*) FILTER (WHERE type = 'sensor') AS fault_rows
            FROM trig GROUP BY 1, 2
        )
        SELECT channel_id, d, agg.* EXCLUDE (channel_id, d), coalesce(flips, 0) AS flips,
               coalesce(alarm, 0) AS alarm, coalesce(fault, 0) AS fault,
               coalesce(fault_state, 0) AS fault_state, coalesce(fault_rows, 0) AS fault_rows
        FROM agg LEFT JOIN flips USING (channel_id, d) LEFT JOIN tr USING (channel_id, d)
        WHERE channel_id IN (SELECT channel_id FROM sch) AND d < {ND}""")
    ch = con.sql('SELECT * FROM sch WHERE channel_id IN (SELECT channel_id FROM daily) ORDER BY channel_id').fetchnumpy()
    daily = con.sql('SELECT * FROM daily ORDER BY channel_id, d').fetchnumpy()
    equip = con.sql(f"""SELECT object_id, {DAY} AS d, count(*) AS n FROM trig
                        WHERE type = 'equipment' AND ts < TIMESTAMP '{config.DATA_END}' GROUP BY 1, 2""").fetchnumpy()
    return ch, daily, equip


def build() -> None:
    t = time.time()
    con = config.connect(read_only=True)
    con.sql('SET enable_progress_bar=false')
    ch, daily, equip = load_daily(con)
    con.close()
    ids = ch['channel_id']
    nc = len(ids)
    ci = np.searchsorted(ids, daily['channel_id'])
    dd = daily['d'].astype(np.int64)
    print(f'каналов-датчиков {nc}, канало-суток с событиями {len(ci):,} за {time.time() - t:.0f} с', flush=True)

    stypes = sorted(set(ch['stype'].tolist()))
    objects, oi = np.unique(ch['object_id'], return_inverse=True)
    colls, cli = np.unique(ch['collector_id'], return_inverse=True)
    # отказы по всем каналам — для соседей по объекту и коллектору
    fault_all = np.zeros((nc, ND), np.float32)
    fault_all[ci, dd] = daily['fault']
    obj_fault = np.zeros((len(objects), ND), np.float32)
    np.add.at(obj_fault, oi, fault_all)
    coll_fault = np.zeros((len(colls), ND), np.float32)
    np.add.at(coll_fault, cli, fault_all)
    del fault_all
    obj_n = np.bincount(oi, minlength=len(objects)).astype(np.float32)
    obj_equip = np.zeros((len(objects), ND), np.float32)
    m = np.isin(equip['object_id'], objects)
    np.add.at(obj_equip, (np.searchsorted(objects, equip['object_id'][m]), equip['d'][m].astype(np.int64)),
              equip['n'][m])
    obj_fault_w = {w: wsum(obj_fault, w) for w in (7, 30, 365)}
    obj_mass = obj_fault >= MASS
    obj_mass_w = {w: wsum(obj_mass.astype(np.float32), w) for w in (7, 30, 365)}
    coll_fault_w = {w: wsum(coll_fault, w) for w in (7, 30)}
    obj_equip_w = {w: wsum(obj_equip, w) for w in (1, 7)}

    ok_day, fresh = days()
    cal = calendar()
    first = np.full(nc, ND, np.int64)
    np.minimum.at(first, ci, dd)
    idx = np.arange(ND)
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob('part_*.parquet'):
        p.unlink()
    total, names = 0, []
    for part, c0 in enumerate(range(0, nc, CHUNK)):
        c1 = min(c0 + CHUNK, nc)
        n = c1 - c0
        sel = slice(np.searchsorted(ci, c0), np.searchsorted(ci, c1))
        r, c = ci[sel] - c0, dd[sel]
        x = {}
        for k in DAILY:
            x[k] = np.zeros((n, ND), np.float32)
            x[k][r, c] = daily[k][sel]
        f = {}
        for w in (1, 7, 30):
            f[f'ev_{w}'] = wsum(x['ev'], w)
            f[f'flips_{w}'] = wsum(x['flips'], w)
            f[f'undef_{w}'] = wsum(x['undef'], w)
        f['ev_change'] = f['ev_7'] / (f['ev_30'] * 7 / 30 + 1)
        f['off_7'], f['off_30'] = wsum(x['off'], 7), wsum(x['off'], 30)
        f['alarm_7'], f['alarm_30'] = wsum(x['alarm'], 7), wsum(x['alarm'], 30)
        f['silent'] = since(x['ev'], 365)
        f['since_alarm'] = since(x['alarm'], 365)
        nv = {w: wsum(x['nv'], w) for w in (1, 7, 30)}
        sv = {w: wsum(x['sv'], w) for w in (1, 7, 30)}
        with np.errstate(invalid='ignore', divide='ignore'):
            for w in (1, 7, 30):
                f[f'mean_{w}'] = np.where(nv[w] > 0, sv[w] / nv[w], np.nan).astype(np.float32)
            for w in (1, 7):
                var = wsum(x['sv2'], w) / nv[w] - f[f'mean_{w}'] ** 2
                f[f'std_{w}'] = np.where(nv[w] > 0, np.sqrt(np.maximum(var, 0)), np.nan).astype(np.float32)
            f['zero_7'] = np.where(nv[7] > 0, wsum(x['zero'], 7) / nv[7], np.nan).astype(np.float32)
        f['trend'] = f['mean_1'] - f['mean_30']
        f['readings_1'] = nv[1]
        f['readings_change'] = nv[1] / (nv[30] / 30 + 1)
        has = x['nv'] > 0
        f['min_7'] = wext(np.where(has, x['minv'], np.nan), 7, np.min)
        f['max_7'] = wext(np.where(has, x['maxv'], np.nan), 7, np.max)
        fault = x['fault']
        fault_7 = wsum(fault, HEALTHY)
        for w in (30, 90, 365):
            f[f'fault_{w}'] = wsum(fault, w)
        f['fault_state_365'] = wsum(x['fault_state'], 365)
        f['fault_rows_365'] = np.log1p(wsum(x['fault_rows'], 365))
        o, cl = oi[c0:c1], cli[c0:c1]
        single = fault * ~obj_mass[o]
        f['single_365'] = wsum(single, 365)
        f['since_fault'] = since(fault, CAP)
        f['since_single'] = since(single, CAP)
        age = (idx[None] - first[c0:c1, None]).astype(np.float32)
        f['age'] = np.minimum(age, CAP)
        f['history'] = np.minimum(age, fresh[None])          # сколько суток истории реально есть
        f['obj_fault_7'] = obj_fault_w[7][o] - fault_7
        f['obj_fault_30'] = obj_fault_w[30][o] - f['fault_30']
        f['obj_fault_rate_365'] = (obj_fault_w[365][o] - f['fault_365']) / np.maximum(obj_n[o] - 1, 1)[:, None]
        f['coll_fault_7'] = coll_fault_w[7][cl] - fault_7
        f['coll_fault_30'] = coll_fault_w[30][cl] - f['fault_30']
        f['obj_equip_1'], f['obj_equip_7'] = obj_equip_w[1][o], obj_equip_w[7][o]
        for w in (7, 30, 365):
            f[f'obj_mass_{w}'] = obj_mass_w[w][o]
        f['obj_sensors'] = np.broadcast_to(obj_n[o][:, None], (n, ND))
        for k, v in cal.items():
            f[k] = np.broadcast_to(v[None], (n, ND))
        st = ch['stype'][c0:c1]
        for i, s in enumerate(stypes):
            f[f'st_{i}'] = np.broadcast_to((st == s).astype(np.float32)[:, None], (n, ND))

        keep = ok_day[None] & (age >= 0) & (fault_7 == 0) & (wsum(x['ev'], 365) > 0)
        rr, dk = np.nonzero(keep)
        cols = {'channel_id': ids[c0:c1][rr], 'object_id': ch['object_id'][c0:c1][rr].astype(np.int32),
                'stype': st[rr].astype(str), 'd': dk.astype(np.int32), 'year': YEAR[dk],
                'next_fault': to_next(fault, NEXT_CAP)[rr, dk],
                'next_single': to_next(single, NEXT_CAP)[rr, dk],
                'next_fault_state': to_next(x['fault_state'], NEXT_CAP)[rr, dk]}
        cols.update({k: np.ascontiguousarray(v[rr, dk]) for k, v in f.items()})
        names = list(f)
        pl.DataFrame(cols).write_parquet(OUT / f'part_{part:03d}.parquet', compression='zstd')
        total += len(rr)
        print(f'  каналы {c0}–{c1}: строк {len(rr):,}, {time.time() - t:.0f} с', flush=True)
    meta = {'features': names, 'stypes': stypes, 'next_cap': NEXT_CAP, 'healthy': HEALTHY, 'future': FUTURE,
            'mass': MASS}
    (OUT / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'готово: {total:,} строк, признаков {len(names)}, {time.time() - t:.0f} с')


def load(years: list[int], columns: list[str], target: str, horizon: int, step: int) -> pl.DataFrame:
    """Строки годов years: все положительные и каждые step-е сутки отрицательных."""
    lf = pl.scan_parquet(OUT / 'part_*.parquet').filter(pl.col('year').is_in(years))
    if step > 1:
        lf = lf.filter((pl.col(target) <= horizon) | (pl.col('d') % step == 0))
    return lf.select(columns).collect()


def xy(df: pl.DataFrame, features: list[str], target: str, horizon: int, step: int):
    X = np.ascontiguousarray(df.select(features).to_numpy().astype(np.float32))
    y = (df[target].to_numpy() <= horizon).astype(np.float32)
    return X, y, np.where(y > 0, 1.0, float(step)).astype(np.float32)


def train(args) -> None:
    import xgboost as xgb
    t = time.time()
    H, tgt = args.horizon, f'next_{args.target}'
    meta = json.loads((OUT / 'meta.json').read_text(encoding='utf-8'))
    features = meta['features']
    tr = load(SPLITS['train'], features + [tgt, 'stype'], tgt, H, NEG_STEP)
    va = load(SPLITS['val'], features + [tgt], tgt, H, NEG_STEP)
    Xt, yt, wt = xy(tr, features, tgt, H, NEG_STEP)
    Xv, yv, wv = xy(va, features, tgt, H, NEG_STEP)
    # частота отказа по типу датчика на обучении — базовый уровень «тип датчика»
    rate = (tr.with_columns(pl.Series('y', yt), pl.Series('w', wt)).group_by('stype')
            .agg((pl.col('y') * pl.col('w')).sum() / pl.col('w').sum()).rows())
    rate = dict(rate)
    del tr, va
    print(f'обучение {Xt.shape}, положительных {int(yt.sum()):,}; проверка {Xv.shape}, {time.time() - t:.0f} с',
          flush=True)
    p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'device': 'cuda',
         'max_depth': 6, 'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.6,
         'min_child_weight': 20, 'max_bin': 256, 'reg_lambda': 5.0}
    dt = xgb.QuantileDMatrix(Xt, yt, weight=wt, max_bin=256, feature_names=features)
    dv = xgb.QuantileDMatrix(Xv, yv, weight=wv, ref=dt, feature_names=features)
    booster = xgb.train(p, dt, num_boost_round=4000, evals=[(dv, 'val')], early_stopping_rounds=200,
                        verbose_eval=False)
    it = booster.best_iteration + 1
    del dt, dv, Xt, Xv
    print(f'xgb: {it} деревьев, PR-AUC проверки (взвешенно) {booster.best_score:.4f}, {time.time() - t:.0f} с',
          flush=True)

    # полные проверка и тест — по файлам, чтобы не держать всё в памяти
    other = 'next_fault' if tgt != 'next_fault' else 'next_single'
    scores = {s: {k: [] for k in ('channel_id', 'd', 'stype', tgt, other, 'xgb', 'history', 'recency',
                                  'stype_rate')} for s in ('val', 'test')}
    for path in sorted(OUT.glob('part_*.parquet')):
        part = pl.read_parquet(path)
        for s in scores:
            df = part.filter(pl.col('year').is_in(SPLITS[s]))
            if not len(df):
                continue
            X = np.ascontiguousarray(df.select(features).to_numpy().astype(np.float32))
            out = scores[s]
            out['xgb'].append(booster.inplace_predict(X, iteration_range=(0, it)))
            hist, since_ = (('single_365', 'since_single') if tgt == 'next_single' else ('fault_365', 'since_fault'))
            out['history'].append(df[hist].to_numpy() + 1e-3 * df['fault_365'].to_numpy())
            out['recency'].append(-df[since_].to_numpy())
            st = df['stype'].to_numpy()
            out['stype_rate'].append(np.array([rate.get(v, 0.0) for v in st]))
            for k in ('channel_id', 'd', tgt, other):
                out[k].append(df[k].to_numpy())
            out['stype'].append(st)
    scores = {s: {k: np.concatenate(v) for k, v in out.items()} for s, out in scores.items()}

    run = config.WORK / 'runs' / f'sensor_{args.target}_h{H}'
    (run / 'preds').mkdir(parents=True, exist_ok=True)
    booster.save_model(run / 'xgb.json')
    report = {'target': tgt, 'horizon_days': H, 'trees': it, 'rows': {s: len(v['d']) for s, v in scores.items()},
              'importance': importance(booster), 'models': {}}
    group = lambda st: np.array([GROUPS.get(v, 'прочие') for v in st])
    grp = {s: group(v['stype']) for s, v in scores.items()}
    for name in ('xgb', 'history', 'recency', 'stype_rate'):
        v_, s_ = scores['val'], scores['test']
        yv_ = (v_[tgt] <= H).astype(np.int8)
        thr = metrics.best_threshold(yv_, v_[name])
        row = {}
        for split, sc in (('val', v_), ('test', s_)):
            args_ = (sc['channel_id'], sc['d'])
            row[split] = metrics.evaluate(*args_, sc[tgt], sc[name], thr, H, NEXT_CAP)
            row[f'{split}_{other[5:]}'] = metrics.evaluate(*args_, sc[other], sc[name], thr, H, NEXT_CAP)
            for g in ('дым', 'температура', 'газ', 'прочие'):
                m = grp[split] == g
                row[f'{split}_{g}'] = metrics.evaluate(sc['channel_id'][m], sc['d'][m], sc[tgt][m],
                                                       sc[name][m], thr, H, NEXT_CAP)
            if name == 'xgb':
                y = (sc[tgt] <= H).astype(np.float32)
                row[split]['ece'] = metrics.ece(y, sc[name])
                np.save(run / 'preds' / f'xgb_{split}.npy', sc[name].astype(np.float32))
                np.savez(run / 'preds' / f'index_{split}.npz', channel_id=sc['channel_id'], d=sc['d'])
        # порог по F1 почти не даёт тревог; для плана ТО честнее сравнить модели при одинаковой нагрузке:
        # каждые сутки проверяется доля BUDGETS исправных датчиков с наибольшим риском
        for b in BUDGETS:
            alarm = daily_top(s_['d'], s_[name], b).astype(np.float32)
            row[f'test_top{b}'] = metrics.evaluate(s_['channel_id'], s_['d'], s_[tgt], alarm, 0.5, H, NEXT_CAP)
        report['models'][name] = row
        te = row['test']
        print(f'  {name:10s} test PR-AUC {te["pr_auc"]:.3f} (база {te["base_rate"]:.4f}) P {te["precision"]:.3f} '
              f'R(отказы) {te["recall_episodes"]:.3f} из {te["episodes"]} | '
              + ' '.join(f'{g} {row["test_" + g]["pr_auc"]:.3f}' for g in ('дым', 'температура', 'газ', 'прочие'))
              + ' | ' + ' '.join(f'топ {b:.1%}: P {row[f"test_top{b}"]["precision"]:.3f} '
                                 f'R {row[f"test_top{b}"]["recall_episodes"]:.3f}' for b in BUDGETS),
              flush=True)
    (run / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'готово за {time.time() - t:.0f} с')


def daily_top(d: np.ndarray, p: np.ndarray, share: float, seed: int = 7) -> np.ndarray:
    """Тревога — у доли share каналов с наибольшим риском в каждые сутки. Порог-квантиль тут не годится:
    у базовых уровней оценки целые, на границе квантиля тысячи равных, и нагрузка выходит вдвое больше
    заданной. Равные оценки разводим случайно."""
    order = np.lexsort((np.random.default_rng(seed).random(len(p)), -np.nan_to_num(p, nan=-np.inf), d))
    ds = d[order]
    start = np.r_[0, np.flatnonzero(np.diff(ds)) + 1]
    n = np.diff(np.r_[start, len(ds)])
    alarm = np.zeros(len(p), bool)
    alarm[order] = np.arange(len(ds)) - np.repeat(start, n) < np.repeat(np.ceil(share * n), n)
    return alarm


def importance(booster, top: int = 20) -> list[tuple[str, float]]:
    g = booster.get_score(importance_type='total_gain')
    total = sum(g.values()) or 1
    return [(k, round(v / total, 4)) for k, v in sorted(g.items(), key=lambda x: -x[1])[:top]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['build', 'train'])
    ap.add_argument('--horizon', type=int, default=7)
    ap.add_argument('--target', default='single', choices=['single', 'fault'])
    args = ap.parse_args()
    build() if args.step == 'build' else train(args)


if __name__ == '__main__':
    main()
