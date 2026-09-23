"""Шаг 8. Ретропрогон 2026 года «как в реальном времени» (Ф5-1).

Каждые сутки 2026 года в 07:00 — к началу смены — система видит только строки журнала, пришедшие
до этого момента. По ним заново строятся разметка (labels.build) и часовые ряды (features.load_base),
и модель выдаёт прогноз на 24 ч по каждому объекту и типу. Проверяются две вещи.

1. Нет признаков из будущего. Вектор признаков из обрезанного журнала сравнивается со строкой
   витрины за тот же час. Разметка местами смотрит вперёд (Н8 — 10 мин, персонал — 15 мин, выезд —
   30 мин); если где-то забыт сдвиг KNOWN, строки разойдутся.
2. Что прогноз дал бы диспетчеру: по суткам — сколько тревог, сколько из них сбылось, сколько
   эпизодов поймано и за сколько часов до начала.

Журнал для каждого момента — 100 суток до него (самое длинное окно признаков — 90 суток) плюс вся
история режима охраны: для проникновения нужно знать, стоит ли объект на охране.

Смесь раздела 34 (`--run "$MIX5"`, тип~прогоны) считается так же, как в эксплуатации: оценка
каждого зерна переводится в долю его же оценок на проверке 2025 (`operating.load_mix`, `ref`),
доли складываются, а порог — доля часов под тревогой по оценкам всего парка за 90 суток до
момента прогноза (раздел 32). Доли по типам — из раздела 38 (`--shares`).

У отказа оборудования в смеси стоит сеть (раздел 48). Ей нужна не строка витрины, а 168 ч часовых
рядов объекта и его коллектора плюс статика — всё это пересобирается из того же обрезанного
журнала (`seq_pack`), и сверка идёт не с витриной, а с прогнозом сети из рабочего прогона: вход
собран верно, если вероятности совпали.

    python retro.py [--run main_h24] [--days N] > ../work/retro.md
    python retro.py --run "$MIXT" > ../work/retro_mixt.md
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta

import duckdb
import numpy as np
import polars as pl

import config
import features as ft
import labels
import operating as op

HOUR = 7            # момент прогноза — 07:00, начало смены
LOOK_DAYS = 100     # сут журнала перед моментом прогноза
LOOK_H = ft.CAP + 168   # ч часовых рядов: окна до 90 сут плюс запас
TOL = 1e-4
MONTHS = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
SEQ_L = 168         # ч окна сети (seqmodel.L)
# Календарь в статике сети — в том же порядке и с теми же делителями, что в `seqmodel.py`
CAL_SEQ = [('hour', 23), ('dow', 6), ('month', 12), ('doy_sin', 1), ('doy_cos', 1),
           ('holiday', 1), ('long_holiday', 1), ('may9', 1), ('days_to_holiday', 60)]


def cutoffs(days: int | None) -> list[datetime]:
    first = datetime(2026, 1, 1, HOUR)
    out = [first + timedelta(days=i) for i in range((config.DATA_END - first).days + 1)]
    return [t for t in out if t < config.DATA_END][:days]


def open_db(first: datetime, last: datetime) -> duckdb.DuckDBPyConnection:
    path = config.WORK / 'retro' / 'retro.duckdb'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    con.sql(f"SET memory_limit='{os.environ.get('TF_MEMORY', '2GB')}'")
    con.sql(f"SET temp_directory='{(config.WORK / 'spill').as_posix()}'")
    con.sql(f"ATTACH '{config.DB.as_posix()}' AS src (READ_ONLY)")
    con.sql('CREATE TABLE obj AS SELECT * FROM src.obj')
    con.sql('CREATE TABLE ch AS SELECT * FROM src.ch')
    con.sql(f"""CREATE TABLE ev_all AS SELECT * FROM src.ev WHERE ts < TIMESTAMP '{last}'
                AND (ts >= TIMESTAMP '{first - timedelta(days=LOOK_DAYS)}' OR stype = 'Состояние охраны')
                ORDER BY ts""")
    con.sql('DETACH src')
    return con


def seq_pack(base: np.ndarray, collectors: np.ndarray, comp: np.ndarray, cal: dict, h: int) -> tuple:
    """Вход сети (`seqmodel.py`) из того же окна журнала: ряды объекта, ряды коллектора и статика.

    Ряды приводятся к тому же виду, что в `work/seq.npz`: nan → 0, знаковый log1p, float16 —
    иначе оценка разойдётся с обучением уже на третьем знаке. Окно сети — 168 ч, но часы с
    прошлого эпизода и счётчики эпизодов за 30 и 90 суток считаются по всему окну ретропрогона:
    в обучении они считались по всей истории объекта и там, и там насыщаются на 90 сутках.
    """
    x = np.nan_to_num(base, nan=0.0)
    x = (np.sign(x) * np.log1p(np.abs(x))).astype(np.float16).astype(np.float32)
    mean = {c: x[collectors == c].mean(0) for c in np.unique(collectors)}
    xin = np.concatenate([x[:, -SEQ_L:], np.stack([mean[c] for c in collectors])[:, -SEQ_L:]], -1)
    ep = np.expm1(x[:, :, [ft.IDX[k] for k in ft.ONSETS]])
    long = [np.log1p(ep[:, -w:].sum(1)) for w in (720, ft.CAP)]
    since = np.log1p(np.stack([[ft.since_last(np.expm1(b[:, ft.IDX[f'onset_{tp}']]))[-1]
                                for tp in config.TYPES] for b in x]))
    day = np.array([cal[k][h] / d for k, d in CAL_SEQ], np.float32)
    s = np.concatenate([np.log1p(comp), np.broadcast_to(day, (len(x), len(day))), since] + long, 1)
    return xin.astype(np.float32), s.astype(np.float32)


def snapshot(con, t: datetime, meta: dict, cal: dict, seq: bool = False):
    """Строки признаков всех объектов на момент t, собранные только из журнала до t."""
    con.sql(f"""CREATE OR REPLACE VIEW ev AS SELECT * FROM ev_all WHERE ts < TIMESTAMP '{t}'
                AND (ts >= TIMESTAMP '{t - timedelta(days=LOOK_DAYS)}' OR stype = 'Состояние охраны')""")
    labels.build(con)
    hi = int((t - ft.T0).total_seconds() // 3600)     # строка витрины h = hi − 1: момент прогноза — конец часа
    lo = hi - LOOK_H
    with contextlib.redirect_stdout(io.StringIO()):
        base, objects, oi = ft.load_base(con, lo, hi)
    # режим охраны мог смениться раньше окна — берём последнее известное состояние
    t_lo = ft.T0 + timedelta(hours=lo)
    g = ft.IDX['guard_last']
    for o, armed in con.sql(f"""SELECT object_id, arg_max(armed::INT, ts) FROM guard
                                WHERE ts < TIMESTAMP '{t_lo}' GROUP BY 1""").fetchall():
        if o in oi and np.isnan(base[oi[o], 0, g]):
            base[oi[o], 0, g] = armed
    info = con.sql('SELECT object_id, collector_id, kind FROM obj3 ORDER BY object_id').fetchall()
    comp = {o: np.zeros(len(meta['stypes']), np.float32) for o in objects}
    for o, s, n in con.sql('SELECT object_id, stype, count(*) FROM ch WHERE object_id IN '
                           '(SELECT object_id FROM obj3) GROUP BY ALL').fetchall():
        comp[o][meta['stypes'].index(s)] = n
    collectors = np.array([c for _, c, _ in info])
    cols = [ft.IDX[k] for k in ft.ONSETS + ft.TRIGS]
    coll = {c: base[collectors == c][:, :, cols].sum(axis=0) for c in np.unique(collectors)}
    h = hi - 1
    rows = []
    for (o, c, kind), b in zip(info, base):
        f = ft.object_features(b, coll[c])
        row = {'object_id': o, 'h': h, 'guard_object': float(kind == 'guardObject')}
        row.update({k: float(v[h]) for k, v in cal.items()})
        row.update({f'comp_{i}': float(x) for i, x in enumerate(comp[o])})
        row.update({k: float(v[-1]) for k, v in f.items()})
        rows.append(row)
    df = pl.DataFrame(rows).select(['object_id', 'h'] + meta['features'])
    if not seq:
        return df, None
    comp_m = np.stack([comp[o] for o, _, _ in info])
    return df, seq_pack(base, collectors, comp_m, cal, h)


def load_models(run: str) -> dict:
    import xgboost as xgb
    from catboost import CatBoostClassifier
    d = config.WORK / 'runs' / run / 'models'
    out = {}
    for tp in config.TYPES:
        if (d / f'xgb_{tp}.json').exists():
            b = xgb.Booster()
            b.load_model(d / f'xgb_{tp}.json')
            b.set_param({'device': 'cpu'})
            out[('xgb', tp)] = lambda X, b=b: b.inplace_predict(X)
        if (d / f'cat_{tp}.cbm').exists():
            m = CatBoostClassifier()
            m.load_model(str(d / f'cat_{tp}.cbm'))
            out[('cat', tp)] = lambda X, m=m: m.predict_proba(X)[:, 1]
    return out


# Доли часов под тревогой по типам — рабочие, из settings/operating.json (раздел 46)
SHARES = ','.join(f'{k}={v}' for k, v in config.shares().items())


def mix_parts(name: str, tp: str) -> list:
    """Прогоны смеси для типа: [(прогон, семейство, вес)], разбор как в `operating.load_mix`."""
    table = dict(part.split('~', 1) for part in name.split(';') if part)
    name = table.get(tp, table.get('*', ''))
    out = []
    for part in name.split('+'):
        if part:
            run, _, w = part.partition(':')
            run, _, m = run.partition('/')
            out.append((run, m or 'xgb', float(w) if w else 1.0))
    return out


def load_mix_models(name: str) -> dict:
    """Для каждого типа — функция (X, S) -> оценка смеси на шкале проверки 2025.

    `S` — оценки сетей по тем же строкам, посчитанные заранее (`net_scores`): сеть смотрит не в
    строку витрины, а в часовые ряды, и считается одна на все типы сразу.
    """
    import xgboost as xgb
    from catboost import CatBoostClassifier
    out = {}
    for tp in config.TYPES:
        fns = []
        for run, m, w in mix_parts(name, tp):
            d = config.WORK / 'runs' / run / 'models'
            if m.startswith('tcn'):
                f = lambda X, S, key=f'{m}_{tp}': S[key]
            elif m == 'xgb':
                b = xgb.Booster()
                b.load_model(d / f'xgb_{tp}.json')
                b.set_param({'device': 'cpu'})
                f = lambda X, S, b=b: b.inplace_predict(X)
            else:
                c = CatBoostClassifier()
                c.load_model(str(d / f'cat_{tp}.cbm'))
                f = lambda X, S, c=c: c.predict_proba(X)[:, 1]
            base = np.sort(op.split(run, 'val', 2025, tp, m)[3])
            fns.append((f, base, w))
        out[tp] = lambda X, S, fns=fns: sum(w * np.searchsorted(base, f(X, S), side='right') / len(base)
                                            for f, base, w in fns)
    return out


def load_nets(name: str) -> dict:
    """Сети, которые встречаются в смеси: семейство -> веса рабочего прогона.

    Одна сеть выдаёт все шесть типов сразу, поэтому грузится по разу на семейство, а не на тип.
    """
    import torch
    from seqmodel import Net
    fams = {(run, m) for tp in config.TYPES for run, m, _ in mix_parts(name, tp) if m.startswith('tcn')}
    out = {}
    for run, m in sorted(fams):
        state = torch.load(config.WORK / 'runs' / run / 'models' / f'{m}.pt', map_location='cpu', weights_only=True)
        width, c_in = state['inp.weight'].shape[:2]
        net = Net(c_in, state['head.0.weight'].shape[1] - 2 * width, len(config.TYPES), width, 0.0)
        net.load_state_dict(state)
        out[m] = net.eval()
    return out


def net_scores(nets: dict, pack: tuple, df: pl.DataFrame, dev: str) -> pl.DataFrame:
    """Оценки сетей на строках момента: те же вероятности, что в прогоне, если вход собран верно."""
    import torch
    x, s = (torch.from_numpy(a).to(dev) for a in pack)
    cols = {'object_id': df['object_id'], 'h': df['h']}
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
        for m, net in nets.items():
            p = torch.sigmoid(net.to(dev)(x, s).float()).cpu().numpy()
            cols.update({f'{m}_{tp}': p[:, k] for k, tp in enumerate(config.TYPES)})
    return pl.DataFrame(cols)


def run_scores(run: str, m: str, tp: str, obj: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Оценка сети из рабочего прогона на тех же строках — с ней сверяется пересчёт по журналу."""
    d = config.WORK / 'runs' / run / 'preds'
    idx = np.load(d / 'index_test.npz')
    key = idx['object_id'].astype(np.int64) * 10 ** 7 + idx['h']
    order = np.argsort(key)
    want = obj.astype(np.int64) * 10 ** 7 + h
    pos = order[np.searchsorted(key[order], want)]
    assert np.array_equal(key[pos], want), f'в прогнозах {run}/{m} нет части строк ретропрогона'
    return np.load(d / f'{m}_{tp}_test.npy')[pos]


def mix_history(name: str, tp: str) -> tuple:
    """Оценки смеси по витрине за 2025 и 2026 — история для скользящего порога."""
    _, hv, _, pv = op.load_mix(name, 'val', 2025, tp, 'xgb', ref=('val', 2025))
    _, ht, _, pt = op.load_mix(name, 'test', 2026, tp, 'xgb', ref=('val', 2025))
    h, p = np.concatenate([hv, ht]), np.concatenate([pv, pt])
    order = np.argsort(h, kind='stable')
    return h[order], p[order]


def rolling(h_hist: np.ndarray, p_hist: np.ndarray, hours: np.ndarray, share: float) -> np.ndarray:
    """Порог на каждый момент: квантиль оценок парка за 90 суток строго до часа прогноза."""
    thr = {}
    for x in np.unique(hours):
        lo, hi = np.searchsorted(h_hist, [x - 90 * 24 + 1, x + 1])
        thr[x] = float(np.quantile(p_hist[lo:hi], 1 - share))
    return np.array([thr[x] for x in hours])


def thresholds(run: str) -> dict:
    """Пороги прогона из всех его отчётов сразу.

    Раньше читался единственный файл `report_xgb_cat.json` — имя, которое получается только у
    прогона, обученного командой `--models xgb,cat`. У короткой базы (раздел 24) семейства
    обучались порознь, и файлов два. Собираем из всех, что есть.
    """
    out: dict = {}
    for p in sorted((config.WORK / 'runs' / run).glob('report_*.json')):
        for tp, block in json.loads(p.read_text(encoding='utf-8')).items():
            out.setdefault(tp, {'scores': {}})['scores'].update(block['scores'])
    assert out, f'у прогона {run} нет ни одного отчёта'
    return out


def md(df: pl.DataFrame) -> str:
    fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(df.columns) + ' |', '|' + '---|' * len(df.columns)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in df.rows()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24')
    ap.add_argument('--days', type=int, default=None, help='только первые N суток — для пробы')
    ap.add_argument('--shares', default=SHARES, help='смесь: доля часов под тревогой по типам')
    args = ap.parse_args()
    mix = '~' in args.run
    t = time.time()
    meta = json.loads((config.WORK / 'features' / 'meta.json').read_text(encoding='utf-8'))
    feats = meta['features']
    report = None if mix else thresholds(args.run)
    H = config.HORIZON
    ts = cutoffs(args.days)
    hours = [int((x - ft.T0).total_seconds() // 3600) - 1 for x in ts]
    batch = (pl.scan_parquet(config.WORK / 'features' / '2026.parquet').filter(pl.col('h').is_in(hours))
             .select(['object_id', 'h'] + feats + [f'next_{tp}' for tp in config.TYPES]).collect())
    have = set(batch['h'].to_list())
    ts = [x for x, h in zip(ts, hours) if h in have]
    print(f'моментов прогноза {len(ts)}, строк витрины {len(batch)}', file=sys.stderr, flush=True)

    con = open_db(ts[0], ts[-1])
    print(f'журнал для ретропрогона: {con.sql("SELECT count(*) FROM ev_all").fetchone()[0]:,} строк, '
          f'{time.time() - t:.0f} с', file=sys.stderr, flush=True)
    cal = ft.calendar()
    nets = load_nets(args.run) if mix else {}
    dev = 'cpu'
    if nets:
        import torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'сетей в смеси: {len(nets)} ({dev})', file=sys.stderr, flush=True)
    snaps, seqs = [], []
    for i, x in enumerate(ts):
        t1 = time.time()
        df, pack = snapshot(con, x, meta, cal, seq=bool(nets))
        snaps.append(df)
        if nets:
            seqs.append(net_scores(nets, pack, df, dev))
        print(f'  {x:%Y-%m-%d %H:%M}: {time.time() - t1:.0f} с', file=sys.stderr, flush=True)
    con.close()
    retro = pl.concat(snaps).join(batch.select('object_id', 'h'), on=['object_id', 'h'], how='semi')
    batch = batch.join(retro.select('object_id', 'h'), on=['object_id', 'h'], how='semi')
    retro, batch = retro.sort('object_id', 'h'), batch.sort('object_id', 'h')
    out_dir = config.WORK / 'retro'
    retro.write_parquet(out_dir / 'rows.parquet')

    # 1. сверка с витриной
    R = retro.select(feats).to_numpy().astype(np.float64)
    B = batch.select(feats).to_numpy().astype(np.float64)
    same = (np.isnan(R) & np.isnan(B)) | (np.abs(R - B) <= TOL * np.maximum(1, np.abs(B)))
    bad = [(feats[j], int((~same[:, j]).sum())) for j in range(len(feats)) if not same[:, j].all()]
    print(f'\n## Сверка с витриной\n\nСтрок {len(retro):,} ({len(ts)} моментов × объекты), признаков {len(feats)}. '
          f'Совпало значений {same.sum():,} из {same.size:,}.\n')
    if bad:
        print(md(pl.DataFrame(bad, schema=['признак', 'строк с расхождением'], orient='row')))
        j = feats.index(bad[0][0])
        for i in np.flatnonzero(~same[:, j])[:5].tolist():
            print(f'- {bad[0][0]}: объект {retro["object_id"][i]}, h {retro["h"][i]}: '
                  f'ретро {R[i, j]}, витрина {B[i, j]}')

    # 2. прогнозы: тот же вход — тот же выход; сутки за сутками
    models = load_mix_models(args.run) if mix else load_models(args.run)
    shares = {k: float(v) for k, v in (x.split('=') for x in args.shares.split(','))}
    Xr = np.ascontiguousarray(R, dtype=np.float32)
    Xb = np.ascontiguousarray(B, dtype=np.float32)
    obj_k, h_k = retro['object_id'].to_numpy(), retro['h'].to_numpy()
    Sr, Sb = {}, {}
    if nets:
        # оценки сети: слева пересчитанные по обрезанному журналу, справа — из рабочего прогона
        cols = retro.select('object_id', 'h').join(pl.concat(seqs), on=['object_id', 'h'], how='left')
        for tp in config.TYPES:
            for run, m, _ in mix_parts(args.run, tp):
                if m.startswith('tcn'):
                    Sr[f'{m}_{tp}'] = cols[f'{m}_{tp}'].to_numpy()
                    Sb[f'{m}_{tp}'] = run_scores(run, m, tp, obj_k, h_k)
        print('\n## Сверка оценок сети\n\nВход сети — часовые ряды, а не строка витрины, поэтому он '
              'сверяется отдельно: вероятность, пересчитанная по обрезанному журналу, против '
              'вероятности того же зерна из рабочего прогона на той же строке.\n')
        d = [{'сеть и тип': k, 'макс |Δp|': float(np.abs(Sr[k] - Sb[k]).max()),
              'медиана |Δp|': float(np.median(np.abs(Sr[k] - Sb[k]))),
              'ранговая корреляция': float(np.corrcoef(np.argsort(np.argsort(Sr[k])),
                                                       np.argsort(np.argsort(Sb[k])))[0, 1])} for k in sorted(Sr)]
        print(md(pl.DataFrame(d)))
    day = (retro['h'].to_numpy() + 1) // 24
    lines, per_day = [], []
    for tp in config.TYPES:
        nxt = batch[f'next_{tp}'].to_numpy()
        y = nxt <= H
        scores = {'rules': (Xr[:, feats.index(f'trig_{tp}_24h')] > 0, None)}
        if mix:
            pr, pb = models[tp](Xr, Sr), models[tp](Xb, Sb)
            thr = rolling(*mix_history(args.run, tp), retro['h'].to_numpy(), shares[tp])
            scores['смесь'] = (pr >= thr, float(np.abs(pr - pb).max()))
        for name in () if mix else ('xgb', 'cat'):
            if (name, tp) in models:
                pr, pb = models[(name, tp)](Xr), models[(name, tp)](Xb)
                thr = report[tp]['scores'][name]['test']['threshold']
                scores[name] = (pr >= thr, float(np.abs(pr - pb).max()))
        for name, (alarm, dp) in scores.items():
            hit = alarm & y
            lines.append({'тип': tp, 'модель': name, 'суток': len(ts), 'тревог в сутки': alarm.sum() / len(ts),
                          'Precision': hit.sum() / max(alarm.sum(), 1),
                          'эпизодов (объекто-суток)': int(y.sum()), 'поймано': int(hit.sum()),
                          'Recall': hit.sum() / max(y.sum(), 1),
                          'упреждение, медиана ч': float(np.median(nxt[hit])) if hit.any() else float('nan'),
                          'макс |Δp| с витриной': dp if dp is not None else float('nan')})
            per_day.append(pl.DataFrame({'type': tp, 'model': name, 'day': day, 'object_id': retro['object_id'],
                                         'alarm': alarm, 'y': y, 'next': nxt}))
    per_day = pl.concat(per_day)
    per_day.write_parquet(out_dir / 'days.parquet')
    print('\n## Прогноз в 07:00 на 24 ч\n')
    print(md(pl.DataFrame(lines)))

    # 3. по месяцам: держится ли качество в течение полугода
    hit = (pl.col('alarm') & pl.col('y')).sum()
    month = (per_day.with_columns(month=(pl.lit(ft.T0) + pl.duration(days=pl.col('day'))).dt.month())
             .group_by('type', 'model', 'month', maintain_order=True)
             .agg(pr=pl.format('{} / {}', (hit / pl.col('alarm').sum().clip(1)).round(2),
                               (hit / pl.col('y').sum().clip(1)).round(2)))
             .pivot(on='month', index=['type', 'model'], values='pr'))
    month = month.rename({c: MONTHS[int(c) - 1] for c in month.columns[2:]})
    print('\n## Precision / Recall по месяцам\n')
    print(md(month))
    print(f'\nготово за {time.time() - t:.0f} с')


if __name__ == '__main__':
    main()
