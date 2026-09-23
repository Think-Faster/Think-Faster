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

    python retro.py [--run main_h24] [--days N] > ../work/retro.md
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

import config
import features as ft
import labels

HOUR = 7            # момент прогноза — 07:00, начало смены
LOOK_DAYS = 100     # сут журнала перед моментом прогноза
LOOK_H = ft.CAP + 168   # ч часовых рядов: окна до 90 сут плюс запас
TOL = 1e-4
MONTHS = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']


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


def snapshot(con, t: datetime, meta: dict, cal: dict) -> pl.DataFrame:
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
    return pl.DataFrame(rows).select(['object_id', 'h'] + meta['features'])


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


def seq_pack(con, t: datetime, meta: dict, L: int = 168) -> dict[str, np.ndarray]:
    """Вход сети на момент t (INTEGRATION §1.4): 168 ч + статика, повторяет seqmodel.batch.

    Строится из того же обрезанного журнала, что и snapshot(): часовые ряды от `lo` до t, из них
    — 66 рядов объекта и те же 66, усреднённые по коллектору, знаковый log1p в float16 (см.
    features.main). Статика — состав объекта, календарь часа, часы с прошлого эпизода по шести
    типам и счётчики эпизодов за 30 и 90 суток (CAP), ровно в том порядке, в котором сеть видела
    их при обучении. Возврат: x (объекты × 168 × 132) и s (объекты × статика), float16.
    """
    hi = int((t - ft.T0).total_seconds() // 3600)
    lo = hi - L - ft.CAP
    with contextlib.redirect_stdout(io.StringIO()):
        base, objects, oi = ft.load_base(con, lo, hi)
    nch = base.shape[2]
    arr = np.where(np.isnan(base), 0.0, base).astype(np.float32)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    info = con.sql('SELECT object_id, collector_id FROM obj3 ORDER BY object_id').fetchall()
    collectors = np.array([c for _, c in info], np.int64)
    uniq = np.unique(collectors)
    coll = np.stack([arr[collectors == c].mean(0) for c in uniq]).astype(np.float16)
    o2c = np.searchsorted(uniq, collectors)
    hh = np.arange(hi - L, hi) - lo
    xo = arr[:, hh]
    xc = coll[o2c][:, hh].astype(np.float32)
    x = np.concatenate([xo, xc], -1).astype(np.float16)          # объекты × 168 × 132

    ons = np.nancumsum(np.nan_to_num(base[:, :, [ft.IDX[k] for k in ft.ONSETS]], nan=0.0), 1)
    since = np.stack([np.log1p(ft.since_last(ons[:, :, j])) for j in range(len(ft.ONSETS))], -1)
    th = hh[-1]
    long = np.log1p(np.stack([(ons[:, th] - ons[:, np.clip(th - w, 0, None)])
                              for w in (720, ft.CAP)], -1))
    cal = ft.calendar()
    calv = np.stack([cal['hour'][th] / 23, cal['dow'][th] / 6, cal['month'][th] / 12,
                     cal['doy_sin'][th], cal['doy_cos'][th], cal['holiday'][th],
                     cal['long_holiday'][th], cal['may9'][th], cal['days_to_holiday'][th] / 60], -1)
    comp = np.zeros((len(objects), len(meta['stypes'])), np.float32)
    for o, s, n in con.sql('SELECT object_id, stype, count(*) FROM ch WHERE object_id IN '
                           '(SELECT object_id FROM obj3) GROUP BY ALL').fetchall():
        comp[oi[o], meta['stypes'].index(s)] = n
    s = np.concatenate([np.log1p(comp), np.broadcast_to(calv, (len(objects), 9)),
                        since.astype(np.float32), long.astype(np.float32)], -1).astype(np.float16)
    return {'x': x, 's': s}


def load_nets(export: Path | None = None) -> list:
    """Сети TCN из выгрузки. Возвращает список сетей в порядке manifest.seq.nets."""
    import torch
    import seqmodel
    nets_dir = (export or config.WORK / 'export' / 'nets')
    manifest = json.loads((config.WORK / 'export' / 'manifest.json').read_text(encoding='utf-8'))
    names = manifest['seq'].get('nets', sorted(p.name for p in nets_dir.glob('*.pt')))
    out = []
    for name in names:
        state = torch.load(nets_dir / name, map_location='cpu', weights_only=True)
        net = seqmodel.Net(state['c_in'], state['c_static'], state['n_out'],
                           state['width'], state['dropout'])
        net.load_state_dict(state['state_dict'])
        net.eval()
        out.append(net)
    return out


def net_scores(nets: list, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Среднее sigmoid-выходов сетей: объекты × 6 типов. Только для инференса."""
    import torch
    with torch.no_grad():
        out = []
        xt, st = torch.from_numpy(x), torch.from_numpy(s)
        for o in range(0, len(xt), 4096):
            acc = None
            for net in nets:
                p = torch.sigmoid(net(xt[o:o + 4096], st[o:o + 4096]).float())
                acc = p if acc is None else acc + p
            out.append((acc / len(nets)).numpy())
    return np.concatenate(out)


def load_mix_models(export: Path | None = None) -> dict:
    """Модели выгрузки: тип → {'family', 'seeds': {зерно: вызываемое}}.

    Зерно — лямбда по строке признаков (порядок manifest.features), как в load_models.
    """
    import xgboost as xgb
    from catboost import CatBoostClassifier
    export = export or config.WORK / 'export'
    manifest = json.loads((export / 'manifest.json').read_text(encoding='utf-8'))
    out = {}
    for tp, block in manifest['models'].items():
        fam = block['family']
        seeds = {}
        for seed, info in block['seeds'].items():
            p = export / info['path'] / f'{fam}_{tp}.{EXT[fam]}'
            if fam == 'cat':
                m = CatBoostClassifier()
                m.load_model(str(p))
                seeds[seed] = lambda X, m=m: m.predict_proba(X)[:, 1]
            else:
                b = xgb.Booster()
                b.load_model(p)
                b.set_param({'device': 'cpu'})
                seeds[seed] = lambda X, b=b: b.inplace_predict(X)
        out[tp] = {'family': fam, 'seeds': seeds}
    return out


def rolling(history: pl.DataFrame, share: float, window_days: int = 90) -> float:
    """Квантиль оценок за последние window_days суток — скользящий порог на долю share.

    history: колонки h (час) и score по одному типу. Порог не свойство модели, а место в
    распределении оценок парка (INTEGRATION §2.1): пересчитывается на каждый новый час.
    """
    if not history.height:
        return float('nan')
    last = history['h'].max()
    keep = history.filter(pl.col('h') >= last - window_days * 24)
    return float(keep['score'].quantile(1.0 - share)) if keep.height else float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24')
    ap.add_argument('--days', type=int, default=None, help='только первые N суток — для пробы')
    args = ap.parse_args()
    t = time.time()
    meta = json.loads((config.WORK / 'features' / 'meta.json').read_text(encoding='utf-8'))
    feats = meta['features']
    report = thresholds(args.run)
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
    snaps = []
    for i, x in enumerate(ts):
        t1 = time.time()
        snaps.append(snapshot(con, x, meta, cal))
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
    models = load_models(args.run)
    Xr = np.ascontiguousarray(R, dtype=np.float32)
    Xb = np.ascontiguousarray(B, dtype=np.float32)
    day = (retro['h'].to_numpy() + 1) // 24
    lines, per_day = [], []
    for tp in config.TYPES:
        nxt = batch[f'next_{tp}'].to_numpy()
        y = nxt <= H
        scores = {'rules': (Xr[:, feats.index(f'trig_{tp}_24h')] > 0, None)}
        for name in ('xgb', 'cat'):
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
