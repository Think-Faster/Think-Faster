"""Прогон вперёд TCN против CatBoost в часах: поймано и свежих при равных ложных часах.

Тест 2026 режется на отрезки по датам переобучения (раз в 30 суток с 2026-01-01). Прогноз каждого
отрезка берётся у модели, обученной к его началу. Оценка переводится в долю внутри отрезка: у
каждой новой модели своя шкала, а порог в работе держится долей часов под тревогой (раздел 32),
так что на каждом отрезке под тревогой одна и та же доля. Зёрна смешиваются средним этих долей.

Семейства (argv[2:], через пробел):
  mix               — прогоны раздела 34 из MIX5 (обучены до 2024 года, дальше не трогаются)
  tcn24             — пять зёрен TCN раздела 42 (обучены до 2024 года)
  cat:<стратегия>   — retrain.py --save: work/roll/roll_cat_<тип>_<стратегия>.npz
  tcn:<стратегия>:<зёрна> — seqmodel.py --cutoff: all, d365, ft (дообучение), frozen (сеть на 2026-01-01)
Первое семейство — опорное: у остальных поймано и свежих считаются при его ложных часах.

    python rollcmp.py "$MIX5" cat:all mix tcn24 cat:frozen tcn:frozen:0 tcn:all:0 tcn:ft:0
"""
import sys
from pathlib import Path

import numpy as np
import config
import metrics
import operating

MIX = sys.argv[1]
FAMS = sys.argv[2:]
H = 24
S38 = {'fire': .030, 'gas': .026, 'flood': .022, 'equipment': .097, 'sensor': .026, 'intrusion': .005}
SHARES = np.unique(np.round(np.geomspace(0.001, 0.3, 90), 6))
CUTS = ['2026-01-01', '2026-01-31', '2026-03-02', '2026-04-01', '2026-05-01', '2026-05-31']
T0 = np.datetime64('2019-01-01')
CUT_H = np.array([int((np.datetime64(c) - T0) / np.timedelta64(1, 'h')) for c in CUTS])
ROLL = config.WORK / 'roll'
SEEDS5 = '+'.join(['main_h24/tcn'] + [f'main_h24/tcn_s{s}' for s in range(1, 5)])


def pct_fold(p, fold):
    out = np.empty(len(p))
    for f in np.unique(fold):
        m = fold == f
        out[m] = p[m].argsort().argsort() / m.sum()
    return out


def family(spec, tp, o, h, key, fold, order):
    if spec == 'mix':
        o2, h2, _, p = operating.load_mix(MIX, 'test', [2026], tp, 'xgb', '')
        return align(o2, h2, p, key)
    if spec == 'tcn24':
        return operating.load_mix(SEEDS5, 'test', [2026], tp, 'xgb', '')[3]
    kind, st, *rest = spec.split(':')
    if kind == 'cat':
        z = np.load(ROLL / f'roll_cat_{tp}_{st}.npz')
        return align(z['o'], z['h'], z['p'], key)
    acc = 0
    for s in rest[0].split(','):
        p = np.zeros(len(o))
        for f, c in enumerate(CUTS):
            if st == 'frozen' or (st == 'ft' and f == 0):
                name = f'all_s{s}_{CUTS[0]}'
            else:
                name = f'{st}_s{s}_{c}'
            q = np.load(ROLL / f'{name}_{tp}.npy')[order]
            p[fold == f] = q[fold == f]
        acc = acc + pct_fold(p, fold)
    return acc


def align(o2, h2, p, key):
    k2 = o2.astype(np.int64) * 10**7 + h2
    order = np.argsort(k2)
    pos = np.searchsorted(k2[order], key)
    assert np.array_equal(k2[order][pos], key), 'строки не совпали'
    return p[order][pos]


def measure(d, p, share):
    alarm = p >= np.quantile(p, 1 - share)
    W = d['W']
    Wv = np.where(W >= 0, W, 0)
    hit = alarm[Wv] & (W >= 0)
    caught = hit.any(1)
    first = hit.argmax(1)
    idx = np.flatnonzero(alarm)
    pos = np.full(len(p), -1, np.int64)
    if len(idx):
        oo, hh = d['o'][idx], d['h'][idx]
        start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
        pos[idx] = hh - hh[start][np.cumsum(start) - 1]
    stand = caught & (pos[Wv[np.arange(len(W)), first]] + (H - first) >= metrics.RUN_CAP)
    return int((alarm & ~d['y']).sum()), int(caught.sum()), int((caught & ~stand).sum())


def at_cost(d, p, cost):
    pts = np.array([measure(d, p, s) for s in SHARES], float)
    return [float(np.interp(cost, pts[:, 0], np.maximum.accumulate(pts[:, j]))) for j in (1, 2)]


types = [a.split('=', 1)[1] for a in FAMS if a.startswith('--types=')]
FAMS = [a for a in FAMS if not a.startswith('--types=')]
TYPES = types[0].split(',') if types else ['equipment', 'fire']
print('| тип | семейство | поймано / свежих / ложных ч при доле раздела 38 | '
      'поймано при ложных ч опоры | свежих при ложных ч опоры |')
print('|---|---|---|---:|---:|')
for tp in TYPES:
    o, h, n, _ = operating.load_mix(SEEDS5, 'test', [2026], tp, 'xgb', '')
    order = np.lexsort((h, o))
    o, h, n = o[order], h[order], n[order]
    key = o.astype(np.int64) * 10**7 + h
    fold = np.searchsorted(CUT_H, h, side='right') - 1
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    d = dict(o=o, h=h, y=n <= H, W=W[(W >= 0).any(1)])
    ref = None
    for spec in FAMS:
        try:
            raw = family(spec, tp, o, h, key, fold, order)
        except FileNotFoundError as e:
            print(f'| {tp} | {spec} | нет файла {Path(e.filename).name} | | |', flush=True)
            continue
        if spec == 'tcn24':
            raw = raw[order]
        p = pct_fold(np.asarray(raw, float), fold)
        m = measure(d, p, S38[tp])
        if ref is None:
            ref = measure(d, p, S38[tp])[0]
        kc, kf = at_cost(d, p, ref)
        print(f'| {tp} | {spec} | {m[1]} / {m[2]} / {m[0]} | {kc:.0f} | {kf:.0f} |', flush=True)
