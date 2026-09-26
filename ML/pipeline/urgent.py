"""Срочный контур на 6 ч поверх суточного. Что он добавляет к суточной модели на рабочей доле.

Суточная — CatBoost ×5 на входах A′ (stfx_main_h24*, газ — main_h24*), тревога на доле config.shares().
Срочная — CatBoost ×5 с горизонтом 6 ч (main_h6<цель>*), своя доля s6 из сетки.
Добавка: эпизоды, не пойманные суточной за 24 ч, но пойманные срочной за 6 ч (упреждение ≥ 1 ч).
Цена: ложные сигналы срочной вне тревог суточной (серия часов, за которой 6 ч нет эпизода),
в долях ложных сигналов суточной (серия, за которой 24 ч нет эпизода).

    TF_WORK=work_pa python urgent.py _prim [зёрен]      # analytics.md, раздел 65
"""
import sys
import numpy as np
import config
import operating

TG = sys.argv[1] if len(sys.argv) > 1 else '_prim'
TYPES = config.TYPES
S = config.shares()
S6 = [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
SEEDS = ['', '_s1', '_s2', '_s3', '_s4'][:int(sys.argv[2]) if len(sys.argv) > 2 else 5]
YEARS = {'val': [2025], 'test': [2026]}


def pct(p):
    return p.argsort().argsort() / len(p)


def score(runs, on, yy, tp):
    acc, base = 0, None
    for r in runs:
        o, h, n, p = operating.split(r, on, yy, tp, 'cat')
        if base is None:
            base = (o, h, n)
        acc = acc + pct(p)
    return base, acc / len(runs)


def runs_of(o, h, m):
    """Серии подряд идущих часов по объекту: номер серии для каждой строки с m, иначе -1."""
    idx = np.flatnonzero(m)
    lab = np.full(len(m), -1)
    if len(idx):
        st = np.r_[True, (o[idx][1:] != o[idx][:-1]) | (h[idx][1:] - h[idx][:-1] > 1)]
        lab[idx] = np.cumsum(st) - 1
    return lab


def false_runs(o, h, m, ok):
    lab = runs_of(o, h, m)
    k = lab.max() + 1
    if k <= 0:
        return 0
    good = np.zeros(k, bool)
    np.logical_or.at(good, lab[m], ok[m])
    return int((~good).sum())


def caught(key, eps, alarm, W):
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(W, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    ok = key[pos] == wk
    hit = alarm[pos] & ok
    lead = np.where(hit.any(1), W - hit.argmax(1), 0)
    return hit.any(1), lead


print(f'# R1: срочный контур 6 ч (цель {TG or "все эпизоды"}) поверх суточного\n')
for tp in TYPES:
    base24 = [('main_h24' if tp == 'gas' else 'stfx_main_h24') + s for s in SEEDS]
    run6 = [f'main_h6{TG}' + s for s in SEEDS]
    print(f'## {tp}, доля суточной {S[tp]:.3f}\n')
    print('| период | s6 | первичных | пойм. суточной | +срочной | медиана упрежд., ч | всех эпизодов | +срочной '
          '| ложных сигн. суточной | +срочной (доля) | суточная с той же прибавкой: +первичных |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for on, yy in YEARS.items():
        (o, h, n), p24 = score(base24, on, yy, tp)
        np_ = operating.split(base24[0], on, yy, tp, 'cat', '_prim')[2]
        _, p6 = score(run6, on, yy, tp)
        order = np.lexsort((h, o))
        o, h, n, np_, p24, p6 = (a[order] for a in (o, h, n, np_, p24, p6))
        key = o.astype(np.int64) * 10**7 + h
        a24 = p24 >= np.quantile(p24, 1 - S[tp])
        f24 = false_runs(o, h, a24, n <= 24)
        ep_all = np.array(sorted(zip(o[n < 168].tolist(), (h + n)[n < 168].tolist())), np.int64).reshape(-1, 2)
        ep_all = np.unique(ep_all, axis=0)
        mp = np_ < 168
        ep_pr = np.unique(np.array(list(zip(o[mp].tolist(), (h + np_)[mp].tolist())), np.int64).reshape(-1, 2), axis=0)
        c24p, _ = caught(key, ep_pr, a24, 24)
        c24a, _ = caught(key, ep_all, a24, 24)
        # опора при той же цене: поднять долю суточной так, чтобы ложных сигналов прибавилось столько же
        grid = np.unique(np.r_[S[tp], S[tp] * np.geomspace(1.02, 12, 40)])
        cost, gain = [], []
        for sh in grid:
            a = p24 >= np.quantile(p24, 1 - sh)
            cost.append(false_runs(o, h, a, n <= 24) - f24)
            gain.append(int((caught(key, ep_pr, a, 24)[0] & ~c24p).sum()))
        cost, gain = np.maximum.accumulate(cost), np.maximum.accumulate(gain)
        for s6 in S6:
            a6 = p6 >= np.quantile(p6, 1 - s6)
            c6p, l6p = caught(key, ep_pr, a6, 6)
            c6a, _ = caught(key, ep_all, a6, 6)
            gp = c6p & ~c24p
            ga = c6a & ~c24a
            f6 = false_runs(o, h, a6 & ~a24, n <= 6)
            med = f'{np.median(l6p[gp]):.0f}' if gp.any() else '—'
            print(f'| {on} | {s6:g} | {len(ep_pr)} | {c24p.sum()} | {gp.sum()} | {med} | {len(ep_all)} | {ga.sum()} '
                  f'| {f24} | {f6} ({f6 / max(f24, 1):.0%}) | {np.interp(f6, cost, gain):.0f} |', flush=True)
    print()
