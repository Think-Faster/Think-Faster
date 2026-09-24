"""Раздел 44, продолжение: насколько глубоко пропущены эпизоды, какие предвестники у первичных,
«ранние» ложные сигналы против фона, периодичность повторов оборудования.

    python patterns2.py "$MIXT"
"""
import sys
import numpy as np
import pyarrow.parquet as pq

import config
from smooth import SHARE, load

H, GAP = 24, 6
TYPES = list(SHARE)
rng = np.random.default_rng(0)

tabs = {y: pq.read_table(config.WORK / 'features' / f'{y}.parquet') for y in (2025, 2026)}
names = tabs[2026].schema.names
CNT = [c for c in names if c.endswith(('_1h', '_6h', '_24h')) and not c.startswith(('coll_', 'next_'))
       and not c.startswith(('gas_max', 'gas_mean', 'temp_mean'))]
keyv = {}
for y, t in tabs.items():
    keyv[y] = t['object_id'].to_numpy().astype(np.int64) * 10**7 + t['h'].to_numpy().astype(np.int64)
    o = np.argsort(keyv[y])
    keyv[y] = (keyv[y][o], o)


def take(on, o, h, cols):
    y = 2025 if on == 'val' else 2026
    ks, order = keyv[y]
    k = np.asarray(o, np.int64) * 10**7 + np.asarray(h, np.int64)
    i = np.minimum(np.searchsorted(ks, k), len(ks) - 1)
    ok = ks[i] == k
    idx = order[i]
    out = {}
    for c in cols:
        v = tabs[y][c].to_numpy(zero_copy_only=False).astype(float)[idx]
        v[~ok] = np.nan
        out[c] = v
    return out, ok


res = []
for on in ('val', 'test'):
    res.append(f'\n## {on}\n')
    deep, early, lifts, period = [], [], [], []
    for tp in TYPES:
        d = load(tp, on)
        o, h, p, y = d['o'], d['h'], d['p'], d['y']
        thr = 1 - SHARE[tp]
        alarm = p >= thr
        W = d['W']
        Wv = np.where(W >= 0, W, 0)
        mx = np.where(W >= 0, p[Wv], -1).max(1)
        miss = mx < thr
        # (б) глубина пропуска: во сколько раз надо поднять долю тревожных часов, чтобы поймать
        need = (1 - mx[miss]) / SHARE[tp]
        deep.append(f'| {tp} | {miss.sum()} | {np.mean(need <= 1.5):.0%} | {np.mean(need <= 2):.0%} | '
                    f'{np.mean(need <= 4):.0%} | {np.mean(need > 10):.0%} | {np.median(need):.1f} |')
        # строки эпизодов: час до начала
        last = Wv[np.arange(len(W)), (W >= 0).cumsum(1).argmax(1)]
        eo, eh = o[last], h[last]
        f, ok = take(on, eo, eh, [f'next_{tp}', f'next_{tp}_prim', f'since_{tp}'] + CNT)
        prim = (f[f'next_{tp}_prim'] == f[f'next_{tp}']) & ok
        # контроль: часы тех же периодов без эпизода типа в 72 ч и без своего эпизода 7 суток
        ctl_i = rng.choice(len(o), size=min(60000, len(o)), replace=False)
        fc, okc = take(on, o[ctl_i], h[ctl_i], [f'next_{tp}', f'since_{tp}'] + CNT)
        cm = okc & (fc[f'next_{tp}'] > 72) & (fc[f'since_{tp}'] >= 168)
        for grp, g in (('пропущенные первичные', miss & prim), ('пойманные первичные', ~miss & prim)):
            if g.sum() < 8:
                continue
            rows = []
            for c in CNT:
                a = np.nanmean(f[c][g] > 0)
                b = np.nanmean(fc[c][cm] > 0)
                if a >= 0.10:
                    rows.append((a / max(b, 1e-3), c, a, b))
            rows.sort(reverse=True)
            top = ', '.join(f'`{c}` {a:.0%} против {b:.1%} (×{l:.0f})' for l, c, a, b in rows[:5])
            lifts.append(f'| {tp} | {grp} | {g.sum()} | {top or "—"} |')
        # (а) ранние ложные: свой эпизод через 24–72 ч после конца ложного сигнала против фона
        idx = np.flatnonzero(alarm)
        so, sh, sy = o[idx], h[idx], y[idx]
        start = np.r_[True, (so[1:] != so[:-1]) | (sh[1:] - sh[:-1] > GAP + 1)]
        run = np.cumsum(start) - 1
        nr = run[-1] + 1
        true = np.bincount(run, weights=sy, minlength=nr) > 0
        endh = np.zeros(nr, np.int64)
        np.maximum.at(endh, run, sh)
        fe, oke = take(on, so[start], endh, [f'next_{tp}', f'since_{tp}'])
        n1 = fe[f'next_{tp}'][~true]
        fs = np.nanmean((n1 > H) & (n1 <= 72))
        # фон: часы без тревоги, без эпизода в 24 ч, с таким же распределением «часы с прошлого»
        na = ~alarm & ~y
        bi = rng.choice(np.flatnonzero(na), size=min(60000, na.sum()), replace=False)
        fb, okb = take(on, o[bi], h[bi], [f'next_{tp}', f'since_{tp}'])
        sf = fe[f'since_{tp}'][~true]
        bins = np.array([0, 24, 72, 168, 720, 10**6])
        wb = np.histogram(sf[~np.isnan(sf)], bins)[0].astype(float)
        wb /= wb.sum()
        nb, sb = fb[f'next_{tp}'], fb[f'since_{tp}']
        base = 0.0
        for k in range(len(bins) - 1):
            m = okb & (sb >= bins[k]) & (sb < bins[k + 1])
            if m.any():
                base += wb[k] * np.mean((nb[m] > H) & (nb[m] <= 72))
        early.append(f'| {tp} | {(~true).sum()} | {fs:.0%} | {base:.0%} | {fs / max(base, 1e-3):.1f} |')
        # (г) периодичность повторов: интервал от прошлого эпизода у пропущенных повторов
        if tp in ('equipment', 'sensor', 'fire'):
            iv = f[f'since_{tp}'][miss & ~prim & ok] + 1
            iv = iv[(iv >= 20) & (iv < 168)]
            near = np.mean(np.abs(((iv + 12) % 24) - 12) <= 1) if len(iv) else np.nan
            ivc = f[f'since_{tp}'][~miss & ~prim & ok] + 1
            ivc = ivc[(ivc >= 20) & (ivc < 168)]
            nearc = np.mean(np.abs(((ivc + 12) % 24) - 12) <= 1) if len(ivc) else np.nan
            period.append(f'| {tp} | {len(iv)} | {near:.0%} | {len(ivc)} | {nearc:.0%} | 12% |')
        print(on, tp, file=sys.stderr, flush=True)
    res.append('### Глубина пропуска: во сколько раз поднять долю тревожных часов типа, чтобы эпизод поймался\n')
    res.append('| тип | пропущено | ≤ 1,5× | ≤ 2× | ≤ 4× | > 10× | медиана |')
    res.append('|---|---:|---:|---:|---:|---:|---:|')
    res += deep
    res.append('\n### Предвестники первичных эпизодов: доля с признаком > 0 в час до начала против контроля\n')
    res.append('| тип | группа | эпизодов | пять признаков с наибольшим подъёмом (не реже 10%) |')
    res.append('|---|---|---:|---|')
    res += lifts
    res.append('\n### «Ранние» ложные сигналы: свой эпизод через 24–72 ч после конца, против фона с тем же «часов с прошлого»\n')
    res.append('| тип | ложных сигналов | доля ранних | фон | во сколько раз |')
    res.append('|---|---:|---:|---:|---:|')
    res += early
    res.append('\n### Суточный ритм повторов: интервал 20–168 ч в пределах ±1 ч от кратного 24 ч (наугад — 12%)\n')
    res.append('| тип | пропущенных повторов | ±1 ч от 24k | пойманных повторов | ±1 ч от 24k | наугад |')
    res.append('|---|---:|---:|---:|---:|---:|')
    res += period
print('\n'.join(res))
