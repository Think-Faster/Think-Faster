"""Паттерны ложных сигналов и пропущенных эпизодов на итоговой конфигурации (раздел 44).

Пропущенный эпизод — ни одного часа тревоги в 24 ч до начала. Для каждого эпизода берётся строка
витрины в последний момент прогноза (час до начала) и сравнивается у пропущенных и пойманных:
первичный ли (такого же типа на объекте не было 7 суток), тишина в журнале объекта за 24 ч,
часы с прошлого эпизода того же типа, триггеры своего типа за 24 ч, эпизоды в коллекторе, другие
типы на объекте, подтверждён ли выездом.

Ложный сигнал (склейка 6 ч) раскладывается по причинам, по порядку, первая подошедшая:
хвост (свой эпизод меньше 7 суток назад), шумовой триггер своего типа в сигнале, эпизод другого
типа на объекте в сигнале или 24 ч после, свой эпизод через 24–72 ч после конца (промах горизонта),
свой эпизод в коллекторе на другом объекте, выезд бригады в коллекторе, без объяснения.
Для сравнения те же признаки у подтвердившихся сигналов.

    python patterns.py "$MIXT"
"""
import sys
import numpy as np
import pyarrow.parquet as pq

import config
from smooth import SHARE, load

H = 24
GAP = 6
TYPES = list(SHARE)


def feats():
    cols = ['object_id', 'h', 'events_24h', 'visit_24h', 'armed', 'hour']
    for t in TYPES:
        cols += [f'since_{t}' if t != 'gas' else 'since_gas', f'trig_{t}_24h', f'onset_{t}_24h',
                 f'coll_onset_{t}_24h', f'next_{t}', f'next_{t}_conf', f'next_{t}_prim']
    cols += ['noise_fire_24h', 'noise_flood_24h']
    parts = [pq.read_table(config.WORK / 'features' / f'{y}.parquet', columns=cols).to_pandas() for y in (2025, 2026)]
    import pandas as pd
    f = pd.concat(parts, ignore_index=True)
    f['key'] = f.object_id.astype(np.int64) * 10**7 + f.h.astype(np.int64)
    return f.sort_values('key').set_index('key')


F = feats()
KEYS = F.index.values


def rows(o, h):
    k = np.asarray(o, np.int64) * 10**7 + np.asarray(h, np.int64)
    i = np.minimum(np.searchsorted(KEYS, k), len(KEYS) - 1)
    ok = KEYS[i] == k
    return i, ok


def col(name, i, ok, fill=np.nan):
    v = F[name].values[i].astype(float)
    v[~ok] = fill
    return v


def pct(m):
    return f'{100 * np.nanmean(m):.0f}%' if len(m) else '—'


out = []
for on in ('val', 'test'):
    out.append(f'\n## {on}\n')
    miss_rows, fs_rows, cls_tot = [], [], {}
    for tp in TYPES:
        d = load(tp, on)
        o, h, p = d['o'], d['h'], d['p']
        alarm = p >= np.quantile(p, 1 - SHARE[tp])
        # эпизоды: пойман ли
        W = d['W']
        caught = (alarm[np.where(W >= 0, W, 0)] & (W >= 0)).any(1)
        eo = o[np.where(W >= 0, W, 0).max(1)]
        # начало эпизода — из eps по тем же эпизодам
        eps = d['eps']
        key = o.astype(np.int64) * 10**7 + h
        wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
        pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
        keep = (key[pos] == wk).any(1)
        eps = eps[keep]
        t0 = eps[:, 1]
        i, ok = rows(eps[:, 0], t0 - 1)
        nxt = col(f'next_{tp}', i, ok)
        prim = col(f'next_{tp}_prim', i, ok) == nxt
        conf = col(f'next_{tp}_conf', i, ok) == nxt
        quiet = col('events_24h', i, ok) == 0
        since = col(f'since_{tp}', i, ok)
        trig = col(f'trig_{tp}_24h', i, ok) > 0
        coll = (col(f'coll_onset_{tp}_24h', i, ok) - col(f'onset_{tp}_24h', i, ok)) > 0
        other = sum(col(f'onset_{t}_24h', i, ok) for t in TYPES if t != tp) > 0
        hr = (col('hour', i, ok) + 1) % 24
        night = (hr < 7) | (hr >= 22)
        for name, m in (('пропущено', ~caught), ('поймано', caught)):
            miss_rows.append(f'| {tp} | {name} | {m.sum()} | {pct(prim[m])} | {pct(quiet[m])} | '
                             f'{np.nanmedian(since[m]) if m.any() else 0:.0f} | {pct(trig[m])} | {pct(coll[m])} | '
                             f'{pct(other[m])} | {pct(conf[m])} | {pct(night[m])} |')
        # классы пропущенных
        m = ~caught
        cls = np.select([~prim & m, prim & quiet & m, prim & ~quiet & m], ['повтор', 'первичный из тишины', 'первичный, журнал не пуст'], '')
        for c in ('повтор', 'первичный из тишины', 'первичный, журнал не пуст'):
            cls_tot.setdefault(tp, {})[c] = int((cls == c).sum())
        # сигналы
        idx = np.flatnonzero(alarm)
        so, sh, sy = o[idx], h[idx], d['y'][idx]
        start = np.r_[True, (so[1:] != so[:-1]) | (sh[1:] - sh[:-1] > GAP + 1)]
        run = np.cumsum(start) - 1
        nr = run[-1] + 1
        true = np.bincount(run, weights=sy, minlength=nr) > 0
        s_o = so[start]
        s_h0 = sh[start]
        s_h1 = np.zeros(nr, np.int64)
        np.maximum.at(s_h1, run, sh)
        L = s_h1 - s_h0 + 1
        i0, ok0 = rows(s_o, s_h0)
        i1, ok1 = rows(s_o, s_h1)
        tail = col(f'since_{tp}', i0, ok0) < 168
        noise = col(f'trig_{tp}_24h', i1, ok1) > 0
        if tp in ('fire', 'flood'):
            noise |= col(f'noise_{tp}_24h', i1, ok1) > 0
        oth = np.zeros(nr, bool)
        for t in TYPES:
            if t != tp:
                oth |= col(f'next_{t}', i0, ok0) <= L + H
        n1 = col(f'next_{tp}', i1, ok1)
        near = (n1 > H) & (n1 <= 72)
        i2, ok2 = rows(s_o, s_h1 + H)
        collx = (col(f'coll_onset_{tp}_24h', i2, ok2) - col(f'onset_{tp}_24h', i2, ok2)) > 0
        visit = col('visit_24h', i1, ok1) > 0
        causes = [('хвост своего эпизода < 7 сут', tail), ('шум / триггер своего типа без эпизода', noise),
                  ('эпизод другого типа на объекте', oth), ('свой эпизод через 24–72 ч', near),
                  ('свой эпизод в коллекторе', collx), ('выезд бригады в коллекторе', visit)]
        lab = np.full(nr, 'без объяснения', object)
        for nm, m in causes[::-1]:
            lab[m] = nm
        for grp, g in (('ложные', ~true), ('подтвердившиеся', true)):
            parts = ' | '.join(pct(m[g]) for _, m in causes)
            share = ' | '.join(f'{(lab[g] == nm).sum()}' for nm, _ in causes) + f' | {(lab[g] == "без объяснения").sum()}'
            fs_rows.append(f'| {tp} | {grp} | {g.sum()} | {np.median(L[g]):.0f} | {parts} | {share} |')
        print(on, tp, 'ok', file=sys.stderr, flush=True)
    out.append('### Эпизоды: пропущенные против пойманных (признаки в час до начала)\n')
    out.append('| тип | | эпизодов | первичный | тишина 24 ч | ч с прошлого (медиана) | триггер своего типа 24 ч | эпизод в коллекторе 24 ч | другой тип на объекте 24 ч | подтверждён выездом | ночь 22–7 |')
    out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    out += miss_rows
    out.append('\n### Пропущенные по классам\n')
    out.append('| тип | повтор (< 7 сут) | первичный из тишины | первичный, журнал не пуст |')
    out.append('|---|---:|---:|---:|')
    for tp, c in cls_tot.items():
        out.append(f'| {tp} | ' + ' | '.join(str(v) for v in c.values()) + ' |')
    out.append('\n### Сигналы: признаки (доля, не взаимоисключающие) и причина по порядку (число)\n')
    out.append('| тип | | сигналов | длина, ч (медиана) | хвост < 7 сут | шум/триггер | другой тип | свой через 24–72 ч | свой в коллекторе | выезд | '
               'хвост | шум | другой тип | 24–72 ч | коллектор | выезд | без объяснения |')
    out.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    out += fs_rows
print('\n'.join(out))
