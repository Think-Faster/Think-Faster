"""Смесь рангов CatBoost ×5 (A′) и сети TCN ×5 (pa_all, данные 2022–2023, входы A′) по типам.

Вес сети выбирается на проверке 2025, тест 2026 — подтверждение. Доля часов — рабочая (config.shares()).
С ключом --versions — версии оборудования на A′ (тест 2026, доля 0,059): v1 как в выгрузке,
v1 с пятью зёрнами сети и с сетью на A′, v2–v4, смесь v2 + v3. Раздел 65 analytics.md.
Меры как в sticky.point: поймано, свежих (тревога стоит < RUN_CAP ч), ложных сигналов (склейка 6 ч), ложных ч.

    TF_WORK=work_pa python netblend.py fire,sensor,flood,intrusion
    TF_WORK=work_pa python netblend.py --versions
"""
import sys
import numpy as np
import config
import metrics
import operating

H = 24
S = config.shares()
WS = (0, 0.1, 0.25, 0.4, 0.5)
ROLL = config.WORK / 'roll'
pct = lambda q: q.argsort().argsort() / len(q)


def rows(tp, on, yy):
    acc = 0
    for s in ['', '_s1', '_s2', '_s3', '_s4']:
        run = ('main_h24' if tp == 'gas' else 'stfx_main_h24') + s
        o, h, n, p = operating.split(run, on, yy, tp, 'cat')
        acc = acc + pct(p)
    net = sum(pct(np.load(ROLL / f'pa_all_s{s}_2024-01-01_{tp}{"_val" if on == "val" else ""}.npy'))
              for s in range(5))
    assert len(net) == len(o)
    order = np.lexsort((h, o))
    o, h, n, cat, net = o[order], h[order], n[order], pct(acc[order]), pct(net[order])
    key = o.astype(np.int64) * 10**7 + h
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    return dict(o=o, h=h, y=n <= H, W=W[(W >= 0).any(1)]), cat, net


def point(d, p, share):
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
        st = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
        pos[idx] = hh - hh[st][np.cumsum(st) - 1]
    stand = caught & (pos[Wv[np.arange(len(W)), first]] + (H - first) >= metrics.RUN_CAP)
    tot, true = metrics.signals(d['o'], d['h'], d['y'], alarm, gap=6)
    return int(caught.sum()), int((caught & ~stand).sum()), tot - true, int((alarm & ~d['y']).sum())


def blends(TYPES):
    print('| тип | доля | вес сети | проверка 2025: поймано / свежих / ложных сигн. / ложных ч | тест 2026: то же |')
    print('|---|---:|---:|---|---|')
    for tp in TYPES:
        res = {}
        for on, yy in (('val', [2025]), ('test', [2026])):
            d, cat, net = rows(tp, on, yy)
            res[on] = {w: point(d, (1 - w) * cat + w * net, S[tp]) for w in WS}
        for w in WS:
            v, t = res['val'][w], res['test'][w]
            print(f'| {tp} | {S[tp]} | {w} | {v[0]} / {v[1]} / {v[2]} / {v[3]} | {t[0]} / {t[1]} / {t[2]} / {t[3]} |', flush=True)

    # при равных ложных сигналах: доля каждой смеси — наибольшая из сетки, при которой ложных сигналов не больше,
    # чем у бустинга (вес 0) при рабочей доле
    print('\n| тип | вес сети | проверка: поймано / свежих при ложных ≤ бустинга (доля) | тест: то же |')
    print('|---|---:|---|---|')
    for tp in TYPES:
        out = {}
        for on, yy in (('val', [2025]), ('test', [2026])):
            d, cat, net = rows(tp, on, yy)
            cap = point(d, cat, S[tp])[2]
            for w in WS:
                p = (1 - w) * cat + w * net
                best = None
                for sh in S[tp] * np.geomspace(0.6, 1.6, 21):
                    r = point(d, p, sh)
                    if r[2] <= cap and (best is None or r[1] > best[1]):
                        best = (r[0], r[1], sh)
                out[(on, w)] = best
        for w in WS:
            v, t = out[('val', w)], out[('test', w)]
            f = lambda x: '—' if x is None else f'{x[0]} / {x[1]} ({x[2]:.4f})'
            print(f'| {tp} | {w} | {f(v)} | {f(t)} |', flush=True)


def versions():
    tp, sh = 'equipment', S['equipment']
    d, cat, _ = rows(tp, 'test', [2026])
    o, h, *_ = operating.split('stfx_main_h24', 'test', [2026], tp, 'cat')
    order = np.lexsort((h, o))
    net = lambda tag, seeds: pct(sum(pct(np.load(ROLL / f'{tag}_s{s}_{"2024-01-01" if "all" in tag else "2026-07-01"}_{tp}.npy')[order])
                                       for s in seeds))
    v2, v3, v4 = net('old_all', range(5)), net('old_prodgap730', range(5)), net('old_prod', range(5))
    fams = {'CatBoost ×5': cat, 'v1: 0,75 CatBoost + 0,25 сеть ×3': 0.75 * cat + 0.25 * net('old_all', range(3)),
            'v1 с сетью ×5': 0.75 * cat + 0.25 * v2, 'v1 с сетью A′ ×5': 0.75 * cat + 0.25 * net('pa_all', range(5)),
            'v2': v2, 'v3': v3, 'v4': v4, 'v2 + v3 поровну': 0.5 * v2 + 0.5 * v3}
    print('\n| вариант | поймано / свежих | ложных сигн. | ложных ч | при ложных ≤ v1: поймано / свежих (доля) |')
    print('|---|---|---:|---:|---|')
    cap = point(d, fams['v1: 0,75 CatBoost + 0,25 сеть ×3'], sh)[2]
    for k, p in fams.items():
        r = point(d, p, sh)
        best = None
        for s in sh * np.geomspace(0.4, 1.6, 25):
            q = point(d, p, s)
            if q[2] <= cap and (best is None or q[1] > best[1]):
                best = (q[0], q[1], s)
        print(f'| {k} | {r[0]} / {r[1]} | {r[2]} | {r[3]} | {best[0]} / {best[1]} ({best[2]:.3f}) |', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--versions']:
        versions()
    else:
        blends(sys.argv[1].split(',') if len(sys.argv) > 1 else config.TYPES)
