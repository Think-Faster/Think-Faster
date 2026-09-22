"""Предел длины тревоги: после C часов подряд тревога уходит в статус «объект в зоне риска».

Часы статуса не считаются ложной тревогой и эпизодов не ловят. Бюджет ложных часов на проверке
раздаётся жадно по вогнутым оболочкам, как в `curve.split`, с минимумом полноты 30% (пожару и газу
60%); доли часов под тревогой переносятся на тест. Кроме раздачи по пойманным считаются доли
раздела 38 без пересчёта (`s38`) и раздача по свежим поимкам (`fresh`, раздел 40). Итог — раздел 41
аналитики: предел не ставится.

    python runcap.py "$MIX5" 80000
"""
import sys
import numpy as np
import metrics
import operating
import config

RUN = sys.argv[1]
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
CAPS = [None, 336, 240, 168, 120]
S38 = {'fire': .030, 'gas': .026, 'flood': .022, 'equipment': .097, 'sensor': .026, 'intrusion': .005}
H, GAP = 24, 6
FLOOR = {'fire': .6, 'gas': .6}
SHARES = np.unique(np.round(np.geomspace(0.0005, 0.5, 140), 6))


def prep(tp, on):
    o, h, n, p = operating.load_mix(RUN, on, operating.years(RUN, on), tp, 'xgb', '')
    order = np.lexsort((h, o))
    o, h, n, p = o[order], h[order], n[order], p[order]
    y = n <= H
    key = o.astype(np.int64) * 10**7 + h
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]   # от раннего к позднему
    pos = np.searchsorted(key, wk)
    pos = np.minimum(pos, len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    W = W[(W >= 0).any(1)]
    return dict(o=o, h=h, y=y, p=p, W=W)


def run_pos(d, alarm, gap=GAP):
    """Час от начала сигнала (склейка GAP) для каждой строки тревоги; вне тревоги -1."""
    o, h = d['o'], d['h']
    idx = np.flatnonzero(alarm)
    out = np.full(len(o), -1, np.int64)
    if not len(idx):
        return out
    oo, hh = o[idx], h[idx]
    start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > gap + 1)]
    first = hh[start][np.cumsum(start) - 1]
    out[idx] = hh - first
    return out


def measure(d, share, cap):
    p, y, W = d['p'], d['y'], d['W']
    t = float(np.quantile(p, 1 - share))
    alarm = p >= t
    pos = run_pos(d, alarm)
    cnt = alarm & (pos < cap) if cap else alarm
    Wv = np.where(W >= 0, W, 0)
    hit = cnt[Wv] & (W >= 0)
    caught = hit.any(1)
    # «постоянная» поимка: к первому часу тревоги в окне сигнал уже горел, вместе с упреждением ≥ 7 сут
    first = hit.argmax(1)
    lead = H - first
    pos0 = run_pos(d, cnt, 0)   # как в metrics.evaluate: сплошная серия без склейки
    stand = caught & (pos0[Wv[np.arange(len(W)), first]] + lead >= metrics.RUN_CAP)
    status = alarm & ~cnt
    in_status = ((status[Wv] & (W >= 0)).any(1)) & ~caught
    sig, true = metrics.signals(d['o'], d['h'], y.astype(np.int8), cnt, GAP)
    return dict(fh=int((cnt & ~y).sum()), caught=int(caught.sum()), stand=int(stand.sum()),
                status_h=int(status.sum()), in_status=int(in_status.sum()), eps=len(W),
                false_sig=sig - true)


def hull(pts):
    pts = sorted(pts)
    rise, best = [], -1
    for c, k, s in pts:
        if k <= best:
            continue
        if rise and rise[-1][0] == c:
            rise.pop()
        rise.append((c, k, s))
        best = k
    out = [(0, 0, None)]
    for pt in rise:
        while len(out) >= 2:
            (c1, k1, _), (c2, k2, _) = out[-2], out[-1]
            if (k2 - k1) * (pt[0] - c1) <= (pt[1] - k1) * (c2 - c1):
                out.pop()
            else:
                break
        out.append(pt)
    return out


def greedy(env):
    idx, spent = {}, 0
    for tp, e in env.items():
        need = FLOOR.get(tp, .3) * e[-1][1]
        i = next(i for i, (c, k, _) in enumerate(e) if k >= need)
        idx[tp], spent = i, spent + e[i][0]
    while True:
        best, gain = None, 0.0
        for tp, e in env.items():
            i = idx[tp]
            if i + 1 >= len(e):
                continue
            dc, dk = e[i + 1][0] - e[i][0], e[i + 1][1] - e[i][1]
            if dc > 0 and spent + dc <= BUDGET and dk / dc > gain:
                best, gain = tp, dk / dc
        if best is None:
            break
        spent += env[best][idx[best] + 1][0] - env[best][idx[best]][0]
        idx[best] += 1
    return {tp: env[tp][idx[tp]][2] for tp in env}


data = {tp: (prep(tp, 'val'), prep(tp, 'test')) for tp in config.TYPES}
print(f'бюджет {BUDGET} ложных часов на проверке, склейка {GAP} ч; тест 2026', flush=True)
print('| предел сигнала | цель раздачи | поймано | из них тревогой ≥ 7 сут | свежих | ложных часов | '
      'ложных сигналов | часов статуса | эпизодов только под статусом | доли по типам |')
print('|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
for cap in CAPS:
    val = {tp: {s: measure(data[tp][0], s, cap) for s in SHARES} for tp in config.TYPES}
    for goal in ('s38', 'caught', 'fresh'):
        f = (lambda m: m['caught']) if goal == 'caught' else (lambda m: m['caught'] - m['stand'])
        env = {tp: hull([(m['fh'], f(m), s) for s, m in val[tp].items()]) for tp in config.TYPES}
        pick = dict(S38) if goal == 's38' else greedy(env)
        tot = dict(caught=0, stand=0, fh=0, false_sig=0, status_h=0, in_status=0)
        per = []
        for tp, s in pick.items():
            m = measure(data[tp][1], s, cap) if s else dict.fromkeys(tot, 0)
            for k in tot:
                tot[k] += m[k]
            per.append(f'{tp[:4]} {0 if s is None else s:.3f}')
            print(f'  {cap} {goal} {tp}: share {s} test {m}', file=sys.stderr, flush=True)
        print(f"| {cap or '—'} | {goal} | {tot['caught']} | {tot['stand']} | {tot['caught'] - tot['stand']} | "
              f"{tot['fh']} | {tot['false_sig']} | {tot['status_h']} | {tot['in_status']} | {', '.join(per)} |",
              flush=True)
