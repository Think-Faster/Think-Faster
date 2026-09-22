"""Смесь бустинга с TCN по типам: при долях часов раздела 38 и при равных ложных часах.

argv[1] — MIX5, argv[2] — семейство(а) TCN через запятую (main_h24/tcn,main_h24/tcn_s1,...).
Для каждого типа и веса TCN w (доля TCN в среднем рангов) печатается на проверке и тесте:
поймано, свежих, ложных часов при доле раздела 38; и поймано при ложных часах базы (интерполяция
по сетке долей) — это честное сравнение при равной цене. Раздел 42 аналитики.

    python tcnmix.py "$MIX5" main_h24/tcn,main_h24/tcn_s1,main_h24/tcn_s2,main_h24/tcn_s3,main_h24/tcn_s4
"""
import sys
import numpy as np
import metrics
import operating
import config

MIX = sys.argv[1]
TCN = sys.argv[2].split(',')
TYPES = sys.argv[3].split(',') if len(sys.argv) > 3 else config.TYPES
H = 24
S38 = {'fire': .030, 'gas': .026, 'flood': .022, 'equipment': .097, 'sensor': .026, 'intrusion': .005}
WEIGHTS = [0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 1]
SHARES = np.unique(np.round(np.geomspace(0.001, 0.3, 90), 6))


def pct(x):
    return x.argsort().argsort() / len(x)


def load(tp, on):
    table = dict(part.split('~', 1) for part in MIX.split(';') if part)
    yr = operating.years(MIX, on)
    o, h, n, p = operating.load_mix(table[tp], on, yr, tp, 'xgb', '')
    q = None
    for t in TCN:
        o2, h2, n2, p2 = operating.load_mix(t, on, yr, tp, 'xgb', '')
        assert np.array_equal(o, o2) and np.array_equal(h, h2)
        q = pct(p2) if q is None else q + pct(p2)
    order = np.lexsort((h, o))
    o, h, n = o[order], h[order], n[order]
    pb, pt = pct(p)[order], pct(q)[order]
    key = o.astype(np.int64) * 10**7 + h
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    keep = (W >= 0).any(1)
    return dict(o=o, h=h, y=n <= H, pb=pb, pt=pt, W=W[keep], eobj=eps[keep, 0])


def run_pos(o, h, alarm):
    idx = np.flatnonzero(alarm)
    out = np.full(len(o), -1, np.int64)
    if not len(idx):
        return out
    oo, hh = o[idx], h[idx]
    start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
    out[idx] = hh - hh[start][np.cumsum(start) - 1]
    return out


def measure(d, w, share):
    p = (1 - w) * d['pb'] + w * d['pt']
    alarm = p >= np.quantile(p, 1 - share)
    W = d['W']
    Wv = np.where(W >= 0, W, 0)
    hit = alarm[Wv] & (W >= 0)
    caught = hit.any(1)
    first = hit.argmax(1)
    pos = run_pos(d['o'], d['h'], alarm)
    stand = caught & (pos[Wv[np.arange(len(W)), first]] + (H - first) >= metrics.RUN_CAP)
    return int((alarm & ~d['y']).sum()), int(caught.sum()), int(stand.sum()), caught


def at_cost(d, w, cost):
    """Поймано при заданных ложных часах: линейно между соседними долями сетки."""
    pts = [measure(d, w, s)[:2] for s in SHARES]
    c = np.array([x for x, _ in pts], float)
    k = np.array([y for _, y in pts], float)
    k = np.maximum.accumulate(k)
    return float(np.interp(cost, c, k))


print('| тип | вес TCN | проверка: поймано / свежих / ложных ч | тест: поймано / свежих / ложных ч | '
      'проверка: поймано при ложных ч базы | тест: поймано при ложных ч базы |')
print('|---|---:|---|---|---:|---:|')
for tp in TYPES:
    dv, dt = load(tp, 'val'), load(tp, 'test')
    s = S38[tp]
    base = {nm: measure(d, 0, s) for nm, d in (('v', dv), ('t', dt))}
    for w in WEIGHTS:
        mv, mt = measure(dv, w, s), measure(dt, w, s)
        kv, kt = at_cost(dv, w, base['v'][0]), at_cost(dt, w, base['t'][0])
        print(f'| {tp} | {w:.2f} | {mv[1]} / {mv[1] - mv[2]} / {mv[0]} | {mt[1]} / {mt[1] - mt[2]} / {mt[0]} | '
              f'{kv:.0f} | {kt:.0f} |', flush=True)
        if tp == 'intrusion':
            objs, cnt = np.unique(dt['eobj'][mt[3]], return_counts=True)
            top = sorted(zip(cnt, objs), reverse=True)[:3]
            print(f'  intrusion w={w:.2f}: пойманные на тесте по объектам {top}', file=sys.stderr, flush=True)
