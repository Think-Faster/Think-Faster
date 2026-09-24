"""Накопление уверенности по часам: сглаживание оценки и «порог держится k часов подряд».

Для каждого типа оценка смеси переводится в долю внутри периода и преобразуется причинно (только
прошлые часы того же объекта):
  raw      — как сейчас;
  mean<k>  — среднее за последние k часов;
  ema<k>   — экспоненциальное сглаживание с полураспадом k часов;
  min<k>   — минимум за k часов: тревога только если порог держится k часов подряд;
  max<k>   — максимум за k часов (контроль в обратную сторону: тревога держится k часов после всплеска).
Порог каждого варианта подбирается так, чтобы ложных часов было ровно столько же, сколько у raw
при доле раздела 38 (или своей из argv[2]); при них считаются поймано, свежих, сигналы и ложные
сигналы со склейкой 6 ч.

    python smooth.py "$MIXT"            # при равных ложных часах, по типам
    python smooth.py "$MIXT" --front    # кривые: при равных ложных сигналах, суммы по типам

Раздел 43 аналитики: сглаживание перепроверено на итоговой конфигурации и снова не берётся.
"""
import sys
import numpy as np
from scipy.signal import lfilter
import metrics
import operating

MIX = sys.argv[1] if len(sys.argv) > 1 else ''
FRONT = '--front' in sys.argv
SHARE = {'fire': .030, 'gas': .026, 'flood': .018, 'equipment': .072, 'sensor': .009, 'intrusion': .005}
H, GAP = 24, 6
VARIANTS = ['raw', 'mean3', 'mean6', 'mean12', 'ema3', 'ema6', 'ema12', 'min2', 'min3', 'min4', 'max3']


def shifted(p, o, j):
    q = np.roll(p, j)
    same = np.roll(o, j) == o
    same[:j] = False
    return np.where(same, q, np.nan)


def transform(p, o, v):
    if v == 'raw':
        return p
    k = int(''.join(c for c in v if c.isdigit()))
    if v.startswith('ema'):
        a = 1 - 0.5 ** (1 / k)
        out = np.empty_like(p)
        starts = np.r_[0, np.flatnonzero(o[1:] != o[:-1]) + 1, len(p)]
        for s, e in zip(starts[:-1], starts[1:]):
            zi = [(1 - a) * p[s]]
            out[s:e] = lfilter([a], [1, -(1 - a)], p[s:e], zi=zi)[0]
        return out
    st = np.stack([p] + [shifted(p, o, j) for j in range(1, k)])
    if v.startswith('mean'):
        return np.nanmean(st, 0)
    if v.startswith('min'):
        return np.nanmin(st, 0)
    return np.nanmax(st, 0)


def load(tp, on):
    yr = operating.years(MIX, on)
    o, h, n, p = operating.load_mix(MIX, on, yr, tp, 'xgb', '')
    order = np.lexsort((h, o))
    o, h, n, p = o[order], h[order], n[order], p[order]
    p = p.argsort().argsort() / len(p)
    key = o.astype(np.int64) * 10**7 + h
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    return dict(o=o, h=h, y=n <= H, p=p, W=W[(W >= 0).any(1)], eps=eps)


def measure(d, alarm):
    W = d['W']
    Wv = np.where(W >= 0, W, 0)
    hit = alarm[Wv] & (W >= 0)
    caught = hit.any(1)
    first = hit.argmax(1)
    idx = np.flatnonzero(alarm)
    pos = np.full(len(alarm), -1, np.int64)
    sig = true = 0
    if len(idx):
        oo, hh = d['o'][idx], d['h'][idx]
        start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
        pos[idx] = hh - hh[start][np.cumsum(start) - 1]
        sig, true = metrics.signals(d['o'], d['h'], d['y'], alarm, GAP)
    stand = caught & (pos[Wv[np.arange(len(W)), first]] + (H - first) >= metrics.RUN_CAP)
    # упреждение: за сколько часов до начала эпизода зажглась первая тревога в окне H
    lead = (H - first)[caught]
    return dict(fh=int((alarm & ~d['y']).sum()), caught=int(caught.sum()), fresh=int((caught & ~stand).sum()),
                sig=sig, false_sig=sig - true, lead=float(np.median(lead)) if len(lead) else 0.0)


def at_fh(d, q, target):
    """Порог, при котором ложных часов столько же, сколько target (бисекция по доле)."""
    lo, hi = 1e-5, 0.5
    for _ in range(30):
        mid = (lo + hi) / 2
        m = measure(d, q >= np.quantile(q, 1 - mid))
        if m['fh'] < target:
            lo = mid
        else:
            hi = mid
    return measure(d, q >= np.quantile(q, 1 - lo))


def front():
    FV = ['raw', 'max3', 'max6', 'mean3', 'ema3', 'ema6', 'min2']
    F = np.geomspace(0.25, 4, 17)
    for on in ('val', 'test'):
        data = {tp: load(tp, on) for tp in SHARE}
        curves = {}
        for v in FV:
            pts = np.zeros((len(F), 4))
            for tp, s in SHARE.items():
                d = data[tp]
                q = transform(d['p'], d['o'], v)
                for i, f in enumerate(F):
                    m = measure(d, q >= np.quantile(q, 1 - min(s * f, 0.5)))
                    pts[i] += [m['false_sig'], m['caught'], m['fresh'], m['fh']]
            curves[v] = pts
            print(f'{on} {v}: ' + ' '.join(f'[f={f:.2f} лс={a:.0f} п={b:.0f} св={c:.0f} лч={e:.0f}]'
                                           for f, (a, b, c, e) in zip(F, pts)), file=sys.stderr, flush=True)
        ref = curves['raw'][8]            # f = 1: доли как есть
        print(f'\n### {on}: при ложных сигналах raw ({ref[0]:.0f}) и при его поимках ({ref[1]:.0f})\n')
        print('| вариант | поймано при равных ложных сигналах | свежих | ложных ч | '
              'ложных сигналов при равных поимках | ложных ч |')
        print('|---|---:|---:|---:|---:|---:|')
        for v, pts in curves.items():
            o = np.argsort(pts[:, 0])
            a = [np.interp(ref[0], pts[o, 0], pts[o, j]) for j in (1, 2, 3)]
            o2 = np.argsort(pts[:, 1])
            b = [np.interp(ref[1], pts[o2, 1], pts[o2, j]) for j in (0, 3)]
            print(f'| {v} | {a[0]:.0f} | {a[1]:.0f} | {a[2]:.0f} | {b[0]:.0f} | {b[1]:.0f} |', flush=True)


if __name__ == '__main__' and FRONT:
    front()
elif __name__ == '__main__':
    print('| тип | период | вариант | поймано | свежих | ложных ч | сигналов | ложных сигналов | медиана упреждения, ч |')
    print('|---|---|---|---:|---:|---:|---:|---:|---:|')
    tot = {}
    for tp, s in SHARE.items():
        for on in ('val', 'test'):
            d = load(tp, on)
            base = measure(d, d['p'] >= np.quantile(d['p'], 1 - s))
            for v in VARIANTS:
                q = transform(d['p'], d['o'], v)
                m = base if v == 'raw' else at_fh(d, q, base['fh'])
                acc = tot.setdefault((on, v), np.zeros(5))
                acc += [m['caught'], m['fresh'], m['fh'], m['sig'], m['false_sig']]
                print(f"| {tp} | {on} | {v} | {m['caught']} | {m['fresh']} | {m['fh']} | {m['sig']} | {m['false_sig']} | "
                      f"{m['lead']:.0f} |", flush=True)
    for (on, v), a in sorted(tot.items()):
        print(f'| **итого** | {on} | {v} | {a[0]:.0f} | {a[1]:.0f} | {a[2]:.0f} | {a[3]:.0f} | {a[4]:.0f} | |')
