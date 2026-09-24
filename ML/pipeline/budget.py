"""Цена ложных часов: сколько поимок теряется за каждые −10% ложных часов, по типам.

Рабочая точка — доли типов как есть и правило раздела 44 (после отклонения «маловероятно» до
события, k = 0,2). Доля типа умножается на общий множитель; по сетке множителей строится кривая,
и на ней ищутся точки с ложными часами 120%…50% от рабочей.

    python budget.py "$MIXT"
"""
import sys
import numpy as np
from smooth import SHARE, load, measure
from reject import simulate

R, N, K = 1, 10**6, 0.2
F = np.geomspace(0.2, 1.6, 22)
LEVELS = [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
COLS = ('fh', 'false_sig', 'caught', 'fresh')

for on in ('val', 'test'):
    tot = np.zeros((len(F), 4))
    rows = {}
    for tp, s in SHARE.items():
        d = load(tp, on)
        by = {}
        for a, b in d['eps']:
            by.setdefault(a, []).append(b)
        ons = {a: np.array(sorted(b), np.int64) for a, b in by.items()}
        pts = np.zeros((len(F), 4))
        for i, f in enumerate(F):
            m = measure(d, simulate(d, ons, min(s * f, 0.5), R, N, K))
            pts[i] = [m[c] for c in COLS]
        base = measure(d, simulate(d, ons, s, R, N, K))
        rows[tp] = (pts, np.array([base[c] for c in COLS]), len(d['eps']))
        tot += pts
        print(on, tp, 'готово', file=sys.stderr, flush=True)
    print(f'\n### {on}\n')
    print('Клетка — поймано / свежих / ложных сигналов и в скобках изменение свежих к рабочей точке.\n')
    print('| тип | эпизодов | рабочая точка: ложных ч / поймано / свежих / ложных сигн. | '
          + ' | '.join(f'{int(round(l * 100))}% ложных ч' for l in LEVELS if l != 1.0) + ' |')
    print('|---|---|---|' + '---|' * (len(LEVELS) - 1))
    for tp, (pts, b, ne) in list(rows.items()) + [('все (одна доля)', (tot, None, None))]:
        if b is None:
            b = np.array([np.interp(1.0, F, tot[:, j]) for j in range(4)])
            ne = sum(r[2] for r in rows.values())
        o = np.argsort(pts[:, 0])
        cells = []
        for l in LEVELS:
            if l == 1.0:
                continue
            v = [np.interp(b[0] * l, pts[o, 0], pts[o, j]) for j in (2, 3, 1)]
            out = b[0] * l > pts[o, 0].max() or b[0] * l < pts[o, 0].min()
            cells.append('—' if out else f'{v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f} ({(v[1] / b[3] - 1) * 100:+.0f}%)')
        print(f'| {tp} | {ne} | {b[0]:.0f} / {b[2]:.0f} / {b[3]:.0f} / {b[1]:.0f} | ' + ' | '.join(cells) + ' |', flush=True)
