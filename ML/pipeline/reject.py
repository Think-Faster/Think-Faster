"""Отклонение диспетчером: после отказа от прогноза объект по этому типу N часов «маловероятен».

Симуляция в реальном времени, без знания будущего. Сигнал (склейка 6 ч) начинается в t0. Диспетчер
проверяет объект через r часов: если к t0 + r на объекте этого типа ничего не случилось, он
отклоняет прогноз. Дальше до t0 + r + N тревога по этому объекту и типу горит только при оценке из
верхней доли share·k (k = 0 — не горит вовсе). Настоящий эпизод снимает режим «маловероятно» сразу.
`clear` вместо N: отклонение держится, пока тревога не сойдёт сама (пауза длиннее склейки), — как
«отложить до возврата в норму» в ISA 18.2; новый подъём оценки после этого снова тревожит.
Контроль `oracle`: отклоняются только сигналы, которые и правда окажутся ложными (диспетчер
безошибочен) — верхняя граница выигрыша.

Сравнение — кривые по общему множителю долей, суммы по шести типам: при равных ложных часах и при
равных ложных сигналах raw.

    python reject.py "$MIXT"
"""
import sys
import numpy as np
from smooth import SHARE, load, measure

GAP = 6
F = np.geomspace(0.35, 2.83, 7)
VARIANTS = [('raw', 0, 0, 0.0), ('r1_N24', 1, 24, 0.0), ('r1_N72', 1, 72, 0.0), ('r3_N72', 3, 72, 0.0),
            ('oracle_N72', -1, 72, 0.0), ('r1_clear', 1, -1, 0.0), ('r3_clear', 3, -1, 0.0), ('r6_clear', 6, -1, 0.0)]
for _N, _nm in ((168, 'N168'), (10**6, 'event')):
    for _k in (0.0, 0.1, 0.2, 0.35, 0.5):
        VARIANTS.append((f'r1_{_nm}_k{_k}', 1, _N, _k))


def simulate(d, ons_by, share, r, N, k):
    p = d['p']
    alarm = p >= np.quantile(p, 1 - share)
    if r == 0:
        return alarm
    thr_mute = np.quantile(p, 1 - share * k) if k > 0 else 2.0
    out = alarm.copy()
    o, h = d['o'], d['h']
    cur = None
    for i in np.flatnonzero(alarm):
        ob, t = o[i], h[i]
        if ob != cur:
            cur, mute_from, mute_until, last_on, sig_start, checked = ob, 0, -1, -10**9, None, True
            last_any = -10**9
            ons = ons_by.get(ob, np.empty(0, np.int64))
        if N < 0 and t - last_any > GAP + 1:
            mute_until = -1                              # тревога сошла сама — отклонение снято
        last_any = t
        if t < mute_until:
            j = np.searchsorted(ons, t, side='right') - 1
            if j >= 0 and ons[j] >= mute_from:        # настоящий эпизод снимает «маловероятно»
                mute_until = -1
            elif p[i] < thr_mute:
                out[i] = False
                continue
        if sig_start is None or t - last_on > GAP + 1:
            sig_start, checked = t, False
        last_on = t
        if checked:
            continue
        j = np.searchsorted(ons, sig_start, side='left')
        if r < 0:
            # oracle: отклоняется, только если за время молчания и горизонт ничего не случится
            checked = True
            if not (j < len(ons) and ons[j] <= sig_start + N + 24):
                mute_from, mute_until = t, (t + N if N > 0 else 10**9)
        elif t >= sig_start + r:
            checked = True
            if not (j < len(ons) and ons[j] <= t):
                mute_from, mute_until = t, (t + N if N > 0 else 10**9)
    return out


ALL = {}
for on in ('val', 'test'):
    data, obo = {}, {}
    for tp in SHARE:
        d = load(tp, on)
        data[tp] = d
        by = {}
        for a, b in d['eps']:
            by.setdefault(a, []).append(b)
        obo[tp] = {a: np.array(sorted(b), np.int64) for a, b in by.items()}
    curves = {}
    for name, r, N, k in VARIANTS:
        pts = np.zeros((len(F), 5))
        for tp, s in SHARE.items():
            for i, f in enumerate(F):
                al = simulate(data[tp], obo[tp], min(s * f, 0.5), r, N, k)
                m = measure(data[tp], al)
                pts[i] += [m['fh'], m['false_sig'], m['caught'], m['fresh'], m['sig']]
        curves[name] = pts
        print(f'{on} {name} f=1: лч {pts[3, 0]:.0f} лс {pts[3, 1]:.0f} поймано {pts[3, 2]:.0f} свежих {pts[3, 3]:.0f}',
              file=sys.stderr, flush=True)
    ALL[on] = curves
    ref = curves['raw'][3]
    print(f'\n### {on}: raw при долях как есть — ложных ч {ref[0]:.0f}, ложных сигналов {ref[1]:.0f}, '
          f'поймано {ref[2]:.0f}, свежих {ref[3]:.0f}\n')
    print('| вариант | при тех же долях: ложных ч / ложных сигн. / поймано / свежих | при равных ложных ч: '
          'поймано / свежих / ложных сигн. | при равных ложных сигн.: поймано / свежих / ложных ч | '
          'raw с тем же числом ложных ч (снижение бюджета): поймано / свежих / ложных сигн. |')
    print('|---|---|---|---|---|')
    for name, pts in curves.items():
        a = pts[3]
        o1 = np.argsort(pts[:, 0])
        b = [np.interp(ref[0], pts[o1, 0], pts[o1, j]) for j in (2, 3, 1)]
        o2 = np.argsort(pts[:, 1])
        c = [np.interp(ref[1], pts[o2, 1], pts[o2, j]) for j in (2, 3, 0)]
        rw = curves['raw']
        o3 = np.argsort(rw[:, 0])
        e = [np.interp(a[0], rw[o3, 0], rw[o3, j]) for j in (2, 3, 1)]
        print(f'| {name} | {a[0]:.0f} / {a[1]:.0f} / {a[2]:.0f} / {a[3]:.0f} | {b[0]:.0f} / {b[1]:.0f} / {b[2]:.0f} | '
              f'{c[0]:.0f} / {c[1]:.0f} / {c[2]:.0f} | {e[0]:.0f} / {e[1]:.0f} / {e[2]:.0f} |', flush=True)

# раздел 44: при равных ложных часах вокруг рабочей точки «168 ч, молчание»
for on, curves in ALL.items():
    base = curves['r1_N168_k0.0'][3]
    print(f'\n### {on}: при равных ложных часах, доли рабочей точки r1_N168_k0.0 ({base[0]:.0f})\n')
    tg = [0.8, 1.0, 1.15, 1.3]
    print('| вариант | ' + ' | '.join(f'{t:g}: свежих / ложных сигн.' for t in tg) + ' |')
    print('|---|' + '---|' * len(tg))
    for name, pts in curves.items():
        o1 = np.argsort(pts[:, 0])
        cells = [f'{np.interp(base[0] * t, pts[o1, 0], pts[o1, 3]):.0f} / {np.interp(base[0] * t, pts[o1, 0], pts[o1, 1]):.0f}'
                 for t in tg]
        print(f'| {name} | ' + ' | '.join(cells) + ' |')
