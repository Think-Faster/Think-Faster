"""Затухающее отклонение: после отказа от прогноза вес тревог объекта снижается и плавно возвращается.

Середина между двумя крайностями: «ничего не делать» (правило выключено) и «ступенька до события»
(раздел 44: доля share·k до следующего настоящего эпизода). Здесь через Δt часов после отклонения
тревога горит, только если оценка входит в долю share·w(Δt), где

    w(Δt) = 1 − (1 − k0) · exp(−Δt / τ)

— сразу после отклонения k0, через τ часов восстановлено 63% разницы, через 3τ — 95%. Настоящий
эпизод этого типа на объекте снимает снижение сразу. Каждый новый сигнал диспетчер проверяет заново
(через r = 1 ч, как в разделе 44): новое отклонение перезапускает отсчёт с k0.

Сравнение честное — при равном числе ложных сигналов: для каждого типа строится кривая «просто
снизить долю» (без правила), и свежие поимки варианта сравниваются со свежими на этой кривой в той
же точке по ложным сигналам. Плюс — правило даёт больше, чем простое снижение порога. Параметры
выбираются на проверке 2025, тест 2026 — контроль.

    python decay.py "$MIXT"            # перебор → work/decay.json
    python decay.py "$MIXT" --report   # таблицы раздела 47
"""
import sys
import json
import numpy as np
import config
from smooth import load, measure
from reject import simulate, GAP

K0 = (0.0, 0.1, 0.2, 0.35, 0.5)
TAU = (6, 24, 72, 168, 336, 720)
F = np.geomspace(0.3, 4, 23)
FR = (0.7, 1.0, 1.4, 2.0, 2.8)


def simulate_decay(d, ons_by, share, k0, tau, r=1):
    """Как `reject.simulate`, но снижение доли затухает с постоянной τ часов."""
    p = d['p']
    base = np.quantile(p, 1 - share)
    alarm = p >= base
    out = alarm.copy()
    o, h = d['o'], d['h']
    n = len(p)
    cur = None
    for i in np.flatnonzero(alarm):
        ob, t = o[i], h[i]
        if ob != cur:
            cur, rej_at, last_on, sig_start, checked = ob, None, -10**9, None, True
            ons = ons_by.get(ob, np.empty(0, np.int64))
        if rej_at is not None:
            j = np.searchsorted(ons, t, side='right') - 1
            if j >= 0 and ons[j] >= rej_at:           # настоящий эпизод снимает снижение
                rej_at = None
            else:
                w = 1 - (1 - k0) * np.exp(-(t - rej_at) / tau)
                if w > 0.999:
                    rej_at = None
                elif p[i] < 1 - share * w - 0.5 / n:  # p — ранги, доля q ≈ порог 1 − q
                    out[i] = False
                    continue
        if sig_start is None or t - last_on > GAP + 1:
            sig_start, checked = t, False
        last_on = t
        if checked or t < sig_start + r:
            continue
        checked = True
        j = np.searchsorted(ons, sig_start, side='left')
        if not (j < len(ons) and ons[j] <= t):
            rej_at = t
    return out


def pack(m, ne):
    return [m['false_sig'], m['fh'], m['caught'], m['fresh'], ne - m['caught']]


def main():
    cfg = config.operating()['types']
    res = {}
    for tp in config.TYPES:
        for on in ('val', 'test'):
            d = load(tp, on)
            by = {}
            for a, b in d['eps']:
                by.setdefault(a, []).append(b)
            ons = {a: np.array(sorted(b), np.int64) for a, b in by.items()}
            ne = len(d['W'])
            s = cfg[tp]['share']
            R = {'curve': [pack(measure(d, simulate(d, ons, s * f, 0, 0, 0.0)), ne) for f in F],
                 'none': pack(measure(d, simulate(d, ons, s, 0, 0, 0.0)), ne)}
            for f in FR:
                for k in (0.0, 0.2):
                    R[f'step_k{k}@{f}'] = pack(measure(d, simulate(d, ons, s * f, 1, 10**6, k)), ne)
                for k0 in K0:
                    for tau in TAU:
                        R[f'decay_k{k0}_t{tau}@{f}'] = pack(measure(d, simulate_decay(d, ons, s * f, k0, tau)), ne)
            res[f'{tp}|{on}'] = {'ne': ne, 'R': R}
            print(tp, on, 'готово', file=sys.stderr, flush=True)
    json.dump(res, open(config.WORK / 'decay.json', 'w'))


cfg = config.operating()['types']


def at(pts, fh, j):
    pts = np.array(pts, float)
    s = np.argsort(pts[:, 1])
    return float(np.interp(fh, pts[s, 1], pts[s, j]))


def curve(r, rule):
    return r['R']['curve'] if rule == 'none' else [r['R'][f'{rule}@{f}'] for f in FR]


def ref_fh(r, tp):
    cur = 'none' if cfg[tp]['reject_k'] is None else 'step_k0.2@1.0'
    return r['R']['none'][1] if cur == 'none' else r['R'][cur][1]


def gains(r, tp, rule):
    fh0 = ref_fh(r, tp)
    out = []
    for lv in LV:
        fh = fh0 * lv
        out.append([at(curve(r, rule), fh, j) - at(curve(r, 'none'), fh, j) for j in (2, 3, 0)])
    return np.array(out)                  # уровни × (поймано, свежих, ложных сигналов)


def report():
    """Таблицы раздела 47 по work/decay.json: выигрыш при равных ложных часах."""
    global o, rules, LV
    o = json.load(open(config.WORK / 'decay.json'))
    LV = (1.0, 1.5, 2.0)                       # уровни ложных часов от текущих настроек типа
    rules = sorted({n.split('@')[0] for n in o['fire|val']['R'] if '@' in n})
    pick = {}
    print('### Лучшее по проверке 2025 (сумма поимок сверх простой смены доли на трёх уровнях ложных часов), тест 2026\n')
    print('| тип | правило | поймано сверх: ×1 / ×1,5 / ×2 ложных ч | свежих сверх | ложных сигналов сверх | '
          'ступенька k=0,2: поймано сверх | ступенька: свежих сверх |')
    print('|---|---|---|---|---|---|---|')
    for tp in config.TYPES:
        v, t = o[f'{tp}|val'], o[f'{tp}|test']
        cand = [x for x in rules if x.startswith('decay')]
        best = max(cand, key=lambda x: gains(v, tp, x)[:, 0].sum())
        pick[tp] = best
        g, gs = gains(t, tp, best), gains(t, tp, 'step_k0.2')
        f = lambda a: ' / '.join(f'{x:+.0f}' for x in a)
        print(f"| {config.TYPE_NAMES[tp]} | {best.replace('decay_', '')} | {f(g[:, 0])} | {f(g[:, 1])} | {f(g[:, 2])} | "
              f"{f(gs[:, 0])} | {f(gs[:, 1])} |")

    print('\n### Проверка 2025 для тех же правил\n')
    print('| тип | правило | поймано сверх | свежих сверх | ложных сигналов сверх |')
    print('|---|---|---|---|---|')
    for tp in config.TYPES:
        g = gains(o[f'{tp}|val'], tp, pick[tp])
        f = lambda a: ' / '.join(f'{x:+.0f}' for x in a)
        print(f"| {config.TYPE_NAMES[tp]} | {pick[tp].replace('decay_', '')} | {f(g[:, 0])} | {f(g[:, 1])} | {f(g[:, 2])} |")

    # сетка по всем типам вместе: сумма выигрыша поимок на тесте при ×1,5
    for on in ('val', 'test'):
        print(f'\n### {on}: поимок сверх простой смены доли, сумма по типам, уровни ×1 / ×1,5 / ×2\n')
        print('| k0 \\ τ, ч | ' + ' | '.join(['6', '24', '72', '168', '336', '720']) + ' |')
        print('|---|' + '---|' * 6)
        for k0 in ('0.0', '0.1', '0.2', '0.35', '0.5'):
            cells = []
            for tau in ('6', '24', '72', '168', '336', '720'):
                s = sum(gains(o[f'{tp}|{on}'], tp, f'decay_k{k0}_t{tau}')[:, 0] for tp in config.TYPES)
                cells.append('/'.join(f'{x:+.0f}' for x in s))
            print(f'| {k0} | ' + ' | '.join(cells) + ' |')
        s = sum(gains(o[f'{tp}|{on}'], tp, 'step_k0.2')[:, 0] for tp in config.TYPES)
        print(f"\nступенька k=0,2 у всех типов: {'/'.join(f'{x:+.0f}' for x in s)}")

    # итог: полнота при равных ложных часах
    for on in ('val', 'test'):
        print(f'\n### {on}: полнота и свежие при равных ложных часах, все типы\n')
        print('| ложных ч (от текущих) | ложных ч | без правила: поймано / полнота / свежих / ложных сигн. | '
              'выбранное затухание по типам: то же |')
        print('|---|---|---|---|')
        ne = sum(o[f'{tp}|{on}']['ne'] for tp in config.TYPES)
        for i, lv in enumerate(LV):
            a = np.zeros(3)
            b = np.zeros(3)
            fh = 0
            for tp in config.TYPES:
                r = o[f'{tp}|{on}']
                x = ref_fh(r, tp) * lv
                fh += x
                a += [at(curve(r, 'none'), x, j) for j in (2, 3, 0)]
                b += [at(curve(r, pick[tp]), x, j) for j in (2, 3, 0)]
            print(f'| ×{lv:g} | {fh:.0f} | {a[0]:.0f} / {a[0] / ne:.1%} / {a[1]:.0f} / {a[2]:.0f} | '
                  f'{b[0]:.0f} / {b[0] / ne:.1%} / {b[1]:.0f} / {b[2]:.0f} |')



if __name__ == '__main__':
    report() if '--report' in sys.argv else main()
