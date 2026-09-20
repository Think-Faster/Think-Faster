"""Шаг 20. Свой порог для каждого объекта: ложные тревоги приходят не отовсюду.

Один порог на все объекты неявно предполагает, что объекты одинаковы. Они не одинаковы: на тесте
2026 по пожару тревоги подняты на 30 объектах, но половину всех ложных сигналов дают шесть из них,
а у отдельных объектов 33 ложных сигнала из 35. Глобальный порог такому объекту ничего не
противопоставляет — поднимая его, мы одинаково режем и шумный объект, и спокойный.

Здесь порог подбирается каждому объекту отдельно — на проверке 2025, и переносится на тест 2026
без подглядывания. Правила:

| правило | что делает |
|---|---|
| `mute` | объект, у которого доля верных на проверке ниже целевой, молчит целиком |
| `own` | объекту поднимается порог до уровня, на котором его доля верных дотягивает до целевой; если не дотягивает ни при каком — молчит |

Опасность метода — подгонка: у объекта с тремя сигналами за год «доля верных» ничего не значит.
Поэтому свой порог получают только объекты, набравшие на проверке не меньше `--min-signals`
сигналов; остальные живут на общем пороге. Вариант без этого ограничения считается рядом и
показан как отброшенный.

Контроль тот же, что у второй ступени: столько же сигналов, но достигнутых простым подъёмом
общего порога. Без него метод ничего не доказывает.

    python perobject.py
    python perobject.py --target 0.6 --min-signals 5
"""
import argparse

import numpy as np

import config
import maintenance as mt
import metrics
import operating as op


def runs(obj, h, y, alarm):
    """Разбиение тревожных часов на сигналы: (объект сигнала, верный ли, индексы часов)."""
    idx = np.where(alarm)[0]
    if not len(idx):
        return np.array([]), np.array([]), []
    o, hh = obj[idx], h[idx]
    order = np.lexsort((hh, o))
    idx, o, hh = idx[order], o[order], hh[order]
    st = np.empty(len(o), bool)
    st[0] = True
    st[1:] = (o[1:] != o[:-1]) | (hh[1:] != hh[:-1] + 1)
    r = np.cumsum(st) - 1
    n = int(r[-1]) + 1
    true = np.bincount(r, weights=y[idx], minlength=n) > 0
    return o[st], true, [idx[r == k] for k in range(n)]


def sig_labels(obj, h, y, alarm) -> np.ndarray:
    """Метка «верный» для каждого часа тревоги: верен сигнал целиком, а не отдельный час."""
    out = np.zeros(len(obj), np.int8)
    _, true, groups = runs(obj, h, y, alarm)
    for t, g in zip(true, groups):
        out[g] = int(t)
    return out


def own_threshold(p_obj, y_obj, thr0, target, grid=60):
    """Наименьший порог ≥ рабочего, при котором доля верных сигналов объекта дотягивает до целевой.

    Считается по сигналам, а не по часам: диспетчер видит серию, а не строки витрины.
    None — не дотягивает ни при каком пороге, объект под молчание.
    """
    above = p_obj[p_obj >= thr0]
    if not len(above):
        return thr0
    for q in np.linspace(0.0, 0.98, grid):
        t = float(np.quantile(above, q))
        m = p_obj >= t
        if not m.any():
            break
        good = float(y_obj[m].mean())
        if good >= target:
            return t
    return None


def control(obj, h, y, nxt, p, thr0, want, H):
    """Столько же сигналов, но просто общим порогом повыше."""
    above = p[p >= thr0]
    best = None
    for q in np.linspace(0.0, 0.995, 200):
        t = float(np.quantile(above, q)) if len(above) else 1.1
        sig, true = metrics.signals(obj, h, y, p >= t)
        if best is None or abs(sig - want) < abs(best[0] - want):
            best = (sig, sig - true, t)
    sig, false, t = best
    m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
    return sig, false, m['caught']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--target', type=float, default=0.5, help='целевая доля верных сигналов объекта')
    ap.add_argument('--min-signals', type=int, default=8,
                    help='сколько сигналов объект должен набрать на проверке, чтобы получить свой порог')
    args = ap.parse_args()
    H = args.horizon

    print(f'Пороги подобраны по объектам на проверке 2025 и перенесены на тест 2026 без изменений. '
          f'Прогон {args.run}, цель по доле верных {args.target}, свой порог получают объекты от '
          f'{args.min_signals} сигналов на проверке.\n'.replace('0.', '0,'))
    print('| тип | правило | объектов затронуто | сигналов | из них ложных | доля верных | '
          'поймано эпизодов |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        ov, hv, nv, pv = op.split(args.run, 'val', 2025, tp, args.model)
        os_, hs, ns, ps = op.split(args.run, 'test', 2026, tp, args.model)
        thr = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        yv, ys = (nv <= H).astype(np.int8), (ns <= H).astype(np.int8)
        av, as_ = pv >= thr, ps >= thr
        name = config.TYPE_NAMES[tp]
        base = metrics.evaluate(os_, hs, ns, ps, thr, H, metrics.RUN_CAP)
        sig, true = metrics.signals(os_, hs, ys, as_)
        print(f'| {name} | общий порог | — | {sig} | {sig - true} | '
              f"{1 - (sig - true) / sig:.3f} | {base['caught']} из {base['episodes']} |"
              .replace('.', ',') if sig else f'| {name} | общий порог | — | 0 | | | |')
        if not sig:
            continue

        vobj, vtrue, _ = runs(ov, hv, yv, av)
        stat = {}
        for o in np.unique(vobj):
            m = vobj == o
            stat[int(o)] = (int(m.sum()), float(vtrue[m].mean()))
        vlab = sig_labels(ov, hv, yv, av)   # «верный» проставлен всему сигналу, а не часу

        for rule in ('mute', 'own'):
            for guard, tag in ((args.min_signals, ''), (1, ' без ограничения по числу сигналов')):
                mask = np.zeros(len(as_), bool)   # True — час гасится
                touched = 0
                for o, (n, good) in stat.items():
                    if n < guard or good >= args.target:
                        continue
                    sel = (os_ == o)
                    if rule == 'mute':
                        mask |= sel & as_
                        touched += 1
                        continue
                    t = own_threshold(pv[sel_v := (ov == o)], vlab[sel_v], thr, args.target)
                    if t is None:
                        mask |= sel & as_
                    else:
                        mask |= sel & as_ & (ps < t)
                    touched += 1
                # пятым идут именно погашенные часы; сигнал пропадает, только если погашен весь
                sig2, false2 = mt.shown_signals(os_, hs, ys, as_, mask)
                m2 = metrics.evaluate(os_, hs, ns, np.where(mask, 0.0, ps), thr, H, metrics.RUN_CAP)
                label = ('молчание шумных объектов' if rule == 'mute' else 'свой порог объекту') + tag
                good = f'{1 - false2 / sig2:.3f}' if sig2 else '—'
                print(f'| {name} | {label} | {touched} | {sig2} | {false2} | {good} | '
                      f"{m2['caught']} из {m2['episodes']} |".replace('.', ','))
                if not tag and sig2:
                    s3, f3, c3 = control(os_, hs, ys, ns, ps, thr, sig2, H)
                    g3 = f'{1 - f3 / s3:.3f}' if s3 else '—'
                    print(f'| {name} | …то же числом сигналов, но общим порогом | — | {s3} | {f3} | '
                          f"{g3} | {c3} из {base['episodes']} |".replace('.', ','))


if __name__ == '__main__':
    main()
