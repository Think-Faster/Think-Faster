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
    python perobject.py --refit 30   # пороги объектов пересчитываются раз в 30 суток
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


def fit_stats(obj, h, y, alarm):
    """По окну подгонки: сколько сигналов у объекта и какая доля из них верна, плюс метки часов."""
    fobj, ftrue, _ = runs(obj, h, y, alarm)
    stat = {}
    for o in np.unique(fobj):
        m = fobj == o
        stat[int(o)] = (int(m.sum()), float(ftrue[m].mean()))
    return stat, sig_labels(obj, h, y, alarm)


def object_mask(rule, guard, target, thr, fit, tobj, tp_, tsel):
    """Какие тревожные часы гасим. `fit` — окно подгонки (obj, h, y, p), только прошлое."""
    fobj, fh, fy, fp = fit
    stat, flab = fit_stats(fobj, fh, fy, fp >= thr)
    mask, touched = np.zeros(len(tobj), bool), 0
    for o, (n, good) in stat.items():
        if n < guard or good >= target:
            continue
        sel = tsel & (tobj == o)
        if rule == 'mute':
            mask |= sel
        else:
            fsel = fobj == o
            t = own_threshold(fp[fsel], flab[fsel], thr, target)
            mask |= sel if t is None else (sel & (tp_ < t))
        touched += 1
    return mask, touched


def rolling_mask(rule, guard, target, thr, H, days, val, test, tsel):
    """То же, но пороги пересчитываются каждые `days` суток по предыдущим `days` суткам.

    Раздел 13 показал, что меняется не парк, а отдельные объекты: у подтопления две трети эпизодов
    2025 года дал один объект, к 2026 самым частым стал другой. Пороги, снятые на 2025 один раз,
    настроены на объекты, которых в тесте уже нет. Здесь они снимаются на скользящем окне.

    Заглядывать в будущее нельзя: окно подгонки — предыдущий отрезок, и его последние H часов
    отброшены, потому что к моменту решения их метка ещё не известна. Для самого первого отрезка
    прошлого внутри теста нет, поэтому берётся проверка целиком — как в обычном режиме.
    """
    tobj, th, ty, tp_ = test
    blk = (th - th.min()) // (days * 24)
    mask, touched = np.zeros(len(tobj), bool), 0
    for b in range(int(blk.max()) + 1):
        prev = blk == b - 1 if b else np.zeros(len(th), bool)
        if prev.any():
            prev &= th < th[prev].max() - H + 1
        # для первого отрезка прошлого внутри теста нет, для пустого — тоже: берём проверку
        fit = (tobj[prev], th[prev], ty[prev], tp_[prev]) if prev.any() else val
        m, t = object_mask(rule, guard, target, thr, fit, tobj, tp_, tsel & (blk == b))
        mask |= m
        touched = max(touched, t)
    return mask, touched


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
    ap.add_argument('--refit', type=int, default=0,
                    help='пересчитывать пороги объектов каждые N суток по предыдущим N суткам; '
                         '0 — один раз на проверке 2025, как было (раздел 14)')
    args = ap.parse_args()
    H = args.horizon

    how = (f'Пороги объектов пересчитываются каждые {args.refit} сут по предыдущим {args.refit} сут '
           f'теста; последние {H} ч окна отброшены — их метка к моменту решения не известна. '
           f'Первый отрезок считается по проверке 2025.' if args.refit else
           'Пороги подобраны по объектам на проверке 2025 и перенесены на тест 2026 без изменений.')
    print(f'{how} Прогон {args.run}, цель по доле верных {args.target}, свой порог получают '
          f'объекты от {args.min_signals} сигналов на окне подгонки.\n'.replace('0.', '0,'))
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

        val, test = (ov, hv, yv, pv), (os_, hs, ys, ps)
        for rule in ('mute', 'own'):
            for guard, tag in ((args.min_signals, ''), (1, ' без ограничения по числу сигналов')):
                if args.refit:
                    mask, touched = rolling_mask(rule, guard, args.target, thr, H,
                                                 args.refit, val, test, as_)
                else:
                    mask, touched = object_mask(rule, guard, args.target, thr, val, os_, ps, as_)
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
