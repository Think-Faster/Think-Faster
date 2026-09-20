"""Отключить прогноз там, где он раньше почти всегда ошибался (пересмотр раздела 14).

Раздел 14 уже отбрасывал «молчание шумных объектов» — и отбрасывал правильно: там молчать
заставляли всех, у кого доля верных ниже половины, а это двадцать объектов из тридцати, и вместе
с ложными уходили настоящие эпизоды. Здесь проверяется то же средство, но дозированно: правило
задаётся не целевой долей верных, а порогом «не меньше n сигналов и верных не больше p», и при
малых p молчать заставляют три-восемь худших, а не всех подряд.

Два контроля, оба обязательны:
- **порогом**: общий порог поднимается до того же числа сигналов (дисциплина разделов 12 и 14);
- **наугад**: отключается столько же объектов, но выбранных случайно, — если выигрыш даёт сам
  факт отключения, а не выбор кого именно, случайный выбор даст то же самое.

И повтор на другом отрезке: правило, выбранное по первой половине проверки, применяется ко второй.
Один замер на тесте 2026 — это один год; повтор показывает, переносится ли правило вообще.
"""
import numpy as np
import config, metrics, operating as op

H = config.HORIZON
RUN = 'main_h24_tuned_r100e20'
RULES = ((2, 0.0), (3, 0.20), (3, 0.34), (5, 0.50))


def runs(o, h, y, a):
    """Сигналы как серии подряд идущих часов: (объект серии, истинна ли серия)."""
    oo, hh, yy = o[a], h[a], y[a]
    if not len(oo):
        return np.array([]), np.array([], bool)
    order = np.lexsort((hh, oo))
    oo, hh, yy = oo[order], hh[order], yy[order]
    st = np.empty(len(oo), bool)
    st[0] = True
    st[1:] = (oo[1:] != oo[:-1]) | (hh[1:] != hh[:-1] + 1)
    run = np.cumsum(st) - 1
    true = np.bincount(run, weights=yy, minlength=run[-1] + 1) > 0
    return oo[st], true


def noisy(o, h, y, p, thr, nmin, pmax):
    """Объекты, которые на этом отрезке шумели: сигналов ≥ nmin, верных ≤ pmax."""
    robj, rtrue = runs(o, h, y, p >= thr)
    if not len(robj):
        return set(), np.array([])
    u, inv = np.unique(robj, return_inverse=True)
    tot = np.bincount(inv, minlength=len(u))
    good = np.bincount(inv, weights=rtrue, minlength=len(u))
    return set(u[(tot >= nmin) & (good / tot <= pmax)].tolist()), u


def at_signals(o, h, nx, p, target, lo, cap):
    """Порог, дающий примерно target сигналов: двоичный поиск вверх от рабочего."""
    hi, out = float(p.max()), None
    for _ in range(40):
        mid = (lo + hi) / 2
        c = metrics.evaluate(o, h, nx, p, mid, H, cap)
        if c['signals'] > target:
            lo = mid
        else:
            hi, out = mid, c
    return out or c


def table(title, fit, apply_, cap):
    """fit и apply_ — кортежи (объекты, часы, next, вероятности) двух отрезков."""
    print(f'\n### {title}\n')
    print('| тип | правило | сигналов | ложных | поймано | отключено | '
          'контроль порогом: сигн. / ложн. / поймано | наугад: сигн. / ложн. / поймано |')
    print('|---|---|---:|---:|---:|---:|---|---|')
    for tp in config.TYPES:
        of, hf, nf, pf = fit[tp]
        oa, ha, na, pa = apply_[tp]
        thr = metrics.best_threshold((nf <= H).astype(np.int8), pf)
        base = metrics.evaluate(oa, ha, na, pa, thr, H, cap)
        print(f"| {config.TYPE_NAMES[tp]} | без отключения | {base['signals']} | "
              f"{base['signals_false']} | {base['caught']} из {base['episodes']} | 0 | — | — |")
        for nmin, pmax in RULES:
            mute, u = noisy(of, hf, (nf <= H).astype(np.int8), pf, thr, nmin, pmax)
            if not mute:
                continue
            keep = np.array([o not in mute for o in oa.tolist()])
            m = metrics.evaluate(oa, ha, na, np.where(keep, pa, 0.0), thr, H, cap)
            ctl = at_signals(oa, ha, na, pa, m['signals'], thr, cap)
            rng = np.random.default_rng(0)
            rs, rf_, rc = [], [], []
            for _ in range(20):
                rm = set(rng.choice(u, size=len(mute), replace=False).tolist())
                k = np.array([o not in rm for o in oa.tolist()])
                r = metrics.evaluate(oa, ha, na, np.where(k, pa, 0.0), thr, H, cap)
                rs.append(r['signals']); rf_.append(r['signals_false']); rc.append(r['caught'])
            print(f"| {config.TYPE_NAMES[tp]} | ≥{nmin} сигн., верных ≤{pmax:.0%} | "
                  f"{m['signals']} | {m['signals_false']} | {m['caught']} из {m['episodes']} | "
                  f"{len(mute)} из {len(u)} | "
                  f"{ctl['signals']} / {ctl['signals_false']} / {ctl['caught']} | "
                  f"{np.median(rs):.0f} / {np.median(rf_):.0f} / {np.median(rc):.0f} |")


def main():
    import json
    cap = json.loads((op.FEAT / 'meta.json').read_text(encoding='utf-8'))['next_cap']
    val, test, v1, v2 = {}, {}, {}, {}
    for tp in config.TYPES:
        val[tp] = op.split(RUN, 'val', 2025, tp, 'xgb')
        test[tp] = op.split(RUN, 'test', 2026, tp, 'xgb')
        o, h, n, p = val[tp]
        cut = (h.min() + h.max()) // 2
        a = h <= cut
        v1[tp] = (o[a], h[a], n[a], p[a])
        v2[tp] = (o[~a], h[~a], n[~a], p[~a])
    print(f'Прогон {RUN}. Порог — по лучшему F1 на отрезке, где выбрано правило.')
    table('Правило выбрано на проверке 2025, применено к тесту 2026', val, test, cap)
    table('Повтор: правило выбрано на первой половине 2025, применено ко второй', v1, v2, cap)


main()
