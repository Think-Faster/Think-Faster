"""Скользящая калибровка порога: держать бюджет тревог, а не значение score (раздел 32).

Порог у нас снимается один раз на проверке 2025 и дальше не меняется. Раздел 18 показал, что
признаки и цель дрейфуют; значит, дрейфует и распределение оценки, а вместе с ним — и то, сколько
тревог даёт один и тот же порог. Полугодие спустя «0,099» означает уже не то, что означало.

Приём из литературы по конформному прогнозу (adaptive conformal / temporal quantile adjustment):
порог задаётся не числом, а **долей**. Каждые сутки он пересчитывается как квантиль оценок за
последние K суток, так что поток тревог держится на заданном уровне сам, что бы ни делал score.
Отсюда же и вопрос, который стоит в задании про дообучение: **на скольких днях калибровать** —
короткое окно подстраивается быстро, но шумит, длинное устойчиво, но опаздывает за дрейфом.

Сравнение честное только при равной полноте: у фиксированного порога и у скользящего разные
шкалы, поэтому у каждого варианта строится кривая «поймано эпизодов — ложных сигналов».

История берётся сквозной: для первых суток теста окно калибровки заходит в 2025 — так и будет в
работе, никакого заглядывания вперёд тут нет.

    python calib.py                    # окна калибровки против фиксированного порога
    python calib.py --smooth 6         # поверх сглаживания оценки (раздел 29)
    python calib.py --transfer         # перенос порога с проверки на тест: держится ли бюджет
    python calib.py --run "$MIX5" --cost hours   # на смеси раздела 34, в ложных часах (раздел 37)
"""
import argparse

import numpy as np

import config
import metrics
import operating as op
import persist

DAYS = (7, 14, 30, 60, 90)
LEVELS = (0.4, 0.5, 0.6, 0.7, 0.75)


def prepare_union(run, tp, model, horizon):
    """Сетка объект × час по 2025 и 2026 сразу, но считаем только по строкам теста.

    Калибровке нужна история до начала теста, а метрикам — только тест. Поэтому сетка общая,
    а индексы строк, порядок и эпизоды берутся из теста.
    """
    # общая шкала для двух периодов: у смеси — доли оценок зерна на проверке 2025
    ov, hv, nv, pv = op.load_mix(run, 'val', 2025, tp, model, ref=('val', 2025))
    ot, ht, nt, pt = op.load_mix(run, 'test', 2026, tp, model, ref=('val', 2025))
    obj = np.concatenate([ov, ot])
    h = np.concatenate([hv, ht])
    p = np.concatenate([pv, pt])
    grid, objs, _, _, h0 = persist.dense(obj, h, p, horizon)
    oi = np.searchsorted(objs, ot)
    hi = (ht - h0).astype(np.int64)
    y = (nt <= horizon).astype(np.float64)
    order = np.lexsort((ht, ot))
    eps = metrics.onsets(ot, ht, nt, metrics.RUN_CAP)
    seen = set(zip(ot.tolist(), ht.tolist()))
    keep = [(o, e) for o, e in eps if any((o, e - k) in seen for k in range(1, horizon + 1))]
    eo = np.searchsorted(objs, np.array([o for o, _ in keep]))
    ec = np.array([e for _, e in keep]) - h0
    return {'grid': grid, 'oi': oi, 'hi': hi, 'order': order, 'horizon': horizon, 'y': y,
            'oo': ot[order], 'oh': ht[order], 'oy': y[order],
            'eo': eo, 'ec': ec, 'episodes': len(keep), 'test_from': int(hi.min())}


def rolling_threshold(score: np.ndarray, days: int, qs: np.ndarray, start: int) -> np.ndarray:
    """Пороги на каждый час для всех долей сразу: (len(qs), часов).

    Квантиль считается по оценкам всего парка за последние `days` суток. Пересчёт раз в сутки, а не
    каждый час: диспетчерская живёт сменами, и порог, который прыгает ежечасно, невозможно ни
    объяснить, ни проверить. Все доли берутся одним проходом — иначе перебор считался бы часами.
    """
    t = score.shape[1]
    thr = np.full((len(qs), t), np.inf, np.float32)
    win = days * 24
    for d0 in range(start, t, 24):
        hist = score[:, max(d0 - win, 0):d0]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 100:
            continue
        thr[:, d0:d0 + 24] = np.quantile(hist, qs)[:, None]
    return thr


def day_counts(alarm: np.ndarray, start: int) -> np.ndarray:
    """Тревожных часов по всему парку за каждые сутки теста."""
    t = alarm.shape[1]
    return np.array([alarm[:, d:d + 24].sum() for d in range(start, t - 23, 24)], np.float64)


def transfer(args) -> None:
    """Перенос порога с проверки на тест — то, что происходит в эксплуатации.

    Раздел 11 в `models.md`: порог, снятый на проверке, на тесте даёт уже не тот поток тревог.
    Здесь оба способа переносятся вслепую: фиксированное число против доли, пересчитываемой
    каждые сутки по последним 90. Проверяется не полнота, а обещание — сколько тревог в сутки.
    """
    q = args.q
    hours = args.cost == 'hours'
    print(f'Прогон {args.run}, модель {args.model}. Порог снят на проверке 2025 по доле {q:.3f} '
          f'и перенесён на тест 2026 вслепую'
          + (f', сглаживание {args.smooth} ч' if args.smooth > 1 else '') + '.\n')
    print('Тревожных часов в сутки по всему парку: обещано проверкой — получилось на тесте.\n')
    print('| тип | на проверке | фиксированный | скользящий 90 сут |')
    print('|---|---:|---:|---:|')
    keep = {}
    for tp in config.TYPES:
        try:
            ctx = prepare_union(args.run, tp, args.model, args.horizon)
        except FileNotFoundError:
            continue
        score = persist.smoothed(ctx, args.smooth)
        start = ctx['test_from']
        seen = ~np.isnan(score)
        hist = score[:, :start][seen[:, :start]]
        fix = float(np.quantile(hist, q))
        roll = rolling_threshold(score, 90, np.array([q]), start)[0]
        af = (score >= fix) & seen
        ar = (score >= roll[None, :]) & seen
        promised = af[:, :start].sum() / max(start / 24, 1)
        df, dr = day_counts(af, start), day_counts(ar, start)
        keep[tp] = (ctx, score, fix, roll)
        print(f'| {config.TYPE_NAMES[tp]} | {promised:.1f} | {df.mean():.1f} '
              f'(разброс {df.std():.1f}) | {dr.mean():.1f} (разброс {dr.std():.1f}) |', flush=True)

    print('\nЧто это даёт на тесте при том же переносе:\n')
    unit = 'ложных часов' if hours else 'ложных'
    print(f'| тип | фикс: поймано | фикс: {unit} | скольз 90: поймано | скольз 90: {unit} |')
    print('|---|---:|---:|---:|---:|')
    for tp, (ctx, score, fix, roll) in keep.items():
        ev = score.copy()
        ev[:, :ctx['test_from']] = np.nan
        cf, ff, _ = persist.curve(ev, 1, [fix], ctx, args.gap, hours)[0]
        cr, fr, _ = persist.curve(ev - roll[None, :], 1, [0.0], ctx, args.gap, hours)[0]
        n = ctx['episodes']
        print(f'| {config.TYPE_NAMES[tp]} | {cf} из {n} | {ff} | {cr} из {n} | {fr} |', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned_r100e20')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--gap', type=int, default=6, help='склейка дребезга, раздел 27')
    ap.add_argument('--smooth', type=int, default=0, help='сглаживание оценки, раздел 29')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--transfer', action='store_true',
                    help='перенос порога с проверки на тест вместо кривых')
    ap.add_argument('--q', type=float, default=0.995, help='доля для переноса')
    ap.add_argument('--cost', choices=['sig', 'hours'], default='sig',
                    help='мера ложных: сигналы или часы ложной тревоги (раздел 34)')
    args = ap.parse_args()
    if args.transfer:
        transfer(args)
        return

    hours = args.cost == 'hours'
    unit = 'часов ложной тревоги' if hours else 'ложных сигналов'
    names = ['фиксированный'] + [f'скользящий {d} сут' for d in DAYS]
    head = f'Прогон {args.run}, модель {args.model}, тест 2026, склейка дребезга {args.gap} ч'
    print(head + (f', сглаживание {args.smooth} ч' if args.smooth > 1 else '')
          + f'. В ячейках — {unit} при равной доле пойманных эпизодов.\n')
    print('| тип | поймано эпизодов | ' + ' | '.join(names) + ' |')
    print('|---|---|' + '---:|' * len(names))
    for tp in config.TYPES:
        try:
            ctx = prepare_union(args.run, tp, args.model, args.horizon)
        except FileNotFoundError:
            continue
        score = persist.smoothed(ctx, args.smooth)
        # калибровке история нужна, метрикам — нет: тревоги считаем только с первого часа теста
        ev = score.copy()
        ev[:, :ctx['test_from']] = np.nan
        qs = np.linspace(0.90, 0.99999, args.steps)
        flat = ev[~np.isnan(ev)]
        curves = {'фиксированный': persist.curve(ev, 1, np.quantile(flat, qs), ctx, args.gap, hours)}
        for d in DAYS:
            thr = rolling_threshold(score, d, qs, ctx['test_from'])
            # порог зависит от часа, поэтому вычитаем его из оценки и режем по нулю
            curves[f'скользящий {d} сут'] = [
                persist.curve(ev - thr[i][None, :], 1, [0.0], ctx, args.gap, hours)[0]
                for i in range(len(qs))]
        total = ctx['episodes']
        for lv in LEVELS:
            need = lv * total
            cells = []
            for n in names:
                ok = [f for c, f, _ in curves[n] if c >= need]
                cells.append(str(min(ok)) if ok else '—')
            print(f'| {config.TYPE_NAMES[tp]} | {lv:.0%} ({int(need)} из {total}) | '
                  + ' | '.join(cells) + ' |', flush=True)


if __name__ == '__main__':
    main()
