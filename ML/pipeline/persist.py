"""Подтверждение вместо мгновенной тревоги: задержка на включение (раздел 29).

Раздел 27 склеивал повторы уже после того, как тревога поднялась. Здесь вопрос обратный: стоит ли
вообще поднимать её с первого часа. В EEMUA 191 и ISA 18.2 это отдельный приём — on-delay: сигнал
объявляется, только если условие держится k подряд замеров. Смысл в том, что случайный выброс
скоринга живёт один час, а настоящее ухудшение держится.

Проверяются два семейства, оба — только постобработка готовых оценок, модель не трогается:
- `подряд k` — тревога с k-го часа, когда оценка держится выше порога k часов подряд;
- `среднее m` — порог ставится не на оценку часа, а на её среднее за последние m часов.

Сравнение честное только при равной полноте: у каждого варианта свой порог, и порог, снятый по F1,
у них стоит в разных местах. Поэтому для каждого варианта строится кривая «поймано эпизодов —
ложных сигналов», и на заданных долях пойманных эпизодов берётся наименьшее число ложных.

Цена приёма — упреждение: подтверждение съедает часы. Поэтому вторая таблица — медианное
упреждение в той же точке, а не только выигрыш по ложным.

    python persist.py                       # тест 2026
    python persist.py --on val              # проверка 2025
    python persist.py --gap 6               # вместе со склейкой дребезга (раздел 27)
"""
import argparse

import numpy as np

import config
import metrics
import operating as op

LEVELS = (0.4, 0.5, 0.6, 0.7, 0.75)
LEAD_LEVEL = 0.6
BIG = np.iinfo(np.int32).max


ENS = (1, 6, 12)   # окна, по которым усредняет ensemble()


def dense(obj: np.ndarray, h: np.ndarray, p: np.ndarray, horizon: int):
    """Строки витрины → прямоугольник объект × час (пропущенные часы остаются nan)."""
    objs = np.unique(obj)
    oi = np.searchsorted(objs, obj)
    h0 = int(h.min())
    hi = (h - h0).astype(np.int64)
    # запас справа: эпизод может начаться позже последнего часа витрины, окно перед ним нужно целиком
    t = int(hi.max()) + 1 + horizon
    grid = np.full((len(objs), t), np.nan, np.float32)
    grid[oi, hi] = p
    return grid, objs, oi, hi, h0


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """Сдвиг вправо по времени: было k часов назад. Слева — «не было»."""
    out = np.zeros_like(a)
    if k < a.shape[1]:
        out[:, k:] = a[:, :a.shape[1] - k]
    return out


def rolling_mean(grid: np.ndarray, m: int) -> np.ndarray:
    """Среднее оценки за последние m часов, включая текущий. Пропуски в среднее не входят."""
    s = np.nan_to_num(grid, nan=0.0).astype(np.float64)
    v = (~np.isnan(grid)).astype(np.float64)
    z = np.zeros((grid.shape[0], 1))
    cs = np.concatenate([z, np.cumsum(s, axis=1)], axis=1)
    cv = np.concatenate([z, np.cumsum(v, axis=1)], axis=1)
    lo = np.maximum(np.arange(1, grid.shape[1] + 1) - m, 0)
    num = cs[:, 1:] - cs[:, lo]
    den = cv[:, 1:] - cv[:, lo]
    out = np.where(den > 0, num / np.maximum(den, 1.0), np.nan).astype(np.float32)
    out[np.isnan(grid)] = np.nan   # час, которого нет в витрине, тревогой быть не может
    return out


def smooth_rows(obj: np.ndarray, h: np.ndarray, p: np.ndarray, m: int) -> np.ndarray:
    """Сглаживание оценки по объекту (раздел 29) прямо на строках витрины.

    Нужно там, где счёт идёт не по прямоугольнику, а по строкам, — например в factalert.py.
    """
    if m <= 1:
        return p
    grid, _, oi, hi, _ = dense(obj, h, p, 0)
    return rolling_mean(grid, m)[oi, hi]


def variants(grid: np.ndarray) -> dict[str, tuple[np.ndarray, int]]:
    """Имя варианта → (оценка, сколько часов подряд она должна держаться выше порога)."""
    v = {'как есть': (grid, 1)}
    for k in (2, 3, 4):
        v[f'подряд {k} ч'] = (grid, k)
    for m in (3, 6, 12):
        v[f'среднее {m} ч'] = (rolling_mean(grid, m), 1)
    return v


def alarm_grid(score: np.ndarray, thr: float, k: int) -> np.ndarray:
    a = np.nan_to_num(score, nan=-np.inf) >= thr
    out = a
    for j in range(1, k):
        out = out & shift(a, j)
    return out


def signals_fast(order_obj: np.ndarray, order_h: np.ndarray, order_y: np.ndarray,
                 alarm_ordered: np.ndarray, gap: int) -> tuple[int, int]:
    """То же, что metrics.signals, но строки уже отсортированы по (объект, час).

    Порядок один и тот же для всех порогов, а сортировка — самое дорогое место; здесь она вынесена
    наружу, иначе перебор порогов считался бы часами.
    """
    idx = np.flatnonzero(alarm_ordered)
    if not len(idx):
        return 0, 0
    o, hh, yy = order_obj[idx], order_h[idx], order_y[idx]
    start = np.empty(len(idx), bool)
    start[0] = True
    start[1:] = (o[1:] != o[:-1]) | (hh[1:] - hh[:-1] > gap + 1)
    run = np.cumsum(start) - 1
    total = int(run[-1]) + 1
    true = int(np.bincount(run, weights=yy, minlength=total).astype(bool).sum())
    return total, true


def curve(score, k, thr_grid, ctx, gap):
    """Для каждого порога: (поймано эпизодов, ложных сигналов, медианное упреждение)."""
    oi, hi, eo, ec, horizon = ctx['oi'], ctx['hi'], ctx['eo'], ctx['ec'], ctx['horizon']
    pts = []
    for t in thr_grid:
        a = alarm_grid(score, t, k)
        cs = np.concatenate([np.zeros((a.shape[0], 1), np.int32),
                             np.cumsum(a, axis=1, dtype=np.int32)], axis=1)
        # окно перед эпизодом: часы ec-horizon … ec-1, в терминах cs это столбцы ec-horizon … ec
        cols = ec[:, None] - horizon + np.arange(horizon + 1)
        win = cs[eo[:, None], np.clip(cols, 0, cs.shape[1] - 1)]
        got = win[:, -1] - win[:, 0]
        caught = int((got > 0).sum())
        if caught:
            first = np.argmax(win > win[:, :1], axis=1)      # первый час окна с тревогой
            lead = float(np.median((horizon - first + 1)[got > 0]))
        else:
            lead = float('nan')
        sig, true = signals_fast(ctx['oo'], ctx['oh'], ctx['oy'], a[oi, hi][ctx['order']], gap)
        pts.append((caught, sig - true, lead))
    return pts


def prepare(run, part, year, tp, model, horizon):
    run, _, m = run.partition('/')      # `прогон/семейство`, иначе семейство из --model
    obj, h, nxt, p = op.split(run, part, year, tp, m or model)
    grid, objs, oi, hi, h0 = dense(obj, h, p, horizon)
    y = (nxt <= horizon).astype(np.float64)
    order = np.lexsort((h, obj))
    eps = metrics.onsets(obj, h, nxt, metrics.RUN_CAP)
    seen = set(zip(obj.tolist(), h.tolist()))
    keep = [(o, e) for o, e in eps if any((o, e - k) in seen for k in range(1, horizon + 1))]
    eo = np.searchsorted(objs, np.array([o for o, _ in keep]))
    ec = np.array([e for _, e in keep]) - h0
    return {'grid': grid, 'oi': oi, 'hi': hi, 'order': order, 'horizon': horizon, 'y': y,
            'oo': obj[order], 'oh': h[order], 'oy': y[order],
            'eo': eo, 'ec': ec, 'episodes': len(keep)}


def smoothed(ctx: dict, m: int) -> np.ndarray:
    """Сглаживание оценки окном m. Ноль означает среднее сразу по нескольким окнам (раздел 30)."""
    if m == 0:
        return ensemble(ctx, ENS)
    return ctx['grid'] if m <= 1 else rolling_mean(ctx['grid'], m)


def ensemble(ctx: dict, ms=None) -> np.ndarray:
    """Среднее из нескольких окон вместо выбора одного.

    Раздел 30 показал, что слепой выбор окна по проверке стоит 8-16% ложных. Усреднение выбора не
    делает вовсе: оно не может выиграть у лучшего окна, но и промахнуться мимо него не может.
    """
    ms = ms or ENS
    return np.nanmean(np.stack([smoothed(ctx, m) for m in ms]), axis=0)


def point(args) -> None:
    """Рабочая точка: порог снимается на проверке 2025, числа — на тесте 2026.

    Это то же правило, по которому снят порог у сырой оценки, поэтому столбцы сравнимы между
    собой: меняется только то, на что порог ставится. Выбирать порог на тесте нельзя — тогда
    сглаживание получило бы фору, которой в работе не будет.
    """
    smooths = [int(x) for x in args.smooth.split(',')]
    print(f'Прогон {args.run}, модель {args.model}. Порог — по лучшему F1 на проверке 2025, '
          f'числа — на тесте 2026'
          + (f', склейка дребезга {args.gap} ч' if args.gap else '') + '.\n')
    print('| тип | сглаживание | порог | сигналов | ложных | ложных в сутки | поймано эпизодов | упреждение |')
    print('|---|---|---:|---:|---:|---:|---|---:|')
    days = 181
    for tp in config.TYPES:
        try:
            va = prepare(args.run, 'val', 2025, tp, args.model, args.horizon)
            te = prepare(args.run, 'test', 2026, tp, args.model, args.horizon)
        except FileNotFoundError:
            continue
        for m in smooths:
            sv, st = smoothed(va, m), smoothed(te, m)
            rows = sv[va['oi'], va['hi']]
            thr = metrics.best_threshold(va['y'].astype(np.int8), rows)
            c = curve(st, 1, [thr], te, args.gap)[0]
            caught, false, lead = c
            a = alarm_grid(st, thr, 1)
            sig, true = signals_fast(te['oo'], te['oh'], te['oy'], a[te['oi'], te['hi']][te['order']],
                                     args.gap)
            name = 'нет' if m <= 1 else f'{m} ч'
            print(f'| {config.TYPE_NAMES[tp]} | {name} | {thr:.3f} | {sig} | {false} | '
                  f'{false / days:.2f} | {caught} из {te["episodes"]} | '
                  f'{lead:.0f} ч |', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned_r100e20')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--on', default='test', choices=['test', 'val'])
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--gap', type=int, default=0, help='склейка дребезга, раздел 27')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--mode', default='curve', choices=['curve', 'point'],
                    help='curve — сравнение при равной полноте; point — рабочая точка по порогу с проверки')
    ap.add_argument('--smooth', default='1,6,12', help='окна сглаживания для режима point')
    args = ap.parse_args()
    if args.mode == 'point':
        point(args)
        return
    part, year = ('test', 2026) if args.on == 'test' else ('val', 2025)

    names = list(variants(np.zeros((1, 1), np.float32)))
    head = f'Прогон {args.run}, модель {args.model}, ' + ('тест 2026' if args.on == 'test' else 'проверка 2025')
    if args.gap:
        head += f', склейка дребезга {args.gap} ч'
    print(head + '. В ячейках — ложных сигналов при равной доле пойманных эпизодов.\n')
    print('| тип | поймано эпизодов | ' + ' | '.join(names) + ' |')
    print('|---|---|' + '---:|' * len(names))
    leads: dict[str, dict[str, float]] = {}
    for tp in config.TYPES:
        try:
            ctx = prepare(args.run, part, year, tp, args.model, args.horizon)
        except FileNotFoundError:
            continue
        grid = ctx['grid']
        curves = {}
        for name, (score, k) in variants(grid).items():
            flat = score[~np.isnan(score)]
            thr = np.quantile(flat, np.linspace(0.90, 0.99999, args.steps))
            curves[name] = curve(score, k, thr, ctx, args.gap)
        total = ctx['episodes']
        for lv in LEVELS:
            need = lv * total
            cells = []
            for name in names:
                ok = [(f, ld) for c, f, ld in curves[name] if c >= need]
                if ok:
                    best = min(ok)
                    cells.append(str(best[0]))
                    if abs(lv - LEAD_LEVEL) < 1e-9:
                        leads.setdefault(tp, {})[name] = best[1]
                else:
                    cells.append('—')
            print(f'| {config.TYPE_NAMES[tp]} | {lv:.0%} ({int(need)} из {total}) | '
                  + ' | '.join(cells) + ' |', flush=True)
    print(f'\nМедианное упреждение в точке «поймано {LEAD_LEVEL:.0%} эпизодов», часов:\n')
    print('| тип | ' + ' | '.join(names) + ' |')
    print('|---|' + '---:|' * len(names))
    for tp, row in leads.items():
        cells = [f"{row[n]:.0f}" if n in row and row[n] == row[n] else '—' for n in names]
        print(f'| {config.TYPE_NAMES[tp]} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
