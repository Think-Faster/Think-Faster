"""Шаг 12. Рабочая точка: чем платим за каждую пойманную аварию (ТЗ §6).

Порог по лучшему F1 — учебная величина: он считает пропуск и ложную тревогу одинаково дорогими.
У диспетчера это не так. Ложная тревога по пожару — выезд бригады, по отказу датчика — строка в
плане ТО; пропуск затопления — авария, пропуск отказа датчика — ничего страшного. Поэтому порог
здесь выбирается двумя способами, оба — на проверке 2025, на тест переносятся без изменений:

- **под точность**: наибольшая полнота среди порогов, где точность не ниже заданной (0,5 и 0,7);
- **под бюджет**: сколько тревог в сутки по всем 78 объектам диспетчер готов разобрать;
- **суточный наряд**: k объектов с наибольшим риском за сутки — число тревог задано жёстко и
  не зависит от того, как уехало распределение вероятностей за год.

Модели не переобучаются: берутся сохранённые прогнозы прогона (`runs/<тег>/preds`).

**Режим `--match`** сравнивает несколько прогонов **при равной полноте**. Порог по F1 у разных
моделей встаёт в разные точки кривой, и сравнение по нему выходит про порог, а не про модель.
Здесь для каждого прогона перебирается сетка порогов и берётся наименьшее число ложных сигналов
среди точек, где поймано не меньше заданной доли эпизодов теста (раздел 21 аналитики).

    python operating.py --run main_h24_tuned --model xgb
    python operating.py --run main_h24_tuned --model cat --budget 5,20 --topk 3,5
    python operating.py --match main_h24,main_h24_conf     # равная полнота, разные цели обучения
    python operating.py --match main_h24,main_h24+main_h24_conf   # смесь двух целей
    python operating.py --match 'main_h24:0.75+main_h24_conf:0.25'   # …с неравными долями
    python operating.py --match main_h24_r100e20/cat,main_h24_tuned_r100e20/xgb   # семейства
    python operating.py --match main_h24,main_h24_conf --target _conf   # мерить по выездам
"""
import argparse
import json

import numpy as np
import polars as pl

import config
import metrics

FEAT = config.WORK / 'features'


def years(run: str, name: str) -> list:
    """Годы нарезки для прогона. Ветка зашита в начало имени (`main_h24…`, `wide_h24…`), а какие
    годы она отдаёт проверке — знает train.SPLITS. Тест всегда 2026 и от ветки не зависит."""
    import train
    if name == 'test':
        return [2026]
    branch = run.partition('/')[0].split('_')[0]
    return train.SPLITS.get(branch, train.SPLITS['main'])[1]


def split(run: str, name: str, year, tp: str, model: str, target: str = '') -> tuple:
    yy = [year] if isinstance(year, int) else list(year)
    idx = np.load(config.WORK / 'runs' / run / 'preds' / f'index_{name}.npz')
    p = np.load(config.WORK / 'runs' / run / 'preds' / f'{model}_{tp}_{name}.npy')
    df = pl.concat([pl.scan_parquet(FEAT / f'{y}.parquet')
                    .select(['object_id', 'h', f'next_{tp}{target}']) for y in yy]).collect()
    obj, h = idx['object_id'], idx['h']
    own = years(run, name)
    if len(p) != len(df) and set(yy) < set(own):
        # Ветки проверяются на разных годах (wide — 2024 и 2025, main — только 2025). При сравнении
        # берутся годы первого прогона, а из предсказаний другой ветки вырезаются строки этих лет:
        # нарезка идёт по годам подряд, в порядке train.SPLITS.
        sizes = {y: pl.scan_parquet(FEAT / f'{y}.parquet').select(pl.len()).collect().item() for y in own}
        start = np.cumsum([0] + [sizes[y] for y in own])
        keep = np.concatenate([np.arange(start[i], start[i + 1]) for i, y in enumerate(own) if y in yy])
        p, obj, h = p[keep], obj[keep], h[keep]
    assert len(df) == len(p) == len(obj), (len(df), len(p), len(obj))
    assert np.array_equal(df['object_id'].to_numpy(), obj) and np.array_equal(df['h'].to_numpy(), h)
    return obj, h, df[f'next_{tp}{target}'].to_numpy(), p


def daily_top(obj: np.ndarray, h: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    """Суточный наряд: k объектов с наибольшим риском за сутки, тревога горит весь день.

    Порог, снятый с проверки, на тесте уезжает вместе с распределением вероятностей — у подтопления
    он не сработал ни разу. Наряд от этого не зависит: число тревог в сутки задано жёстко.
    """
    d = pl.DataFrame({'o': obj, 'day': h // 24, 'p': p})
    top = (d.group_by(['day', 'o']).agg(pl.col('p').max())
           .filter(pl.col('p').rank('ordinal', descending=True).over('day') <= k))
    sel = set(zip(top['day'].to_list(), top['o'].to_list()))
    return np.array([(dd, oo) in sel for dd, oo in zip((h // 24).tolist(), obj.tolist())], np.float32)


def plural(k: int) -> str:
    return f"{k} объект" + ('' if k % 10 == 1 and k % 100 != 11 else
                            'а' if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14 else 'ов')


def row(obj, h, nx, p, thr, H, cap, days) -> str:
    """Строка таблицы в штуках: сигналов, из них ложных, ложных в сутки, пойманных эпизодов."""
    m = metrics.evaluate(obj, h, nx, p, thr, H, cap)
    if not m['signals']:   # порог с проверки на тесте не сработал ни разу
        return '| 0 | 0 | 0,0 | — | 0,000 | 0 | — |'
    good = m['signals_true'] / m['signals']
    hours = m['alarm_rate'] * len(p) / days
    return (f"| {m['signals']} | {m['signals_false']} | {m['signals_false'] / days:.1f} | "
            f"{good:.3f} | {m['recall_episodes']:.3f} | {hours:.0f} | "
            f"{m['lead_median_h']:.0f} |").replace('.', ',')


def load_mix(name: str, on: str, year: int, tp: str, model: str, label: str = '') -> tuple:
    """Прогон или смесь прогонов: `a+b` — среднее рангов вероятностей.

    Вероятности двух моделей, обученных на разных целях, по величине несопоставимы: у цели
    «только подтверждённые» положительных впятеро меньше, и вероятности систематически ниже.
    Складывать можно только порядок, поэтому смешиваются ранги, а не сами числа.

    Вес пишется через двоеточие: `a:0.75+b:0.25`. Без весов доли равные.

    Семейство пишется через косую черту: `main_h24_r100e20/cat`. Без неё берётся `--model`.
    Это нужно, чтобы сравнивать не только цели обучения, но и семейства между собой: у них
    порог по F1 встаёт в разные точки кривой, и сравнение по нему выходит про порог.
    """
    parts = []
    for part in name.split('+'):
        if part:
            run, _, w = part.partition(':')
            run, _, m = run.partition('/')
            parts.append((run, m or model, float(w) if w else 1.0))
    obj, h, nxt, p = split(parts[0][0], on, year, tp, parts[0][1], label)
    if len(parts) > 1:
        p = parts[0][2] * p.argsort().argsort()
        for run, m, w in parts[1:]:
            _, _, _, q = split(run, on, year, tp, m, label)
            p = p + w * q.argsort().argsort()
    return obj, h, nxt, p.astype(np.float32)


def match(runs: list[str], model: str, H: int, levels: list[float], steps: int,
          on: str = 'test', label: str = '', low: float = 0.90, cost: str = 'sig') -> None:
    """Сколько ложных сигналов стоит одна и та же доля пойманных эпизодов у разных прогонов.

    Точка на кривой выбирается по тому же году, на котором считается, — это подглядывание, и
    оно одинаково выгодно всем прогонам. Чтобы выигрыш не оказался следствием подглядывания,
    тот же расчёт повторяется на проверке (`--on val`): настоящее преимущество держится на обоих.

    `low` — нижний квантиль перебора порогов. При low = 0,90 тревогой может стать не больше
    десятой части часов, и у типа с долей положительных 9% (отказ оборудования) высокая полнота
    в такой перебор не попадает: в таблице появляется прочерк.

    Опускать `low` ниже 0,90 можно только чтобы посмотреть, где проходит эта граница, но не
    чтобы сравнивать прогоны при `cost='sig'`. Ложные там считаются блоками подряд идущих часов
    тревоги, и ниже некоторой доли часов блоки начинают склеиваться между собой: число ложных
    проходит максимум и **падает**, хотя тревога висит всё дольше. У отказа оборудования на
    проверке 2025: 569 ложных при 10% часов, 696 при 20%, 244 при 50% — и в последней точке
    средний сигнал длится 651 час. Граница у каждого типа своя и между периодами уезжает: от 2%
    у пожара до 50% у отказа датчика (раздел 31). Кривая снимается `curve.py`.

    `cost='hours'` считает ту же таблицу в часах ложной тревоги вместо блоков. Часы монотонны по
    порогу по построению — опуская порог, тревожных часов можно только добавить, — поэтому такая
    таблица сравнима в любой точке и слепого пятна у неё нет. Блоки остаются эксплуатационной
    ценой (диспетчер ходит на сигнал), часы — мерой сравнения.
    """
    year = years(runs[0], on)
    print('| тип | поймано эпизодов | ' + ' | '.join(f'`{r}`' for r in runs) + ' |')
    print('|---|---|' + '---:|' * len(runs))
    for tp in config.TYPES:
        curves, total = {}, 0
        for run in runs:
            try:
                obj, h, nxt, p = load_mix(run, on, year, tp, model, label)
            except FileNotFoundError:
                continue
            y = (nxt <= H).astype(np.int8)
            pts = []
            for q in np.linspace(low, 0.99999, steps):
                t = float(np.quantile(p, q))
                a = p >= t
                if not a.any():
                    continue
                sig, true = metrics.signals(obj, h, y, a)
                m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
                # Часы ложной тревоги — часы выше порога, в горизонте которых происшествия нет.
                pts.append((m['caught'], sig - true if cost == 'sig'
                            else int((a & (y == 0)).sum())))
                total = m['episodes']
            curves[run] = pts
        for lv in levels:
            need = lv * total
            cells = [min([f for c, f in curves.get(r, []) if c >= need], default=None)
                     for r in runs]
            cells = ['—' if c is None else str(c) for c in cells]
            print(f'| {config.TYPE_NAMES[tp]} | {lv:.0%} ({int(need)} из {total}) | '
                  + ' | '.join(cells) + ' |')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--precision', default='0.5,0.7')
    ap.add_argument('--budget', default='10', help='тревог в сутки по всем объектам')
    ap.add_argument('--topk', default='3', help='суточный наряд: сколько объектов в сутки')
    ap.add_argument('--match', default='', help='прогоны через запятую: сравнить при равной полноте')
    ap.add_argument('--levels', default='0.4,0.5,0.6,0.7,0.75', help='доли эпизодов для --match')
    ap.add_argument('--steps', type=int, default=70, help='сколько порогов перебрать для --match')
    ap.add_argument('--cost', default='sig', choices=['sig', 'hours'],
                    help='чем мерить ложные: блоками сигналов или часами тревоги (раздел 31)')
    ap.add_argument('--low', type=float, default=0.90,
                    help='нижний квантиль перебора порогов: 0,90 значит «тревога не чаще чем '
                         'в десятой части часов». Типам с частыми происшествиями нужен ниже')
    ap.add_argument('--on', default='test', choices=['test', 'val'], help='год для --match')
    ap.add_argument('--target', default='', choices=['', '_prim', '_conf'],
                    help='какие эпизоды засчитывать за настоящие: _prim — только первичные '
                         '(такого же не было 7 сут), _conf — только те, на которые приехала бригада')
    args = ap.parse_args()
    H = args.horizon
    if args.match:
        match([r for r in args.match.split(',') if r], args.model, H,
              [float(x) for x in args.levels.split(',')], args.steps, args.on, args.target,
              args.low, args.cost)
        return
    cap = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))['next_cap']

    print(f'\nПрогон `{args.run}`, модель {args.model}, горизонт {H} ч. Порог выбран на проверке '
          f'2025, числа — на тесте 2026 (янв–июн).\n')
    print('| тип | правило выбора порога | сигналов | из них ложных | ложных в сутки | '
          'доля верных | Recall (эп.) | часов тревоги в сутки | упреждение, ч |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tp in config.TYPES:
        ov, hv, nv, pv = split(args.run, 'val', years(args.run, 'val'), tp, args.model, args.target)
        os_, hs, ns, ps = split(args.run, 'test', 2026, tp, args.model, args.target)
        yv = (nv <= H).astype(np.int8)
        days_v = (hv.max() - hv.min() + 1) / 24
        days_s = (hs.max() - hs.min() + 1) / 24
        points = [('по лучшему F1', metrics.best_threshold(yv, pv))]
        for t in (float(x) for x in args.precision.split(',') if x):
            points.append((f'точность ≥ {t:.1f}'.replace('.', ','),
                           metrics.threshold_for_precision(yv, pv, t)))
        for b in (float(x) for x in args.budget.split(',') if x):
            points.append((f'{b:.0f} тревог в сутки',
                           metrics.threshold_for_rate(pv, b, len(pv), days_v)))
        for name, thr in points:
            if thr is None:
                print(f'| {config.TYPE_NAMES[tp]} | {name} | недостижима | — | — | — | — | — | — |')
                continue
            print(f'| {config.TYPE_NAMES[tp]} | {name} ' + row(os_, hs, ns, ps, thr, H, cap, days_s))
        for k in (int(x) for x in args.topk.split(',') if x):
            sel = daily_top(os_, hs, ps, k)
            print(f'| {config.TYPE_NAMES[tp]} | наряд: {plural(k)} в сутки '
                  + row(os_, hs, ns, sel, 0.5, H, cap, days_s))


if __name__ == '__main__':
    main()
