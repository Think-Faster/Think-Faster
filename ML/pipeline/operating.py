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

    python operating.py --run main_h24_tuned --model xgb
    python operating.py --run main_h24_tuned --model cat --budget 5,20 --topk 3,5
"""
import argparse
import json

import numpy as np
import polars as pl

import config
import metrics

FEAT = config.WORK / 'features'


def split(run: str, name: str, year: int, tp: str, model: str, target: str = '') -> tuple:
    idx = np.load(config.WORK / 'runs' / run / 'preds' / f'index_{name}.npz')
    p = np.load(config.WORK / 'runs' / run / 'preds' / f'{model}_{tp}_{name}.npy')
    df = pl.scan_parquet(FEAT / f'{year}.parquet').select(['object_id', 'h', f'next_{tp}{target}']).collect()
    obj, h = idx['object_id'], idx['h']
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--precision', default='0.5,0.7')
    ap.add_argument('--budget', default='10', help='тревог в сутки по всем объектам')
    ap.add_argument('--topk', default='3', help='суточный наряд: сколько объектов в сутки')
    ap.add_argument('--target', default='', choices=['', '_prim'],
                    help='_prim — засчитывать только первичные эпизоды: такого же не было 7 сут')
    args = ap.parse_args()
    H = args.horizon
    cap = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))['next_cap']

    print(f'\nПрогон `{args.run}`, модель {args.model}, горизонт {H} ч. Порог выбран на проверке '
          f'2025, числа — на тесте 2026 (янв–июн).\n')
    print('| тип | правило выбора порога | сигналов | из них ложных | ложных в сутки | '
          'доля верных | Recall (эп.) | часов тревоги в сутки | упреждение, ч |')
    print('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tp in config.TYPES:
        ov, hv, nv, pv = split(args.run, 'val', 2025, tp, args.model, args.target)
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
