"""Шаг 15. Сработки при персонале: что из размеченного — авария, а что проверка.

Наблюдение: эпизоды заметно чаще начинаются вскоре после прихода людей на объект. Фон — 8,0%
объекто-часов попадают в два часа после начала визита, а у загазованности такую долю имеют 36,9%
эпизодов, у пожара 22,2%. Похоже, что часть размеченных инцидентов — это не аварии, а сработки при
обслуживании: сварка и пыль дают дым, продувка даёт газ, снятая крышка даёт неисправность.

Скрипт делит эпизоды на две группы по тому, был ли визит в окне перед началом, и считает по
сохранённым прогнозам, какую из групп модель на самом деле предсказывает. Если вся точность держится
на сработках при персонале — разметку надо чистить; если группы предсказываются одинаково — паттерн
есть, но на прогноз он не влияет, и трогать разметку незачем.

Модели не переобучаются.

    python maintenance.py --run main_h24_tuned --window 2
"""
import argparse

import duckdb
import numpy as np

import config
import metrics
import operating as op
from features import hour_of


def onset_hours(con, tp: str, window: int) -> tuple[np.ndarray, np.ndarray]:
    """(объект, час начала) всех эпизодов типа и признак «визит начался в окне перед началом»."""
    d = con.sql(f"""
        WITH e AS (SELECT object_id, collector_id, t0 FROM inc
                   WHERE type = '{tp}' AND noise IS NULL
                     AND object_id IN (SELECT object_id FROM obj3)),
             j AS (SELECT e.object_id, e.t0, v.t0 AS vt FROM e ASOF LEFT JOIN visit v
                   ON v.collector_id = e.collector_id AND v.t0 <= e.t0)
        SELECT object_id, {hour_of('t0')} AS h,
               coalesce(vt IS NOT NULL AND t0 - vt < INTERVAL {window} HOUR, false) AS at_visit
        FROM j""").fetchnumpy()
    return (np.stack([d['object_id'], d['h']]).T, d['at_visit'].astype(bool))


def target(obj: np.ndarray, h: np.ndarray, eps: np.ndarray, H: int) -> np.ndarray:
    """y[i]=1, если эпизод из eps начинается в часах h+1 … h+H на том же объекте."""
    key = obj.astype(np.int64) * (1 << 32)
    have = set((int(o) * (1 << 32) + int(e)) for o, e in eps)
    y = np.zeros(len(obj), np.int8)
    for k in range(1, H + 1):
        cand = key + (h + k)
        y |= np.fromiter((int(c) in have for c in cand), np.int8, len(cand))
    return y


def line(y: np.ndarray, p: np.ndarray, thr: float) -> str:
    alarm = p >= thr
    pr = float(y[alarm].mean()) if alarm.any() else float('nan')
    rc = float(alarm[y == 1].mean()) if y.any() else float('nan')
    ap = float(metrics.average_precision_score(y, p)) if y.any() else float('nan')
    base = float(y.mean())
    return (f'{int(y.sum())} | {base:.4f} | {ap:.3f} | {ap / base:.0f} | {pr:.3f} | {rc:.3f} |'
            ).replace('.', ',')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--window', type=int, default=2, help='окно визита перед началом эпизода, ч')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    args = ap.parse_args()
    H = args.horizon
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)

    print(f'Визит начался не более чем за {args.window} ч до начала эпизода. Тест 2026, '
          f'прогон {args.run}, модель {args.model}.\n')
    print('| тип | группа эпизодов | строк с 1 | база | PR-AUC | подъём | Precision | Recall (стр.) |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        _, _, nv, pv = op.split(args.run, 'val', 2025, tp, args.model)
        obj, h, ns, ps = op.split(args.run, 'test', 2026, tp, args.model)
        thr = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        eps, at_visit = onset_hours(con, tp, args.window)
        y_all = (ns <= H).astype(np.int8)
        y_vis = target(obj, h, eps[at_visit], H)
        y_own = target(obj, h, eps[~at_visit], H)
        name = config.TYPE_NAMES[tp]
        print(f'| {name} | все | ' + line(y_all, ps, thr))
        print(f'| {name} | при персонале | ' + line(y_vis, ps, thr))
        print(f'| {name} | без персонала | ' + line(y_own, ps, thr))
    con.close()


if __name__ == '__main__':
    main()
