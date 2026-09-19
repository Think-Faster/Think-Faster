"""Шаг 15. Работы на объекте: что из размеченного — авария, а что проверка, и когда молчать.

Две задачи в одном скрипте:

- `--mode split` — паттерн: эпизоды заметно чаще начинаются вскоре после прихода людей. Скрипт
  делит эпизоды на «при персонале» и «без персонала» и считает, какую из групп модель на самом
  деле предсказывает.
- `--mode suppress` — цена молчания: если не выдавать алерты туда, где работает бригада, сколько
  ложных сигналов уходит и сколько настоящих эпизодов мы при этом теряем.

Отметки «идут работы» в данных нет — её надо заводить в систему из нарядов и заявок на ТО. Пока её
нет, работы опознаются косвенно: объект снят с охраны или визит начался в последние часы.


Фон — 8,0% объекто-часов попадают в два часа после начала визита, а у загазованности такую долю
имеют 36,9% эпизодов, у пожара 22,2%. Похоже, что часть размеченных инцидентов — это не аварии,
а сработки при обслуживании: сварка и пыль дают дым, продувка даёт газ, снятая крышка даёт
неисправность. Если вся точность модели держится на таких эпизодах — разметку надо чистить; если
группы предсказываются одинаково — паттерн есть, но на прогноз он не влияет.

Модели не переобучаются, берутся сохранённые прогнозы прогона.

    python maintenance.py --run main_h24_tuned --window 2
    python maintenance.py --mode suppress --window 4
"""
import argparse

import duckdb
import numpy as np
import polars as pl

import config
import metrics
import operating as op
from features import T0, hour_of


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


def at_work(con, obj: np.ndarray, h: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Признаки «на объекте сейчас работают» для каждой строки прогноза.

    Настоящей отметки о работах в данных нет, она должна прийти из нарядов и заявок на ТО. Пока её
    нет, работы видно косвенно: объект снят с охраны (значит, кто-то внутри) или недавно начался
    визит (снятие охраны → дверь → движение).
    """
    rows_df = pl.DataFrame({'object_id': obj, 'h': h.astype(np.int64)}).to_arrow()
    con.register('rows_df', rows_df)
    d = con.sql(f"""
        WITH r AS (SELECT object_id, h, TIMESTAMP '{T0}' + h * INTERVAL 1 HOUR AS ts FROM rows_df),
             o AS (SELECT object_id, collector_id FROM obj3),
             g AS (SELECT r.object_id, r.h, r.ts, o.collector_id FROM r JOIN o USING (object_id)),
             a AS (SELECT g.*, gu.armed FROM g ASOF LEFT JOIN guard gu
                   ON gu.object_id = g.object_id AND gu.ts <= g.ts),
             v AS (SELECT a.*, vi.t0 AS vt FROM a ASOF LEFT JOIN visit vi
                   ON vi.collector_id = a.collector_id AND vi.t0 <= a.ts)
        SELECT object_id, h,
               coalesce(NOT armed, false) AS disarmed,
               coalesce(vt IS NOT NULL AND ts - vt < INTERVAL {window} HOUR, false) AS visiting
        FROM v ORDER BY object_id, h""").fetchnumpy()
    order = np.lexsort((h, obj))
    back = np.empty(len(h), np.int64)
    back[order] = np.arange(len(h))
    return {'снят с охраны': d['disarmed'][back].astype(bool),
            f'визит < {window} ч': d['visiting'][back].astype(bool)}


def shown_signals(obj, h, y, alarm, mask) -> tuple[int, int]:
    """Сигналы, которые диспетчер всё-таки увидит, и сколько из них ложные.

    Сигнал — подряд идущие часы тревоги на объекте. Молчание не дробит сигнал на части: если хотя бы
    один его час не погашен, диспетчер этот сигнал увидит целиком; если погашены все — не увидит.
    """
    if not alarm.any():
        return 0, 0
    o, hh, yy, mm = obj[alarm], h[alarm], y[alarm], mask[alarm]
    order = np.lexsort((hh, o))
    o, hh, yy, mm = o[order], hh[order], yy[order], mm[order]
    start = np.empty(len(o), bool)
    start[0] = True
    start[1:] = (o[1:] != o[:-1]) | (hh[1:] != hh[:-1] + 1)
    run = np.cumsum(start) - 1
    n = int(run[-1]) + 1
    true = np.bincount(run, weights=yy, minlength=n) > 0
    muted = np.bincount(run, weights=~mm, minlength=n) == 0     # погашены все часы сигнала
    shown = ~muted
    return int(shown.sum()), int((shown & ~true).sum())


def suppress(con, args) -> None:
    """Цена молчания: сколько ложных сигналов уходит и сколько эпизодов теряется."""
    H, cap = args.horizon, metrics.RUN_CAP
    print(f'Алерты гасятся там, где на объекте работают. Тест 2026, прогон {args.run}, '
          f'модель {args.model}, порог по лучшему F1 на проверке.\n')
    print('| тип | правило молчания | сигналов | из них ложных | доля верных | '
          'поймано эпизодов | часов погашено |')
    print('|---|---|---:|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        _, _, nv, pv = op.split(args.run, 'val', 2025, tp, args.model)
        obj, h, ns, ps = op.split(args.run, 'test', 2026, tp, args.model)
        thr = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        y, alarm = (ns <= H).astype(np.int8), ps >= thr
        flags = at_work(con, obj, h, args.window)
        flags['любое из двух'] = flags['снят с охраны'] | flags[f'визит < {args.window} ч']
        name = config.TYPE_NAMES[tp]
        for rule, mask in [('без молчания', np.zeros(len(h), bool))] + list(flags.items()):
            sig, false = shown_signals(obj, h, y, alarm, mask)
            m = metrics.evaluate(obj, h, ns, np.where(mask, 0.0, ps), thr, H, cap)
            mute = float((alarm & mask).sum() / max(int(alarm.sum()), 1))
            good = 1 - false / sig if sig else float('nan')
            print((f"| {name} | {rule} | {sig} | {false} | {good:.3f} | "
                   f"{m['caught']} из {m['episodes']} | {mute:.3f} |").replace('.', ','))


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
    ap.add_argument('--mode', default='split', choices=['split', 'suppress'])
    args = ap.parse_args()
    H = args.horizon
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)
    if args.mode == 'suppress':
        suppress(con, args)
        con.close()
        return

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
