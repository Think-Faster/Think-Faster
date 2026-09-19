"""Шаг 14. Уверенность прогноза: по чему видно, что этой тревоге можно верить.

Вероятность сама по себе — плохая мера уверенности: бустинг её не калибрует. Здесь проверяются
пять кандидатов, и каждый — числом «какая доля таких тревог подтвердилась на тесте»:

1. **калиброванная вероятность** — изотоническая регрессия, обученная на проверке 2025;
2. **согласие моделей** — XGBoost и CatBoost подняли тревогу оба или только один;
3. **сколько часов тревога уже горит** — разовый всплеск против устойчивой;
4. **сколько соседних типов тревожатся** — сколько из остальных пяти моделей в тот же час подняли
   тревогу по тому же объекту; несколько прогнозов на одно место против одиночного;
5. **сколько разных каналов объекта сработало** и на сколько пикетов они разошлись
   (из `combo.py spread`) — несколько срабатываний в одной точке против одиночного.

Если доля подтвердившихся растёт вместе с признаком, его можно показывать диспетчеру как
уверенность. Модели не переобучаются: берутся сохранённые прогнозы прогона.

    python confidence.py --run main_h24_tuned
    python confidence.py --run main_h24_tuned --types fire,flood
"""
import argparse
import json

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import config
import metrics
import operating as op

FEAT = config.WORK / 'features'
BINS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.01]
# какие семейства каналов считать «своими» для типа (семейства из combo.py SPREAD)
SP_FAM = {'fire': ['smoke'], 'gas': ['gas'], 'flood': ['flood'], 'sensor': ['fault'],
          'equipment': ['unit_bad', 'phase_off'], 'intrusion': ['fault']}


def table(name: str, rows: list[tuple]) -> None:
    print(f'\n**{name}**\n')
    print('| группа | объекто-часов | доля с инцидентом | сигналов | из них ложных | доля верных |')
    print('|---|---:|---:|---:|---:|---:|')
    for g, n, rate, sig, false in rows:
        good = f'{1 - false / sig:.3f}'.replace('.', ',') if sig else '—'
        print(f'| {g} | {n} | {rate:.3f} | {sig} | {false} | {good} |'.replace('.', ','))


def group_rows(obj, h, y, groups: list[tuple[str, np.ndarray]]) -> list[tuple]:
    out = []
    for g, m in groups:
        if not m.any():
            out.append((g, 0, float('nan'), 0, 0))
            continue
        sig, true = metrics.signals(obj, h, y, m)
        out.append((g, int(m.sum()), float(y[m].mean()), sig, sig - true))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--hours', type=int, default=6, help='порог «тревога горит давно», ч')
    args = ap.parse_args()
    H = args.horizon
    spread = config.WORK / 'combo_spread.parquet'
    sp = pl.read_parquet(spread) if spread.exists() else None

    # тревоги всех шести типов на одной сетке объекто-часов — для блока «соседние типы»
    others = {}
    for tp in config.TYPES:
        _, _, nv, pv = op.split(args.run, 'val', 2025, tp, 'xgb')
        _, _, _, ps = op.split(args.run, 'test', 2026, tp, 'xgb')
        others[tp] = ps >= metrics.best_threshold((nv <= H).astype(np.int8), pv)
    near = {tp: sum(v for k, v in others.items() if k != tp).astype(np.int16) for tp in config.TYPES}

    for tp in args.types.split(','):
        ov, hv, nv, pv = op.split(args.run, 'val', 2025, tp, 'xgb')
        os_, hs, ns, ps = op.split(args.run, 'test', 2026, tp, 'xgb')
        cv = np.load(config.WORK / 'runs' / args.run / 'preds' / f'cat_{tp}_val.npy')
        cs = np.load(config.WORK / 'runs' / args.run / 'preds' / f'cat_{tp}_test.npy')
        yv, ys = (nv <= H).astype(np.int8), (ns <= H).astype(np.int8)
        print(f'\n## {config.TYPE_NAMES[tp]}')

        cal = IsotonicRegression(out_of_bounds='clip').fit(pv, yv)
        q = cal.predict(ps)
        rows = []
        for a, b in zip(BINS, BINS[1:]):
            m = (q >= a) & (q < b)
            rows.append((f'{a:.2f}–{min(b, 1.0):.2f}'.replace('.', ','), m))
        table('калиброванная вероятность', group_rows(os_, hs, ys, rows))

        tv = metrics.best_threshold(yv, pv)
        tc = metrics.best_threshold(yv, cv)
        a, b = ps >= tv, cs >= tc
        table('согласие моделей', group_rows(os_, hs, ys, [
            ('обе модели', a & b), ('только XGBoost', a & ~b), ('только CatBoost', ~a & b)]))

        # сколько часов тревога уже горит подряд к этому часу
        d = (pl.DataFrame({'o': os_, 'h': hs, 'a': a.astype(np.int8)}).sort(['o', 'h'])
             .with_columns((pl.col('a') == 0).cum_sum().over('o').alias('grp'))
             .with_columns(pl.col('a').cum_sum().over(['o', 'grp']).alias('run')))
        run = d.sort(['o', 'h'])['run'].to_numpy()
        order = np.lexsort((hs, os_))
        back = np.empty(len(run), np.int64)
        back[order] = np.arange(len(run))
        run = run[back]
        table(f'сколько часов тревога уже горит', group_rows(os_, hs, ys, [
            ('первый час', a & (run == 1)), (f'2–{args.hours} ч', a & (run > 1) & (run <= args.hours)),
            (f'дольше {args.hours} ч', a & (run > args.hours))]))

        n = near[tp]
        table('сколько соседних типов тревожатся в тот же час', group_rows(os_, hs, ys, [
            ('ни одного', a & (n == 0)), ('один', a & (n == 1)), ('два и больше', a & (n >= 2))]))

        # итоговая уверенность: сколько признаков из четырёх сошлось на этой тревоге
        score = ((q >= 0.2).astype(np.int8) + (a & b).astype(np.int8)
                 + (run > args.hours).astype(np.int8) + (n >= 1).astype(np.int8))
        table('уверенность: сколько признаков из четырёх сошлось', group_rows(os_, hs, ys, [
            (str(k), a & (score == k)) for k in range(5)]))

        cols = [f'sp_{f}_nch_24h' for f in SP_FAM[tp]] if sp is not None else []
        cols = [c for c in cols if c in sp.columns] if sp is not None else []
        if cols:
            j = (pl.DataFrame({'object_id': os_, 'h': hs}).join(sp, on=['object_id', 'h'], how='left')
                 .fill_null(0).select(cols).max_horizontal().to_numpy())
            table(f"сколько каналов «{'/'.join(SP_FAM[tp])}» сработало за 24 ч",
                  group_rows(os_, hs, ys, [
                      ('ни одного', a & (j == 0)), ('один', a & (j == 1)),
                      ('2–4', a & (j >= 2) & (j <= 4)), ('5 и больше', a & (j >= 5))]))


if __name__ == '__main__':
    main()
