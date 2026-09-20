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

**Режим `--mode visit`** отвечает на другой вопрос: а точно ли ложная тревога ложная? «Ложная» здесь
значит только «за ней не пошёл размеченный эпизод». Единственный след реальности, не участвующий в
разметке, — выезд бригады (таблица `visit`). Если за ложными тревогами выезд случается чаще, чем за
случайным часом того же объекта, часть из них попала во что-то настоящее, чего нет в журнале
инцидентов. Если не чаще — ложные ложны и по независимому признаку. Тот же вопрос к самой разметке
задан в разделе 18; здесь он задан прогнозу.

    python confidence.py --run main_h24_tuned
    python confidence.py --run main_h24_tuned --types fire,flood
    python confidence.py --mode visit --window 24
"""
import argparse
import json

import duckdb
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


def spans(obj, h, y, alarm) -> tuple[np.ndarray, ...]:
    """Сигналы как отрезки: объект, последний час тревоги, подтвердился ли эпизодом.

    То же разбиение, что в `metrics.signals`, но наружу отдаются сами отрезки: для проверки
    выездом нужен час, когда диспетчер уже всё видел, то есть конец сигнала.
    """
    o, hh, yy = obj[alarm], h[alarm], y[alarm]
    order = np.lexsort((hh, o))
    o, hh, yy = o[order], hh[order], yy[order]
    start = np.empty(len(o), bool)
    start[0] = True
    start[1:] = (o[1:] != o[:-1]) | (hh[1:] != hh[:-1] + 1)
    run = np.cumsum(start) - 1
    n = int(run[-1]) + 1
    last = np.zeros(n, np.int64)
    np.maximum.at(last, run, hh)
    who = np.zeros(n, np.int64)
    who[run] = o                       # отрезок целиком лежит на одном объекте
    return who, last, np.bincount(run, weights=yy, minlength=n).astype(bool)


def visit_ahead(con, lo: int, hi: int, window: int) -> tuple[dict[int, int], np.ndarray]:
    """Матрица «был ли выезд бригады к этому объекту в ближайшие `window` часов».

    Выезды известны по коллектору, а не по объекту (`visit` в `labels.py`), поэтому на объекты
    одного коллектора выезд ложится одинаково — сигнал грубее, чем хотелось бы, но независимый.
    """
    d = con.sql(f"""
        SELECT o.object_id,
               date_diff('hour', TIMESTAMP '2019-01-01', v.t0) AS h
          FROM visit v JOIN obj3 o USING (collector_id)
         WHERE v.t0 >= TIMESTAMP '2019-01-01' + INTERVAL {lo} HOUR
           AND v.t0 <  TIMESTAMP '2019-01-01' + INTERVAL {hi + window + 1} HOUR""").fetchnumpy()
    objects = [r[0] for r in con.sql('SELECT object_id FROM obj3 ORDER BY object_id').fetchall()]
    idx = {o: i for i, o in enumerate(objects)}
    width = hi - lo + window + 2
    v = np.zeros((len(objects), width), np.int32)
    for o, h in zip(np.ma.getdata(d['object_id']).tolist(), np.ma.getdata(d['h']).tolist()):
        if o in idx and lo <= h < lo + width:
            v[idx[o], h - lo] = 1
    c = np.cumsum(v, axis=1)
    # выезд в часах (h, h + window]: разность накопленных сумм по краям окна
    right = np.minimum(np.arange(width) + window, width - 1)
    return idx, (c[:, right] - c) > 0


def visit_table(name: str, rows: list[tuple]) -> None:
    print(f'\n**{name}**\n')
    print('| группа | сигналов | доля с выездом | фон объекта | отн. | фон объекта и часа | отн. |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for g, n, got, e1, e2 in rows:
        if not n:
            print(f'| {g} | 0 | — | — | — | — | — |')
            continue
        r1 = f'{got / e1:.2f}' if e1 > 0 else '—'
        r2 = f'{got / e2:.2f}' if e2 > 0 else '—'
        print(f'| {g} | {n} | {got:.3f} | {e1:.3f} | {r1} | {e2:.3f} | {r2} |'.replace('.', ','))


def visit_mode(args) -> None:
    """Проверка тревог прогноза независимым следом — выездом бригады."""
    H = args.horizon
    EV, YEAR = args.eval, {'val': 2025, 'test': 2026}[args.eval]
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)
    o0, h0, n0, _ = op.split(args.run, EV, YEAR, config.TYPES[0], 'xgb')
    lo, hi = int(h0.min()), int(h0.max())
    idx, ahead = visit_ahead(con, lo, hi, args.window)
    # положительный контроль: сами эпизоды из журнала. Если выезд не учащается даже после
    # настоящего инцидента, прибор слеп, и «нет превышения» у ложных тревог ничего не значит
    real = {}
    for tp in config.TYPES:
        d = con.sql(f"""
            SELECT object_id, date_diff('hour', TIMESTAMP '2019-01-01', t0) AS h
              FROM inc WHERE type = '{tp}' AND noise IS NULL AND year(t0) = {YEAR}
               AND object_id IN (SELECT object_id FROM obj3)""").fetchnumpy()
        real[tp] = (np.ma.getdata(d['object_id']), np.ma.getdata(d['h']))
    con.close()

    # фон: доля часов, за которыми в ближайшие window часов был выезд. Считается по тем же часам,
    # что и тревоги, — иначе сравнивали бы с другим периодом. Двух фонов два: грубый, по объекту, и
    # сопоставленный — по объекту, часу суток и будням/выходным. Бригада ездит в рабочее время, а
    # тревоги на ночь и выходные приходятся не поровну, так что грубый фон отвечает не на тот вопрос
    col = np.array([idx[o] for o in o0.tolist()])
    row = h0 - lo
    base_all = float(ahead[col, row].mean())
    n_obj, hod = ahead.shape[0], (h0 % 24).astype(np.int64)
    wknd = (((h0 // 24) + 1) % 7 >= 5).astype(np.int64)      # 2019-01-01 — вторник
    cell = (col * 24 + hod) * 2 + wknd
    got = ahead[col, row].astype(np.float64)
    size = n_obj * 24 * 2
    cnt = np.bincount(cell, minlength=size)
    cell_rate = np.divide(np.bincount(cell, weights=got, minlength=size), np.maximum(cnt, 1))
    cell_rate[cnt == 0] = base_all
    per_obj = np.full(n_obj, base_all, np.float64)
    for i in range(n_obj):
        m = col == i
        if m.any():
            per_obj[i] = float(got[m].mean())

    print(f'# Проверка тревог выездом бригады ({EV} {YEAR}, окно {args.window} ч)\n')
    print(f'Прогон `{args.run}`, порог снят на проверке 2025. «Доля с выездом» — доля сигналов, '
          f'за последним часом которых в ближайшие {args.window} ч был выезд к коллектору объекта. '
          f'Рядом два фона: «фон объекта» — та же доля для случайного часа тех же объектов в тех же '
          f'пропорциях, «фон объекта и часа» — для случайного часа тех же объектов в тот же час '
          f'суток и тот же тип дня. Общий фон по всем объектам — {base_all:.3f}, сравнивать с ним '
          f'нельзя: тревоги сидят не на случайных объектах и не в случайные часы.'
          .replace('.', ','))
    def measure(name: str, obj: np.ndarray, hh: np.ndarray) -> tuple:
        if not len(obj):
            return (name, 0, 0.0, 0.0, 0.0)
        ci = np.array([idx[o] for o in obj.tolist()])
        sc = (ci * 24 + hh % 24) * 2 + ((hh // 24 + 1) % 7 >= 5)
        hit = ahead[ci, np.clip(hh - lo, 0, ahead.shape[1] - 1)]
        return (name, len(obj), float(hit.mean()), float(per_obj[ci].mean()),
                float(cell_rate[sc].mean()))

    for tp in args.types.split(','):
        ov, hv, nv, pv = op.split(args.run, 'val', 2025, tp, 'xgb')
        os_, hs, ns, ps = op.split(args.run, EV, YEAR, tp, 'xgb')
        ys = (ns <= H).astype(np.int8)
        ro, rh = real[tp]
        keep = np.isin(ro, list(idx)) & (rh >= lo) & (rh <= hi)
        rows = [measure('эпизоды журнала (контроль)', ro[keep], rh[keep])]
        a = ps >= metrics.best_threshold((nv <= H).astype(np.int8), pv)
        if a.any():
            who, last, true = spans(os_, hs, ys, a)
            rows.append(measure('подтвердившиеся эпизодом', who[true], last[true]))
            rows.append(measure('ложные', who[~true], last[~true]))
        visit_table(config.TYPE_NAMES[tp], rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--hours', type=int, default=6, help='порог «тревога горит давно», ч')
    ap.add_argument('--mode', default='signs', choices=['signs', 'visit'],
                    help='signs — пять признаков уверенности; visit — проверка тревог выездом')
    ap.add_argument('--window', type=int, default=24, help='окно ожидания выезда после сигнала, ч')
    ap.add_argument('--eval', default='test', choices=['test', 'val'],
                    help='на чём считать; val — проверка того, что признаки держатся и на другом годе')
    args = ap.parse_args()
    if args.mode == 'visit':
        visit_mode(args)
        return
    H = args.horizon
    EV, YEAR = args.eval, {'val': 2025, 'test': 2026}[args.eval]
    spread = config.WORK / 'combo_spread.parquet'
    sp = pl.read_parquet(spread) if spread.exists() else None

    # тревоги всех шести типов на одной сетке объекто-часов — для блока «соседние типы»
    others = {}
    for tp in config.TYPES:
        _, _, nv, pv = op.split(args.run, 'val', 2025, tp, 'xgb')
        _, _, _, ps = op.split(args.run, EV, YEAR, tp, 'xgb')
        others[tp] = ps >= metrics.best_threshold((nv <= H).astype(np.int8), pv)
    near = {tp: sum(v for k, v in others.items() if k != tp).astype(np.int16) for tp in config.TYPES}

    for tp in args.types.split(','):
        ov, hv, nv, pv = op.split(args.run, 'val', 2025, tp, 'xgb')
        os_, hs, ns, ps = op.split(args.run, EV, YEAR, tp, 'xgb')
        cv = np.load(config.WORK / 'runs' / args.run / 'preds' / f'cat_{tp}_val.npy')
        cs = np.load(config.WORK / 'runs' / args.run / 'preds' / f'cat_{tp}_{EV}.npy')
        yv, ys = (nv <= H).astype(np.int8), (ns <= H).astype(np.int8)
        print(f'\n## {config.TYPE_NAMES[tp]}')

        cal = IsotonicRegression(out_of_bounds='clip').fit(pv, yv)
        q = cal.predict(ps)
        if EV == 'test':   # на проверке калибровка обучена по этим же строкам, смотреть её незачем
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
