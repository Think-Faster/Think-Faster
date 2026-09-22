"""Шаг 18. Вторая ступень: модель, которая пересматривает уже поднятую тревогу.

Основная модель смотрит на все объекто-часы сразу, и её порог — компромисс на всём распределении.
Но задача «не пропустить» и задача «не поднять зря» разные, и решать их одной моделью не обязательно.
В промышленных системах это называют двухступенчатым подтверждением: первая ступень отбирает
кандидатов с запасом по полноте, вторая — работает только на них и отвечает на другой вопрос:
«этой конкретной тревоге верить?».

Вторая ступень видит то, чего у первой нет в момент решения по одной строке: сколько часов тревога
уже горит, что говорит вторая модель, тревожатся ли соседние типы, сколько разных каналов объекта
сработало, идут ли на объекте работы. Те самые признаки, которые в `confidence.py` оказались
связаны с долей подтвердившихся тревог, — здесь они не показываются диспетчеру, а становятся
входом модели.

Обучается на тревогах проверки 2025, применяется к тревогам теста 2026. Первая ступень не
переобучается: берутся сохранённые прогоны. Рабочая точка второй ступени задаётся не порогом,
а долей истинных тревог, которую мы согласны сохранить, — и для каждой доли считается, сколько
ложных сигналов уходит и сколько эпизодов теряется.

    python cascade.py --run main_h24_tuned
    python cascade.py --types fire,equipment --keep 0.95,0.9,0.8
    python cascade.py --scale insample          # отброшенный вариант выбора рабочей точки
    python cascade.py --run "$MIX5" --gap 6     # на смеси раздела 34 (раздел 39)

На смеси (`тип~прогоны`) первая ступень — оценка смеси на шкале проверки 2025 (`operating.load_mix`,
`ref`), порог — доля часов под тревогой по типу из раздела 38, снятая на своём периоде, как там.
Вместо вероятностей двух семейств второй ступени даётся сама оценка смеси, «соседи» — тревоги
остальных типов при их долях.
"""
import argparse
import json

import duckdb
import numpy as np
import polars as pl

import config
import maintenance as mt
import metrics
import operating as op

# склейка дребезга: ставится из --gap один раз и читается там, где считаются сигналы
GAP = [0]

FEAT = config.WORK / 'features'

# Доли часов под тревогой по типам на бюджете 80 тыс. ложных часов (раздел 38)
SHARES = {'fire': 0.030, 'gas': 0.026, 'flood': 0.022, 'equipment': 0.097, 'sensor': 0.026,
          'intrusion': 0.005}


def first_stage(run, name, year, tp):
    """Оценка первой ступени и порог: у смеси — доля часов, у прогона — лучший F1 на проверке."""
    if '~' in run:
        obj, h, nxt, p = op.load_mix(run, name, year, tp, 'xgb', ref=('val', 2025))
        return obj, h, nxt, p, float(np.quantile(p, 1 - SHARES[tp]))
    obj, h, nxt, p = op.split(run, name, year, tp, 'xgb')
    _, _, nv, pv = op.split(run, 'val', 2025, tp, 'xgb')
    return obj, h, nxt, p, metrics.best_threshold((nv <= config.HORIZON).astype(np.int8), pv)


def runlen(obj: np.ndarray, h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Сколько часов тревога горит подряд к этому часу (0 — тревоги нет)."""
    d = (pl.DataFrame({'o': obj, 'h': h, 'a': a.astype(np.int8)}).sort(['o', 'h'])
         .with_columns((pl.col('a') == 0).cum_sum().over('o').alias('g'))
         .with_columns(pl.col('a').cum_sum().over(['o', 'g']).alias('r')))
    r = d['r'].to_numpy()
    order = np.lexsort((h, obj))
    back = np.empty(len(r), np.int64)
    back[order] = np.arange(len(r))
    return r[back].astype(np.float32)


def extras(con, run, name, year, tp, obj, h, feats_sp) -> dict[str, np.ndarray]:
    """Признаки второй ступени поверх прогноза первой."""
    H = config.HORIZON
    out = {}
    if '~' in run:
        out['p_mix'] = first_stage(run, name, year, tp)[3].astype(np.float32)
    else:
        for m in ('xgb', 'cat'):
            out[f'p_{m}'] = np.load(config.WORK / 'runs' / run / 'preds' / f'{m}_{tp}_{name}.npy')
    near = np.zeros(len(h), np.float32)
    for other in config.TYPES:
        if other == tp:
            continue
        _, _, _, ps, t = first_stage(run, name, year, other)
        near += (ps >= t).astype(np.float32)
    out['near'] = near
    if feats_sp is not None:
        j = (pl.DataFrame({'object_id': obj, 'h': h}).join(feats_sp, on=['object_id', 'h'], how='left')
             .fill_null(0))
        for c in feats_sp.columns:
            if c not in ('object_id', 'h'):
                out[f'sp_{c}'] = j[c].to_numpy().astype(np.float32)
    work = mt.at_work(con, obj, h, 2)
    out['disarmed'], out['visit_2h'] = [v.astype(np.float32) for v in work.values()]
    return out


def stricter(obj, h, y, nxt, p, thr0, target_sig, H):
    """Контроль: тот же результат, но простым подъёмом порога первой ступени.

    Без этого сравнения вторая ступень ничего не доказывает — убрать ложные можно и молча, подняв
    порог. Порог подбирается так, чтобы сигналов вышло столько же, сколько дала вторая ступень.
    """
    # только вверх от рабочего порога: вниз число серий тоже падает (тревога сливается в одну
    # непрерывную на весь год), и сравнение выходит бессмысленным
    above = p[p >= thr0]
    best = None
    for q in np.linspace(0.0, 0.995, 200):
        t = float(np.quantile(above, q)) if len(above) else 1.1
        sig, true = metrics.signals(obj, h, y, p >= t, GAP[0])
        if best is None or abs(sig - target_sig) < abs(best[0] - target_sig):
            best = (sig, sig - true, t)
    sig, false, t = best
    m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
    return sig, false, m['caught']


def mart(year: int, rows: np.ndarray, feats: list[str]) -> np.ndarray:
    """Признаки витрины только для нужных строк — целый год в память не поднимается."""
    idx = pl.Series('i', rows.astype(np.uint32))
    df = (pl.scan_parquet(FEAT / f'{year}.parquet').with_row_index('i')
          .filter(pl.col('i').is_in(idx)).select(feats).collect())
    return np.ascontiguousarray(df.to_numpy(), dtype=np.float32)


def oof(X: np.ndarray, y: np.ndarray, cols: list[str], p: dict, folds: int) -> np.ndarray:
    """Оценки второй ступени, полученные вне собственного обучения.

    Рабочая точка задаётся квантилью оценок истинных тревог: «сохранить 0,95» — это порог, ниже
    которого остаются 5% истинных. Брать эту квантиль по тем же тревогам, на которых модель
    училась, нельзя: своим обучающим примерам она ставит оценки выше, чем чужим, порог уезжает
    вверх, и на тесте гасится больше истинных, чем обещано. Поэтому шкала строится по блокам:
    модель учится на четырёх пятых тревог и оценивает оставшуюся пятую.

    Блоки идут подряд по времени, а не вперемешку: тревоги одного объекта в соседние часы почти
    одинаковы, и при случайном разбиении половина серии оказалась бы в обучении, а половина в
    оценке — это то же подглядывание, только незаметное.
    """
    import xgboost as xgb
    out = np.zeros(len(y), np.float32)
    bounds = np.linspace(0, len(y), folds + 1).astype(int)
    for a, b_ in zip(bounds[:-1], bounds[1:]):
        te = np.zeros(len(y), bool)
        te[a:b_] = True
        if y[~te].sum() < 5 or not te.any():
            out[te] = np.nan
            continue
        d = xgb.DMatrix(X[~te], y[~te], feature_names=cols)
        m = xgb.train(p, d, num_boost_round=300, verbose_eval=False)
        out[te] = m.predict(xgb.DMatrix(X[te], feature_names=cols))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--keep', default='1.0,0.95,0.9,0.8,0.7',
                    help='какую долю истинных тревог проверки сохраняем')
    ap.add_argument('--scale', default='oof', choices=['oof', 'insample'],
                    help='по каким оценкам выбирать рабочую точку: вне обучения или по обучающим')
    ap.add_argument('--folds', type=int, default=5, help='блоков для оценок вне обучения')
    ap.add_argument('--gap', type=int, default=0,
                    help='склейка дребезга: повтор на том же объекте в пределах gap '
                         'часов — продолжение прежней тревоги, а не новая (раздел 27)')
    args = ap.parse_args()
    GAP[0] = args.gap
    import xgboost as xgb
    H = args.horizon
    feats = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))['features']
    spread = config.WORK / 'combo_spread.parquet'
    sp = pl.read_parquet(spread) if spread.exists() else None
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)

    how = ('по оценкам вне обучения, блоками по времени' if args.scale == 'oof'
           else 'по обучающим оценкам — вариант с подглядыванием, оставлен для сверки')
    first = ('доля часов по типу из раздела 38' if '~' in args.run else 'по лучшему F1 на проверке')
    print(f'Вторая ступень обучена на тревогах проверки 2025, применена к тесту 2026, прогон '
          f'{args.run}. Порог первой ступени — {first}. '
          f'Рабочая точка выбрана {how}.\n')
    print('| тип | рабочая точка | сигналов | из них ложных | доля верных | поймано эпизодов |')
    print('|---|---|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        ov, hv, nv, pv, thv = first_stage(args.run, 'val', 2025, tp)
        os_, hs, ns, ps, thr = first_stage(args.run, 'test', 2026, tp)
        av, as_ = pv >= thv, ps >= thr
        yv, ys = (nv <= H).astype(np.int8), (ns <= H).astype(np.int8)
        name = config.TYPE_NAMES[tp]
        if av.sum() < 200 or as_.sum() < 10:
            print(f'| {name} | тревог слишком мало ({int(av.sum())} / {int(as_.sum())}) | | | | |')
            continue
        ev = extras(con, args.run, 'val', 2025, tp, ov, hv, sp)
        es = extras(con, args.run, 'test', 2026, tp, os_, hs, sp)
        ev['run'], es['run'] = runlen(ov, hv, av), runlen(os_, hs, as_)
        iv, is_ = np.where(av)[0], np.where(as_)[0]
        Xv = np.hstack([mart(2025, iv, feats), np.stack([v[iv] for v in ev.values()], 1)])
        Xs = np.hstack([mart(2026, is_, feats), np.stack([v[is_] for v in es.values()], 1)])
        cols = feats + list(ev.keys())
        # маленькая модель: тревог тысячи, а не миллионы — глубокая переобучится на год проверки
        p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist',
             'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6,
             'min_child_weight': 20, 'reg_lambda': 5.0}
        d = xgb.DMatrix(Xv, yv[iv], feature_names=cols)
        b = xgb.train(p, d, num_boost_round=300, verbose_eval=False)
        if args.scale == 'oof':
            o = np.argsort(hv[iv], kind='stable')      # блоки должны идти подряд по времени
            qv, yq = oof(Xv[o], yv[iv][o], cols, p, args.folds), yv[iv][o]
        else:
            qv, yq = b.predict(d), yv[iv]
        qs = b.predict(xgb.DMatrix(Xs, feature_names=cols))

        base = metrics.evaluate(os_, hs, ns, ps, thr, H, metrics.RUN_CAP)
        sig, true = metrics.signals(os_, hs, ys, as_, GAP[0])
        print(f'| {name} | без второй ступени | {sig} | {sig - true} | '
              f"{1 - (sig - true) / sig:.3f} | {base['caught']} из {base['episodes']} |"
              .replace('.', ','))
        pos = qv[(yq == 1) & np.isfinite(qv)]
        for keep in [float(x) for x in args.keep.split(',')]:
            t2 = float(np.quantile(pos, 1 - keep)) if keep < 1 else float(pos.min())
            mask = np.zeros(len(as_), bool)
            mask[is_] = qs >= t2
            # гасятся часы, а сигнал — серия часов: считаем так же, как в maintenance.py, иначе
            # погашенная середина серии разрежет один сигнал на два и их станет больше, а не меньше
            sig2, false2 = mt.shown_signals(os_, hs, ys, as_, as_ & ~mask, GAP[0])
            true2 = sig2 - false2
            m2 = metrics.evaluate(os_, hs, ns, np.where(mask, ps, 0.0), thr, H, metrics.RUN_CAP)
            good = f'{1 - (sig2 - true2) / sig2:.3f}' if sig2 else '—'
            print(f'| {name} | сохранить {keep:.2f} истинных | {sig2} | {sig2 - true2} | {good} | '
                  f"{m2['caught']} из {m2['episodes']} |".replace('.', ','))
            if keep < 1:
                s3, f3, c3 = stricter(os_, hs, ys, ns, ps, thr, sig2, H)
                g3 = f'{1 - f3 / s3:.3f}' if s3 else '—'
                print(f'| {name} | …то же числом сигналов, но просто порогом выше | {s3} | {f3} | '
                      f"{g3} | {c3} из {base['episodes']} |".replace('.', ','))
        imp = sorted(b.get_score(importance_type='gain').items(), key=lambda kv: -kv[1])[:5]
        print(f"| {name} | *что смотрит вторая ступень* | "
              f"{', '.join(k for k, _ in imp)} | | | |")
    con.close()


if __name__ == '__main__':
    main()
