"""Шаг 4. Бустинг по типам инцидентов и базовые уровни (Ф5-1).

Валидация по времени: обучение — 2022–2024 (ветка long добавляет 2019–2020), проверка — 2025
(ранняя остановка, порог, калибровка), тест — 2026 один раз. По каждому типу своя бинарная модель
«начнётся ли эпизод в ближайшие H часов». На GPU — XGBoost и CatBoost, LightGBM — на CPU
(в pip-сборке нет CUDA), его можно пускать параллельно отдельным процессом.

Базовые уровни:
- freq — доля часов с инцидентом у объекта на обучении (частота без признаков);
- rules — правило диспетчера «были триггеры этого типа за последние 24 ч», оценка — их число;
- recency — сколько часов с прошлого эпизода этого типа (чем меньше, тем выше риск).

    python train.py --models xgb,cat
    python train.py --models lgbm --branch long
    python train.py --models xgb,cat --params tuned --weather base   # то же с погодой (weather.py)
    python train.py --models xgb,cat --params tuned --combo pairs   # …и со связками датчиков (combo.py)
    python train.py --models xgb --params tuned --fleet   # …и со следом общей причины (fleet.py)
    python train.py --models xgb --soft 0.5   # мягкая цель: неподтверждённый эпизод весит половину
    python train.py --models xgb --params tuned --rounds 300 --early 40   # короткий бюджет деревьев
    python train.py --models xgb --params tuned --rounds 100 --early 20 --pw 0.3   # ложная дороже пропуска
"""
import argparse
import json
import re
import time

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

import config
import metrics

SPLITS = {'main': [2022, 2023, 2024], 'long': [2019, 2020, 2022, 2023, 2024]}
WEATHER_TAG = {'': '', 'base': '_weather', 'ext': '_weather_ext', 'hum': '_weather_hum',
               'both': '_weather_both'}
COMBO_TAG = {'': '', 'pairs': '_combo', 'spread': '_spread', 'both': '_combo_spread'}

# Что именно модель помнит об объекте (раздел 23). `ident` — статический состав объекта: сколько
# каких датчиков на нём висит. Он не меняется никогда, поэтому работает как имя объекта. `hist` —
# сколько происшествий у объекта уже было и давно ли. Она меняется, но медленно: окно 2160 ч.
DROP = {'': None,
        'ident': r'comp_\d+',
        'hist': r'(onset_\w+_\d+h|since_(' + '|'.join(config.TYPES) + r'))',
        'both': r'(comp_\d+|onset_\w+_\d+h|since_(' + '|'.join(config.TYPES) + r'))'}
DROP_TAG = {'': '', 'ident': '_noident', 'hist': '_nohist', 'both': '_nomem'}
FEAT = config.WORK / 'features'


def load(years: list[int], step: int, columns: list[str], wx: pl.DataFrame | None = None,
         cb: pl.DataFrame | None = None) -> pl.DataFrame:
    lf = pl.scan_parquet([FEAT / f'{y}.parquet' for y in years])
    if step > 1:
        lf = lf.filter(pl.col('h') % step == 0)
    lf = lf.select(columns)
    if wx is not None:  # погода одна на все объекты: приклеивается по часу
        lf = lf.join(wx.lazy(), on='h', how='left')
    if cb is not None:  # связки датчиков: по объекту и часу, где связок не было — нули
        extra = [c for c in cb.columns if c not in ('object_id', 'h')]
        lf = lf.join(cb.lazy(), on=['object_id', 'h'], how='left').with_columns(
            [pl.col(c).fill_null(0) for c in extra])
    return lf.collect()


def matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return np.ascontiguousarray(df.select(features).to_numpy(), dtype=np.float32)


# Бюджет деревьев. Раздел 19 показал, что он сам по себе стоит трети ложных сигналов: лишние
# деревья продолжают голосовать, а ранняя остановка с большим запасом позволяет им это делать.
# Значения переопределяются ключами --rounds/--early; retrain.py держит свой бюджет отдельно.
ROUNDS, EARLY = 4000, 200


def fit_xgb(Xt, yt, Xv, yv, params: dict | None = None):
    import xgboost as xgb
    p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'device': 'cuda',
         'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6,
         'min_child_weight': 5, 'max_bin': 256, 'reg_lambda': 1.0}
    p.update(params or {})
    dt = xgb.QuantileDMatrix(Xt, yt, max_bin=p['max_bin'])
    dv = xgb.QuantileDMatrix(Xv, yv, ref=dt)
    booster = xgb.train(p, dt, num_boost_round=ROUNDS, evals=[(dv, 'val')],
                        early_stopping_rounds=EARLY, verbose_eval=False)
    predict = lambda X: booster.inplace_predict(X, iteration_range=(0, booster.best_iteration + 1))
    return booster, predict, booster.best_iteration + 1


def fit_cat(Xt, yt, Xv, yv, params: dict | None = None):
    from catboost import CatBoostClassifier
    p = {'iterations': ROUNDS, 'learning_rate': 0.05, 'depth': 8, 'task_type': 'GPU', 'devices': '0',
         'loss_function': 'Logloss', 'border_count': 254, 'od_type': 'Iter', 'od_wait': EARLY,
         'use_best_model': True, 'verbose': False, 'gpu_ram_part': 0.8}
    p.update(params or {})
    m = CatBoostClassifier(**p)
    m.fit(Xt, yt, eval_set=(Xv, yv))
    return m, lambda X: m.predict_proba(X)[:, 1], m.get_best_iteration() + 1


def fit_lgbm(Xt, yt, Xv, yv, params: dict | None = None):
    import lightgbm as lgb
    p = {'objective': 'binary', 'metric': 'average_precision', 'learning_rate': 0.05, 'num_leaves': 127,
         'min_child_samples': 50, 'feature_fraction': 0.6, 'bagging_fraction': 0.8, 'bagging_freq': 1,
         'num_threads': 12, 'verbose': -1, 'max_bin': 255}
    p.update(params or {})
    dt = lgb.Dataset(Xt, yt, free_raw_data=True)
    dv = lgb.Dataset(Xv, yv, reference=dt)
    m = lgb.train(p, dt, num_boost_round=ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    return m, lambda X: m.predict(X, num_iteration=m.best_iteration), m.best_iteration


FIT = {'xgb': fit_xgb, 'cat': fit_cat, 'lgbm': fit_lgbm}


def tuned(name: str, tp: str, target: str, horizon: int, budget: str = '') -> dict | None:
    """Параметры из tune.py; для LightGBM подбора нет — берутся умолчания.

    Подбор делался под обычную цель. Для `--target _conf/_prim` своего файла нет, и раньше эта
    функция молча возвращала None: прогон назывался `_tuned`, а обучался на умолчаниях, из-за чего
    `main_h24_conf_tuned` вышел побитово равен `main_h24_conf`. Теперь параметры обычной цели
    переносятся на цель-вариант явно и с отметкой в выводе — это допущение, а не подбор.
    """
    h = '' if horizon == config.HORIZON else f'_h{horizon}'
    d = config.WORK / 'runs' / 'tune'
    path = d / f'{name}_{tp}{target}{budget}{h}.json'
    if not path.exists() and budget:
        # подбора под короткий бюджет нет — берём обычный, но говорим об этом (раздел 24)
        path = d / f'{name}_{tp}{target}{h}.json'
        if path.exists():
            print(f'    {name}/{tp}: своего подбора под бюджет {budget[1:]} нет, '
                  f'взяты параметры полного бюджета')
    if not path.exists() and target:
        path = d / f'{name}_{tp}{h}.json'
        if path.exists():
            print(f'    {name}/{tp}: своего подбора под цель {target} нет, '
                  f'взяты параметры обычной цели')
    if not path.exists():
        return None
    p = json.loads(path.read_text(encoding='utf-8'))['params']
    if name == 'cat' and 'subsample' in p:
        p['bootstrap_type'] = 'Bernoulli'
    return p


def importance(model, name: str, features: list[str], top: int = 20) -> list[tuple[str, float]]:
    if name == 'xgb':
        g = model.get_score(importance_type='total_gain')
        pairs = [(features[int(k[1:])], v) for k, v in g.items()]
    elif name == 'cat':
        pairs = list(zip(features, model.get_feature_importance()))
    else:
        pairs = list(zip(features, model.feature_importance('gain')))
    total = sum(v for _, v in pairs) or 1
    return [(f, round(v / total, 4)) for f, v in sorted(pairs, key=lambda x: -x[1])[:top]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='xgb,cat')
    ap.add_argument('--branch', default='main', choices=list(SPLITS))
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--step', type=int, default=3, help='шаг по часам в обучении: соседние часы почти одинаковы')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--target', default='', choices=['', '_prim', '_conf'],
                    help='_prim — только первичные эпизоды: такого же не было 7 сут; '
                         '_conf — только те, на которые приехала бригада')
    ap.add_argument('--soft', type=float, default=1.0,
                    help='вес неподтверждённых выездом эпизодов в обучении: 1 — обычная цель, '
                         '0 — как --target _conf, промежуточные — мягкая метка (раздел 22)')
    ap.add_argument('--params', default='default', choices=['default', 'tuned'])
    ap.add_argument('--weather', default='', choices=['', 'base', 'ext', 'hum', 'both'],
                    help='добавить признаки погоды из weather.py: набор ТЗ, расширенный или оба')
    ap.add_argument('--combo', default='', choices=['', 'pairs', 'spread', 'both'],
                    help='добавить связки соседних датчиков и/или разброс по пикетам из combo.py')
    ap.add_argument('--fleet', action='store_true',
                    help='добавить след общей причины из fleet.py: что в этот час творится во '
                         'всём парке и на скольких объектах сразу (раздел 27)')
    ap.add_argument('--drop', default='', choices=list(DROP),
                    help='убрать память об объекте: ident — статический состав, hist — история '
                         'происшествий объекта, both — и то и другое (раздел 23)')
    ap.add_argument('--rounds', type=int, default=ROUNDS, help='предел числа деревьев (раздел 19)')
    ap.add_argument('--early', type=int, default=EARLY, help='запас ранней остановки (раздел 19)')
    ap.add_argument('--note', default='',
                    help='приписка к имени прогона: те же ключи, но другой подбор параметров — '
                         'иначе прогон лёг бы поверх прежнего и затёр базу, на которой считаны '
                         'разделы 15 и 16')
    ap.add_argument('--pw', type=float, default=1.0,
                    help='вес положительного класса: <1 делает пропуск дешевле ложной тревоги '
                         '(cost-sensitive learning, раздел 26)')
    args = ap.parse_args()
    globals()['ROUNDS'], globals()['EARLY'] = args.rounds, args.early
    models = [m for m in args.models.split(',') if m]
    types = args.types.split(',')
    H = args.horizon
    tg = args.target
    soft = args.soft
    assert soft == 1.0 or not tg, '--soft задаёт вес внутри обычной цели, с --target не сочетается'
    budget = '' if (args.rounds, args.early) == (4000, 200) else f'_r{args.rounds}e{args.early}'
    pw = '' if args.pw == 1.0 else f'_pw{int(round(args.pw * 100)):03d}'
    tag = (f'{args.branch}_h{H}{tg}' + ('' if soft == 1.0 else f'_soft{int(round(soft * 100)):02d}')
           + ('_tuned' if args.params == 'tuned' else '')
           + WEATHER_TAG[args.weather] + ('_fleet' if args.fleet else '')
           + COMBO_TAG[args.combo] + DROP_TAG[args.drop] + budget + pw
           + (f'_{args.note}' if args.note else ''))

    meta = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))
    features, cap = meta['features'], meta['next_cap']
    wx = cb = None
    if args.weather:
        for pack in (['base', 'ext'] if args.weather == 'both' else [args.weather]):
            w = pl.read_parquet(config.WORK / ('weather.parquet' if pack == 'base' else f'weather_{pack}.parquet'))
            wx = w if wx is None else wx.join(w, on='h')
        features = features + [c for c in wx.columns if c != 'h']
    if args.fleet:
        # ряд парка join-ится по часу так же, как погода: он один на все объекты (см. fleet.py)
        w = pl.read_parquet(config.WORK / 'fleet.parquet')
        wx = w if wx is None else wx.join(w, on='h')
        features = features + [c for c in w.columns if c != 'h']
    if args.combo:
        for part in (['pairs', 'spread'] if args.combo == 'both' else [args.combo]):
            c = pl.read_parquet(config.WORK / ('combo.parquet' if part == 'pairs' else 'combo_spread.parquet'))
            cb = c if cb is None else cb.join(c, on=['object_id', 'h'], how='full', coalesce=True).fill_null(0)
        features = features + [c for c in cb.columns if c not in ('object_id', 'h')]
    if args.drop:
        pat = re.compile(DROP[args.drop])
        gone = [c for c in features if pat.fullmatch(c)]
        features = [c for c in features if not pat.fullmatch(c)]
        print(f'убрано признаков: {len(gone)} из {len(gone) + len(features)} '
              f'({", ".join(gone[:4])}…)')
    t = time.time()
    keys = ['object_id', 'h']
    cols = keys + meta['features'] + meta['targets']
    tr = load(SPLITS[args.branch], args.step, cols, wx, cb)
    va = load([2025], 1, cols, wx, cb)
    te = load([2026], 1, cols, wx, cb)
    Xt, Xv, Xs = matrix(tr, features), matrix(va, features), matrix(te, features)
    tr, va, te = (df.select(keys + meta['targets']) for df in (tr, va, te))  # признаки уже в матрицах
    print(f'обучение {Xt.shape}, проверка {Xv.shape}, тест {Xs.shape}: {time.time() - t:.0f} с', flush=True)

    out_dir = config.WORK / 'runs' / tag
    (out_dir / 'preds').mkdir(parents=True, exist_ok=True)
    (out_dir / 'models').mkdir(parents=True, exist_ok=True)
    for name, df in (('val', va), ('test', te)):
        np.savez(out_dir / 'preds' / f'index_{name}.npz', object_id=df['object_id'].to_numpy(), h=df['h'].to_numpy())
    fi = {f: i for i, f in enumerate(features)}
    h1 = va['h'].to_numpy() < va['h'].min() + 181 * 24
    report_path = out_dir / f'report_{"_".join(models)}.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}

    for tp in types:
        yt = (tr[f'next_{tp}{tg}'].to_numpy() <= H).astype(np.float32)
        yv = (va[f'next_{tp}{tg}'].to_numpy() <= H).astype(np.float32)
        if soft != 1.0:
            # мягкая метка: подтверждённый выездом эпизод — 1, остальные — soft. Проверка остаётся
            # обычной: ранняя остановка и порог считаются по той же разметке, по которой идёт оценка
            conf = (tr[f'next_{tp}_conf'].to_numpy() <= H)
            yt = np.where(yt > 0, np.where(conf, 1.0, soft), 0.0).astype(np.float32)
        print(f'\n== {tp}: доля положительных train {yt.mean():.4f}, val {yv.mean():.4f}', flush=True)
        scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # базовые уровни
        rate = tr.group_by('object_id').agg(pl.col(f'next_{tp}{tg}').le(H).mean().alias('r'))
        rmap = dict(zip(rate['object_id'].to_list(), rate['r'].to_list()))
        prior = float(yt.mean())
        scores['freq'] = tuple(np.array([rmap.get(o, prior) for o in df['object_id'].to_list()], np.float32)
                               for df in (va, te))
        # --drop может унести признак, на котором стоит базовый уровень: при снятой истории
        # происшествий `since_<тип>` в матрице нет. Уровень тогда просто не считается — сравнивать
        # модель всё равно надо с теми уровнями, которым признаки оставлены (раздел 23).
        if f'trig_{tp}_24h' in fi:
            scores['rules'] = (Xv[:, fi[f'trig_{tp}_24h']], Xs[:, fi[f'trig_{tp}_24h']])
        if f'since_{tp}' in fi:
            scores['recency'] = (-Xv[:, fi[f'since_{tp}']], -Xs[:, fi[f'since_{tp}']])
        for name in models:
            t1 = time.time()
            par = tuned(name, tp, tg, H, budget) if args.params == 'tuned' else None
            if args.pw != 1.0 and name in ('xgb', 'cat'):
                # Цена ошибки несимметрична: у диспетчера ложный выезд дороже, чем узнать
                # о происшествии не за сутки, а в момент, — канал «по факту» всё равно объявит
                # (раздел 26). scale_pos_weight < 1 записывает это прямо в функцию потерь.
                par = dict(par or {}, scale_pos_weight=args.pw)
            model, predict, iters = FIT[name](Xt, yt, Xv, yv, par)
            scores[name] = (predict(Xv).astype(np.float32), predict(Xs).astype(np.float32))
            print(f'  {name}: {iters} деревьев за {time.time() - t1:.0f} с', flush=True)
            if name == 'xgb':
                model.save_model(out_dir / 'models' / f'xgb_{tp}.json')
            elif name == 'cat':
                model.save_model(str(out_dir / 'models' / f'cat_{tp}.cbm'))
            else:
                model.save_model(str(out_dir / 'models' / f'lgbm_{tp}.txt'))
            report.setdefault(tp, {}).setdefault('importance', {})[name] = importance(model, name, features)
            report[tp].setdefault('iterations', {})[name] = iters
        for name, (pv, ps) in scores.items():
            np.save(out_dir / 'preds' / f'{name}_{tp}_val.npy', pv)
            np.save(out_dir / 'preds' / f'{name}_{tp}_test.npy', ps)
            thr = metrics.best_threshold(yv, pv)
            row = {}
            # val_h1 — январь–июнь 2025: те же месяцы, что в тесте, без летнего сезона подтоплений
            for split, df, p, m in (('val', va, pv, None), ('val_h1', va, pv, h1), ('test', te, ps, None)):
                obj, hh = df['object_id'].to_numpy(), df['h'].to_numpy()
                for target in ('', '_prim', '_conf'):
                    nx = df[f'next_{tp}{target}'].to_numpy()
                    row[f'{split}{target}'] = (metrics.evaluate(obj, hh, nx, p, thr, H, cap) if m is None else
                                               metrics.evaluate(obj[m], hh[m], nx[m], p[m], thr, H, cap))
            if name not in ('rules', 'recency'):
                iso = IsotonicRegression(out_of_bounds='clip').fit(pv, yv)
                ys = (te[f'next_{tp}{tg}'].to_numpy() <= H).astype(np.float32)
                row['ece_test_raw'] = metrics.ece(ys, ps)
                row['ece_test_isotonic'] = metrics.ece(ys, iso.predict(ps))
            report.setdefault(tp, {}).setdefault('scores', {})[name] = row
            v, s, sp = row['val'], row['test'], row['test_prim']
            print(f'  {name:8s} val PR-AUC {v["pr_auc"]:.3f} | test PR-AUC {s["pr_auc"]:.3f} P {s["precision"]:.3f} '
                  f'R(эп) {s["recall_episodes"]:.3f} упрежд {s["lead_median_h"]:.0f} ч | первичные PR-AUC '
                  f'{sp["pr_auc"]:.3f} R(эп) {sp["recall_episodes"]:.3f}', flush=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nготово за {time.time() - t:.0f} с → {report_path}')


if __name__ == '__main__':
    main()
