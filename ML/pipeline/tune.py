"""Шаг 5. Подбор гиперпараметров Optuna по PR-AUC на проверке 2025 (тест 2026 не трогаем).

XGBoost и CatBoost — на GPU. Матрицы квантуются один раз, между типами меняется только метка.
Исследования пишутся в work/optuna.db: прерванный подбор продолжается с того же места.
Лучшие параметры — work/runs/tune/<модель>_<тип>.json, их подхватывает train.py --params tuned.

    python tune.py --model xgb --trials 60
    python tune.py --model xgb --rounds 100 --early 20   # подбор под короткий бюджет (раздел 24)
    python tune.py --model cat --trials 30 --target _prim
"""
import argparse
import json
import time

import numpy as np
import optuna

import config
import train

TUNE = config.WORK / 'runs' / 'tune'


def xgb_study(Xt, Xv, labels, trials: int, storage: str, suffix: str, budget=(4000, 200)) -> dict:
    import xgboost as xgb
    dt = xgb.QuantileDMatrix(Xt, max_bin=256)
    dv = xgb.QuantileDMatrix(Xv, ref=dt)
    best = {}
    for tp, (yt, yv) in labels.items():
        dt.set_label(yt)
        dv.set_label(yv)

        def objective(trial: optuna.Trial) -> float:
            p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'device': 'cuda',
                 'max_bin': 256,
                 'max_depth': trial.suggest_int('max_depth', 3, 10),
                 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                 'min_child_weight': trial.suggest_float('min_child_weight', 1, 500, log=True),
                 'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                 'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
                 'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 100, log=True),
                 'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                 'gamma': trial.suggest_float('gamma', 0, 10),
                 'max_delta_step': trial.suggest_float('max_delta_step', 0, 5)}
            b = xgb.train(p, dt, num_boost_round=budget[0], evals=[(dv, 'val')],
                          early_stopping_rounds=budget[1], verbose_eval=False)
            trial.set_user_attr('iterations', b.best_iteration + 1)
            return float(b.best_score)

        best[tp] = run_study(f'xgb_{tp}{suffix}', objective, trials, storage)
    return best


def cat_study(Xt, Xv, labels, trials: int, storage: str, suffix: str, budget=(4000, 200)) -> dict:
    from catboost import CatBoostClassifier, Pool
    best = {}
    for tp, (yt, yv) in labels.items():
        pt, pv = Pool(Xt, yt), Pool(Xv, yv)

        def objective(trial: optuna.Trial) -> float:
            from sklearn.metrics import average_precision_score
            p = {'iterations': budget[0], 'task_type': 'GPU', 'devices': '0', 'loss_function': 'Logloss',
                 'od_type': 'Iter', 'od_wait': budget[1], 'use_best_model': True, 'verbose': False,
                 'gpu_ram_part': 0.8, 'border_count': 254,
                 'depth': trial.suggest_int('depth', 4, 10),
                 'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
                 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 100, log=True),
                 'random_strength': trial.suggest_float('random_strength', 0.1, 20, log=True),
                 'bootstrap_type': 'Bernoulli',
                 'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 500, log=True),
                 'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise'])}
            m = CatBoostClassifier(**p)
            m.fit(pt, eval_set=pv)
            trial.set_user_attr('iterations', m.get_best_iteration() + 1)
            return float(average_precision_score(yv, m.predict_proba(pv)[:, 1]))

        best[tp] = run_study(f'cat_{tp}{suffix}', objective, trials, storage)
    return best


def run_study(name: str, objective, trials: int, storage: str) -> dict:
    t = time.time()
    study = optuna.create_study(direction='maximize', study_name=name, storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    left = trials - len([x for x in study.trials if x.state == optuna.trial.TrialState.COMPLETE])
    if left > 0:
        study.optimize(objective, n_trials=left, gc_after_trial=True)
    b = study.best_trial
    out = {'value': b.value, 'params': b.params, 'iterations': b.user_attrs.get('iterations'),
           'trials': len(study.trials)}
    TUNE.mkdir(parents=True, exist_ok=True)
    (TUNE / f'{name}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{name}: PR-AUC на проверке {b.value:.4f} после {len(study.trials)} попыток, '
          f'{time.time() - t:.0f} с, {b.params}', flush=True)
    return out


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xgb', choices=['xgb', 'cat'])
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--target', default='', choices=['', '_prim'])
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--rounds', type=int, default=4000, help='предел числа деревьев (раздел 24)')
    ap.add_argument('--early', type=int, default=200, help='запас ранней остановки (раздел 24)')
    args = ap.parse_args()
    meta = json.loads((train.FEAT / 'meta.json').read_text(encoding='utf-8'))
    features = meta['features']
    types = args.types.split(',')
    cols = features + [f'next_{t}{args.target}' for t in types]
    tr = train.load(train.SPLITS['main'][0], args.step, cols)
    va = train.load([2025], 1, cols)
    Xt, Xv = train.matrix(tr, features), train.matrix(va, features)
    labels = {t: tuple((df[f'next_{t}{args.target}'].to_numpy() <= args.horizon).astype(np.float32)
                       for df in (tr, va)) for t in types}
    del tr, va
    print(f'подбор {args.model}{args.target}: обучение {Xt.shape}, проверка {Xv.shape}', flush=True)
    storage = f"sqlite:///{(config.WORK / 'optuna.db').as_posix()}"
    # Раздел 24: подбор, сделанный при 4000 раундах, под короткий бюджет не годится — у него
    # своё имя исследования и свой файл, чтобы прежний не затирался.
    budget = '' if (args.rounds, args.early) == (4000, 200) else f'_r{args.rounds}e{args.early}'
    suffix = (args.target + budget
              + ('' if args.horizon == config.HORIZON else f'_h{args.horizon}'))
    {'xgb': xgb_study, 'cat': cat_study}[args.model](Xt, Xv, labels, args.trials, storage, suffix,
                                                     (args.rounds, args.early))


if __name__ == '__main__':
    main()
