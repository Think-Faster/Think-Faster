"""Модели для эксплуатации: смесь раздела 34, переобученная на всех годах (раздел 38).

Рабочие прогоны учатся на 2022–2024, а 2025 и первое полугодие 2026 держат под проверку и тест.
В эксплуатацию идёт та же смесь, но обученная на всём: раздел 37 показал, что копить всё прошлое
лучше, чем отрезать, а раздел 4 — что годы до 2022 не нужны (другая сеть).

Отложенного года для ранней остановки при этом нет. Поэтому число деревьев каждой модели берётся
из её рабочего прогона — того же типа, семейства, параметров и зерна, где оно выбрано ранней
остановкой по 2025. Данных стало в полтора раза больше, и деревьев можно было бы дать больше, но
это не проверить без отложенного года; берём проверенное число.

Последние `H` часов 2026 не берутся: для них ещё не известно, случится ли эпизод в горизонте, и
они вошли бы в обучение ложными отрицательными.

Выход — `work/export/`: модели `<тип>/<семейство>_s<зерно>`, и `manifest.json` с признаками,
параметрами, деревьями и годами. Оценка каждого типа — среднее рангов пяти зёрен; порог — доля
часов под тревогой по скользящему окну 90 суток (раздел 32, `calib.py`).

    python export.py
    python export.py --types fire --seeds 0   # проба
"""
import argparse
import json
import time
from datetime import date

import numpy as np

import config
import train

# Смесь раздела 34: тип -> (префикс прогона, семейство, параметры в train.tuned)
MIX = {'fire': ('main_h24_tunedh24', 'cat', 'tunedh24'),
       'gas': ('main_h24', 'cat', 'default'),
       'flood': ('main_h24_tuned', 'cat', 'tuned'),
       'equipment': ('main_h24_tunedh24', 'cat', 'tunedh24'),
       'sensor': ('main_h24_tuned', 'xgb', 'tuned'),
       'intrusion': ('main_h24', 'cat', 'default')}
YEARS = [2022, 2023, 2024, 2025, 2026]


def trees(run: str, tp: str, model: str) -> int:
    """Число деревьев, выбранное ранней остановкой в рабочем прогоне."""
    for f in sorted((config.WORK / 'runs' / run).glob('report_*.json')):
        it = json.loads(f.read_text(encoding='utf-8')).get(tp, {}).get('iterations', {}).get(model)
        if it:
            return int(it)
    raise FileNotFoundError(f'нет числа деревьев для {run} {tp} {model}')


def fit(model: str, X, y, params: dict, n: int, seed: int):
    if model == 'xgb':
        import xgboost as xgb
        p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist', 'device': 'cuda',
             'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.6,
             'min_child_weight': 5, 'max_bin': 256, 'reg_lambda': 1.0}
        p.update(params)
        if seed:
            p['seed'] = seed
        return xgb.train(p, xgb.QuantileDMatrix(X, y, max_bin=p['max_bin']), num_boost_round=n)
    from catboost import CatBoostClassifier
    p = {'learning_rate': 0.05, 'depth': 8, 'task_type': 'GPU', 'devices': '0',
         'loss_function': 'Logloss', 'border_count': 254, 'verbose': False, 'gpu_ram_part': 0.8}
    p.update(params)
    p.update(iterations=n, use_best_model=False)
    for k in ('od_type', 'od_wait'):
        p.pop(k, None)
    if seed:
        p['random_seed'] = seed
    m = CatBoostClassifier(**p)
    m.fit(X, y)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--step', type=int, default=3, help='шаг по часам, как в train.py')
    args = ap.parse_args()
    H = config.HORIZON
    meta = json.loads((train.FEAT / 'meta.json').read_text(encoding='utf-8'))
    features = meta['features']
    t = time.time()
    df = train.load(YEARS, args.step, ['object_id', 'h'] + features + meta['targets'])
    df = df.filter(df['h'] <= df['h'].max() - H)
    X = train.matrix(df, features)
    print(f'обучение {X.shape}, годы {YEARS[0]}–{YEARS[-1]}: {time.time() - t:.0f} с', flush=True)

    out = config.WORK / 'export'
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    manifest.update({'features': features, 'years': YEARS, 'horizon': H, 'step': args.step,
                     'rows': int(X.shape[0]), 'built': date.today().isoformat(),
                     'score': 'среднее рангов зёрен по типу', 'threshold': 'доля часов, окно 90 суток'})
    for tp in args.types.split(','):
        run, model, kind = MIX[tp]
        y = (df[f'next_{tp}'].to_numpy() <= H).astype(np.float32)
        objective = '_hours_v2024' if kind == 'tunedh24' else ''
        params = {} if kind == 'default' else train.tuned(model, tp, '', H, objective=objective) or {}
        (out / tp).mkdir(parents=True, exist_ok=True)
        for seed in (int(s) for s in args.seeds.split(',')):
            ref = run if seed == 0 else f'{run}_s{seed}'
            n = trees(ref, tp, model)
            t1 = time.time()
            m = fit(model, X, y, dict(params), n, seed)
            name = f'{model}_s{seed}.' + ('json' if model == 'xgb' else 'cbm')
            m.save_model(str(out / tp / name))
            manifest.setdefault('models', {}).setdefault(tp, {})[str(seed)] = {
                'file': f'{tp}/{name}', 'family': model, 'params': kind, 'trees': n, 'from_run': ref}
            print(f'{tp} {model} зерно {seed}: {n} деревьев, доля положительных {y.mean():.4f}, '
                  f'{time.time() - t1:.0f} с', flush=True)
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
