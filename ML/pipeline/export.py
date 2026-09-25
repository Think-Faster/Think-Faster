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

Отказ оборудования — исключение: там работает сеть (раздел 48), а не бустинг. Её нельзя обучить
тем же вызовом — обучение идёт на видеокарте часами, — поэтому веса берутся готовыми из прогона
вперёд `seqmodel.py --cutoff 2026-07-01 --tag prod_s<зерно>` (обучение до той же даты минус
горизонт, то есть на всей истории) и переносятся в выгрузку. Вход сети описан в `manifest.seq`.

Выход — `work/export/`: модели `<тип>/<семейство>_s<зерно>`, и `manifest.json` с признаками,
параметрами, деревьями и годами. Оценка каждого типа — среднее рангов пяти зёрен; порог — доля
часов под тревогой по скользящему окну 90 суток (раздел 32, `calib.py`).

    python export.py
    python export.py --types fire --seeds 0   # проба
    python export.py --check                  # проверка допущения (раздел 39)
    python export.py --equipment-version 2    # версия отказа оборудования для выбора главным диспетчером

`--check` проверяет само допущение «больше лет при том же числе деревьев — не хуже». Смесь
обучается так же, но на 2022–2025, то есть с годом проверки внутри, и сравнивается с рабочими
прогонами на тесте 2026 при долях часов раздела 38. Модели кладутся в `work/export_check/`.

`--equipment-version N` собирает одну из четырёх версий отказа оборудования (раздел 60, «Версии для
главного диспетчера») в `work/export_equipment_vN/`: только папка `equipment/` и `manifest.json` с
блоком `version`. Ничего не обучает, берёт готовые модели. Остальные типы у всех версий общие —
из `work/export`. Переключение идёт по типу (INTEGRATION.md §11, `ml.model_switch`).
"""
import argparse
import json
import time
from datetime import date

import numpy as np

import config
import metrics
import operating as op
import train

# Смесь раздела 34: тип -> (префикс прогона, семейство, параметры в train.tuned)
MIX = {'fire': ('main_h24_tunedh24', 'cat', 'tunedh24'),
       'gas': ('main_h24', 'cat', 'default'),
       'flood': ('main_h24_tuned', 'cat', 'tuned'),
       'equipment': ('prod', 'tcn', 'seq'),
       'sensor': ('main_h24_tuned', 'xgb', 'tuned'),
       'intrusion': ('main_h24', 'cat', 'default')}
YEARS = [2022, 2023, 2024, 2025, 2026]
# Сеть отказа оборудования (раздел 48): обучена прогоном вперёд на всём до даты минус горизонт
TCN_CUTOFF = '2026-07-01'

# Версии отказа оборудования (раздел 60, «Версии для главного диспетчера»). Номер — выбор главного
# диспетчера; по умолчанию 1. Версия — одна или две части; оценка версии — взвешенное среднее оценок
# частей, оценка части — среднее рангов её зёрен (как у остальных типов). Переобучать по расписанию
# нужно только версию 3; 1 и 2 заморожены, модуль дообучения по ТЗ §8 — по согласованию.
CAT = {'family': 'cat', 'run': 'main_h24_tunedh24', 'seeds': 5, 'trained_to': '2025-01-01'}
NET24 = {'family': 'tcn', 'tag': 'all', 'cutoff': '2024-01-01', 'seeds': 3}
EQ_VERSIONS = {
    1: {'name': 'тихая', 'parts': [(0.75, CAT), (0.25, NET24)],
        'about': 'смесь: бустинг CatBoost 2022–2024 с весом 0,75 и сеть TCN 2022–2023 с весом 0,25, '
                 'заморожена; при тех же ложных сигналах ловит больше всех'},
    2: {'name': 'перевзвод', 'parts': [(1.0, NET24)],
        'about': 'сеть TCN, 2022–2023, заморожена; короткие тревоги к каждому эпизоду, '
                 'больше всего свежих поимок; долю часов стоит опустить до 0,035, иначе шумнее версии 1'},
    3: {'name': 'разрыв в 2 года', 'parts': [(1.0, {'family': 'tcn', 'tag': 'prodgap730', 'cutoff': '2026-07-01',
                                                      'seeds': 3})],
        'about': 'сеть TCN на всей истории, кроме последних 730 суток; переобучается с тем же разрывом '
                 '(опыт G раздела 60: при равных ложных сигналах лучше разрыва в год)'},
    4: {'name': 'липкая', 'parts': [(1.0, {'family': 'tcn', 'tag': 'prod', 'cutoff': '2026-07-01', 'seeds': 5})],
        'about': 'сеть TCN на всей истории — прежний прод; дольше всех держит тревогу'},
}


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


def seq_spec(H: int) -> dict:
    """Чем кормится сеть: те же ряды, что в обучении (`seqmodel.py`), из `work/seq.npz`."""
    import features as ft
    return {'window': 168, 'channels': ft.BASE, 'collector': 'среднее тех же каналов по объектам коллектора',
            'transform': 'знаковый log1p, float16 (features.py)',
            'static': ['log1p состава датчиков', 'календарь: час/день недели/месяц/день года (sin, cos)/'
                       'праздник/длинные выходные/9 мая/дней до праздника', 'log1p часов с прошлого эпизода '
                       'по шести типам', f'log1p числа эпизодов за 720 и {ft.CAP} ч по шести типам'],
            'width': 128, 'dropout': 0.2, 'outputs': list(config.TYPES), 'horizon': H,
            'note': 'выход — шесть логитов, берётся канал своего типа; сигмоида не меняет порядок'}


def export_tcn(tp: str, seeds: list[int], out, manifest: dict) -> None:
    """Сети прогона вперёд, обученные на всей истории до `TCN_CUTOFF` минус горизонт (раздел 48).

    Бустинг для эксплуатации переобучается здесь же, а сеть — нет: её обучение идёт часами на
    видеокарте и запускается отдельно (`seqmodel.py --cutoff`), поэтому готовые веса переносятся.
    """
    import torch
    (out / tp).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        src = config.WORK / 'roll' / f'prod_s{seed}_{TCN_CUTOFF}.pt'
        if not src.exists():
            raise FileNotFoundError(f'нет сети {src.name}: обучить seqmodel.py --cutoff {TCN_CUTOFF} '
                                    f'--tag prod_s{seed}')
        name = f'tcn_s{seed}.pt'
        torch.save(torch.load(src, map_location='cpu', weights_only=True), out / tp / name)
        manifest.setdefault('models', {}).setdefault(tp, {})[str(seed)] = {
            'file': f'{tp}/{name}', 'family': 'tcn', 'params': 'seq', 'trained_to': TCN_CUTOFF,
            'from_run': f'prod_s{seed}_{TCN_CUTOFF}'}
        print(f'{tp} tcn зерно {seed}: сеть от {TCN_CUTOFF} перенесена', flush=True)
    prune(tp, out, manifest)


def export_equipment_version(k: int) -> None:
    """Версия отказа оборудования из готовых моделей в `work/export_equipment_v<k>/`."""
    import shutil
    v, tp, H = EQ_VERSIONS[k], 'equipment', config.HORIZON
    out = config.WORK / f'export_equipment_v{k}'
    (out / tp).mkdir(parents=True, exist_ok=True)
    meta = json.loads((train.FEAT / 'meta.json').read_text(encoding='utf-8'))
    manifest = {'features': meta['features'], 'horizon': H, 'built': date.today().isoformat(),
                'score': 'по частям blend: среднее рангов зёрен семейства; части — со своими весами', 'threshold': 'доля часов, окно 90 суток',
                'version': {'type': tp, 'number': k, **{x: v[x] for x in ('name', 'about')}},
                'models': {tp: {}}}
    manifest['blend'] = []
    for w, part in v['parts']:
        manifest['blend'].append({'family': part['family'], 'weight': w})
        for seed in range(part['seeds']):
            if part['family'] == 'cat':
                ref = part['run'] if seed == 0 else f"{part['run']}_s{seed}"
                src = config.WORK / 'runs' / ref / 'models' / f'cat_{tp}.cbm'
                entry = {'family': 'cat', 'params': 'tunedh24', 'trees': trees(ref, tp, 'cat'), 'from_run': ref,
                         'trained_to': part['trained_to']}
                name = f'cat_s{seed}.cbm'
            else:
                ref = f"{part['tag']}_s{seed}_{part['cutoff']}"
                src = config.WORK / 'roll' / f'{ref}.pt'
                entry = {'family': 'tcn', 'params': 'seq', 'from_run': ref, 'trained_to': part['cutoff']}
                name = f'tcn_s{seed}.pt'
                manifest['seq'] = seq_spec(H)
            if not src.exists():
                raise FileNotFoundError(f'нет модели {src}')
            shutil.copyfile(src, out / tp / name)
            manifest['models'][tp][name.split('.')[0]] = {'file': f'{tp}/{name}', **entry}
            print(f'версия {k}: {tp} {entry["family"]} зерно {seed} из {ref}', flush=True)
    prune(tp, out, manifest)
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')


def prune(tp: str, out, manifest: dict) -> None:
    """Убрать файлы прежней выгрузки, которых нет в манифесте: у типа могло смениться семейство."""
    keep = {v['file'].rsplit('/', 1)[-1] for v in manifest.get('models', {}).get(tp, {}).values()}
    for f in sorted((out / tp).iterdir()):
        if f.name not in keep:
            f.unlink()
            print(f'{tp}: файл прежней выгрузки {f.name} удалён', flush=True)


# Доли часов под тревогой по типам — рабочие, из settings/operating.json (раздел 46)
SHARES = config.shares()


def mix_name(tp: str) -> str:
    run, model, _ = MIX[tp]
    tail = '' if model == 'xgb' else f'/{model}'
    return '+'.join((run if s == 0 else f'{run}_s{s}') + tail for s in range(5))


def check_row(tp: str, obj, h, nxt, p, H: int) -> str:
    t = float(np.quantile(p, 1 - SHARES[tp]))
    m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
    sig, true = metrics.signals(obj, h, (nxt <= H).astype(np.int8), p >= t, 6)
    return f"{m['pr_auc']:.3f} · {m['caught']} из {m['episodes']} · {sig - true} ложных"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--step', type=int, default=3, help='шаг по часам, как в train.py')
    ap.add_argument('--check', action='store_true', help='обучить на 2022–2025 и сравнить на тесте 2026')
    ap.add_argument('--equipment-version', type=int, choices=sorted(EQ_VERSIONS), default=0,
                    help='собрать версию отказа оборудования из готовых моделей (раздел 60)')
    args = ap.parse_args()
    if args.equipment_version:
        export_equipment_version(args.equipment_version)
        return
    years = YEARS[:-1] if args.check else YEARS
    H = config.HORIZON
    meta = json.loads((train.FEAT / 'meta.json').read_text(encoding='utf-8'))
    features = meta['features']
    types = args.types.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    boost = [tp for tp in types if MIX[tp][1] != 'tcn']
    df = X = None
    if boost:
        t = time.time()
        df = train.load(years, args.step, ['object_id', 'h'] + features + meta['targets'])
        df = df.filter(df['h'] <= df['h'].max() - H)
        X = train.matrix(df, features)
        print(f'обучение {X.shape}, годы {years[0]}–{years[-1]}: {time.time() - t:.0f} с', flush=True)
        if args.check:
            test = train.load([2026], 1, ['object_id', 'h'] + features + meta['targets'])
            Xs = train.matrix(test, features)
            rows = []

    out = config.WORK / ('export_check' if args.check else 'export')
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    manifest.update({'features': features, 'years': years, 'horizon': H, 'step': args.step,
                     'rows': int(X.shape[0]) if X is not None else manifest.get('rows'),
                     'built': date.today().isoformat(),
                     'score': 'по частям blend: среднее рангов зёрен семейства; части — со своими весами', 'threshold': 'доля часов, окно 90 суток'})
    if any(MIX[tp][1] == 'tcn' for tp in types):
        manifest['seq'] = seq_spec(H)
    for tp in types:
        run, model, kind = MIX[tp]
        if model == 'tcn':
            if args.check:
                print(f'{tp}: сеть обучена прогоном вперёд (раздел 48), допущение о годах к ней не относится',
                      flush=True)
                continue
            export_tcn(tp, seeds, out, manifest)
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
            continue
        y = (df[f'next_{tp}'].to_numpy() <= H).astype(np.float32)
        objective = '_hours_v2024' if kind == 'tunedh24' else ''
        params = {} if kind == 'default' else train.tuned(model, tp, '', H, objective=objective) or {}
        (out / tp).mkdir(parents=True, exist_ok=True)
        ranks = 0
        for seed in seeds:
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
            if args.check:
                q = (m.inplace_predict(Xs) if model == 'xgb' else m.predict_proba(Xs)[:, 1])
                ranks = ranks + q.argsort().argsort()
        prune(tp, out, manifest)
        if args.check:
            obj, h, nxt = (test[c].to_numpy() for c in ('object_id', 'h', f'next_{tp}'))
            o, hh, nn, pw = op.load_mix(f'{tp}~{mix_name(tp)}', 'test', 2026, tp, model)
            assert np.array_equal(o, obj) and np.array_equal(hh, h)
            rows.append(f'| {config.TYPE_NAMES[tp]} | {SHARES[tp]:.1%} | {check_row(tp, obj, h, nxt, pw, H)} | '
                        f'{check_row(tp, obj, h, nxt, ranks.astype(np.float64), H)} |')
            print(rows[-1], flush=True)
    if args.check:
        print('\nТест 2026, доли часов из настроек, склейка 6 ч. В клетке: PR-AUC · поймано · ложных сигналов.\n')
        print('| тип | доля часов | рабочие прогоны (2022–2024) | те же деревья на 2022–2025 |')
        print('|---|---:|---|---|')
        print('\n'.join(rows))


if __name__ == '__main__':
    main()
