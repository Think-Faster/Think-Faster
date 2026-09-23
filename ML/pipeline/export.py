"""Выгрузка выбранной смеси моделей в work/export + manifest.json (INTEGRATION §3, §6).

Сервис tf-model кормится не из work/runs/, а из стабильной выгрузки: сюда собираются файлы
моделей пяти зёрен по каждому типу и один manifest, по которому сервис знает порядок признаков,
число деревьев, горизонт, параметры входа сети и важность (для поля `reasons` в сообщении §2.3).

    python export.py --run main_h24_tuned                  # CatBoost по всем 6 типам
    python export.py --run main_h24_tuned --fam xgb        # XGBoost (например, отказ датчика)
    python export.py --run main_h24_tuned --types fire,gas
    python export.py --run main_h24_tuned --net-run tcn_best   # вдобавок 5 сетей TCN в nets/

Зёрна: прогон без суффикса — это зерно 0; зерна 1..N лежат в прогонах с суффиксом _s{N}
(см. train.py: --seed). Пять зёрен = зерно 0 + зерна 1..4.

Итоговая структура:
    work/export/<тип>/<семейство>_s<зерно>/<семейство>_<тип>.<ext>
    work/export/nets/<сеть>.pt                      (если --net-run)
    work/export/manifest.json
Сервис грузит модели только из work/export и только по manifest.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import config

EXT = {'cat': 'cbm', 'xgb': 'json', 'lgbm': 'txt'}

# Вход сети (INTEGRATION §1.4): 168 ч, 66 рядов объекта + те же 66 усреднённые по коллектору,
# знаковый log1p в float16
SEQ = {'window': 168, 'ch': 132, 'transform': 'sign_log1p_f16'}


def tag_for(run: str, seed: int) -> str:
    return run if seed == 0 else f'{run}_s{seed}'


def model_path(run: str, fam: str, tp: str) -> Path:
    return config.WORK / 'runs' / run / 'models' / f'{fam}_{tp}.{EXT[fam]}'


def git_sha() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=config.ML, stderr=subprocess.DEVNULL).decode().strip()[:12]
    except Exception:
        return ''


def collect_report(run: str, tp: str, fam: str, seed: int) -> dict:
    """Важность и число деревьев семейства по типу из отчёта прогона.

    Отчёты лежат как report_<семейства>.json; для короткой базы (раздел 24) семейства обучались
    порознь — берём то, что есть, и не падаем, если нужного файла нет.
    """
    d = config.WORK / 'runs' / run
    for p in sorted(d.glob('report_*.json')):
        rep = json.loads(p.read_text(encoding='utf-8'))
        block = rep.get(tp, {})
        if fam in (block.get('importance') or {}) and fam in (block.get('iterations') or {}):
            return {'importance': block['importance'][fam], 'iterations': block['iterations'][fam],
                    'seed': seed}
    return {'importance': None, 'iterations': None, 'seed': seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='базовый тег прогона (зерно 0)')
    ap.add_argument('--fam', default='cat', choices=list(EXT), help='семейство для всех типов')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--seeds', default='1..4',
                    help='дополнительные зёрна: диапазон a..b либо список через запятую')
    ap.add_argument('--net-run', default='', help='прогон с пятью сетями TCN — собрать из него nets/')
    ap.add_argument('--overwrite', action='store_true', help='не ругаться на существующий экспорт')
    args = ap.parse_args()

    export = config.WORK / 'export'
    if export.exists() and not args.overwrite:
        sys.exit('work/export уже существует: --overwrite, чтобы заменить')
    t = time.time()
    meta = json.loads((config.WORK / 'features' / 'meta.json').read_text(encoding='utf-8'))
    seeds = [0] + [int(x) for x in (range(1, 5) if args.seeds == '1..4' else args.seeds.split(','))]
    types = [x for x in args.types.split(',') if x]
    missing = [t for t in types if not model_path(tag_for(args.run, 0), args.fam, t).exists()]
    if missing:
        sys.exit(f'у прогона {args.run} нет моделей {args.fam} для: {", ".join(missing)}')

    manifest = {
        'schema': 1,
        'exported': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'git_sha': git_sha(),
        'features': meta['features'],
        'targets': meta['targets'],
        'stypes': meta['stypes'],
        'horizon': config.HORIZON,
        'next_cap': meta['next_cap'],
        'known': meta['known'],
        'seq': dict(SEQ, present=bool(args.net_run)),
        'models': {},
    }
    for tp in types:
        fams = {}
        for s in seeds:
            tag = tag_for(args.run, s)
            src = model_path(tag, args.fam, tp)
            if not src.exists():
                print(f'  пропуск зерна {s} типа {tp}: нет {src.relative_to(config.WORK)}', flush=True)
                continue
            dst = export / tp / f'{args.fam}_s{s}'
            dst.mkdir(parents=True, exist_ok=True)
            (dst / src.name).write_bytes(src.read_bytes() if hasattr(src, 'read_bytes') else src.open('rb').read())
            rep = collect_report(tag, tp, args.fam, s)
            fams[str(s)] = {'path': f'{tp}/{args.fam}_s{s}', 'iterations': rep['iterations'],
                            'importance': rep['importance']}
        if not fams:
            sys.exit(f'не собрано ни одного зерна по типу {tp}')
        manifest['models'][tp] = {'family': args.fam, 'seeds': fams}
    if args.net_run:
        nets = sorted((config.WORK / 'runs' / args.net_run).glob('*.pt'))
        if not nets:
            nets = sorted((config.WORK / 'runs' / args.net_run / 'models').glob('*.pt'))
        if not nets:
            sys.exit(f'в прогоне {args.net_run} не нашлось *.pt')
        (export / 'nets').mkdir(parents=True, exist_ok=True)
        for n in nets:
            (export / 'nets' / n.name).write_bytes(n.read_bytes())
        manifest['seq']['present'] = True
        manifest['seq']['nets'] = [n.name for n in nets]
    (export / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                          encoding='utf-8')
    print(f'export готов: {export} за {time.time() - t:.1f} с')
    print(f'  типов {len(manifest["models"])}, зёрен {sorted(manifest["models"][types[0]]["seeds"])}'
          if types else '')
    print(f'  признаков {len(manifest["features"])}, стать: {git_sha() or "нет git"}')