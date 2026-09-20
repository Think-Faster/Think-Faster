"""Выбор наилучшего варианта по каждому типу: решение на проверке, отчёт на тесте (раздел 30).

К этому моменту есть две оси, и обе выигрывают поодиночке: короткий бюджет деревьев (раздел 24)
и постобработка оценки — склейка дребезга и сглаживание (разделы 27 и 29). Складывать их выводы
на глаз нельзя: у каждого типа свой оптимум, а выбирать оптимум по тесту — это выбирать по
ответам. Поэтому здесь один протокол на всё:

1. кандидат — пара (прогон, окно сглаживания);
2. у каждого кандидата свой порог, поэтому сравниваются они при равной полноте;
3. выбор делается **на проверке 2025**: у кого меньше ложных сигналов при нужной доле пойманных;
4. печатается число выбранного кандидата **на тесте 2026** — и рядом лучшее возможное на тесте,
   чтобы было видно, сколько стоит то, что выбор сделан вслепую.

Последний столбец — цена выбора. Если он мал, протокол переносится в работу; если велик, значит
выбор на проверке не переносится и раздел честнее закрыть отказом.

    python choose.py
    python choose.py --level 0.5 --gap 6
"""
import argparse

import numpy as np

import config
import persist

SMOOTHS = (1, 3, 6, 12)


def name(run: str, m: int) -> str:
    short = run.replace('main_h24_tuned_', '')
    return short + (f' + среднее {m} ч' if m > 1 else ' + как есть')


def best_at(pts, need: float) -> int | None:
    ok = [f for c, f, _ in pts if c >= need]
    return min(ok) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='main_h24_tuned_r100e20,main_h24_tuned_r12e4')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--level', type=float, default=0.6, help='доля пойманных эпизодов')
    ap.add_argument('--gap', type=int, default=6, help='склейка дребезга, раздел 27')
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    args = ap.parse_args()
    runs = args.runs.split(',')

    print(f'Выбор по проверке 2025, числа по тесту 2026. Уровень: поймано '
          f'{args.level:.0%} эпизодов, склейка дребезга {args.gap} ч.\n')
    print('| тип | выбрано по проверке | ложных на тесте | лучшее возможное на тесте | цена выбора |')
    print('|---|---|---:|---:|---:|')
    picked_sum = oracle_sum = base_sum = 0
    for tp in config.TYPES:
        val_f, test_f = {}, {}
        base = None
        for run in runs:
            try:
                va = persist.prepare(run, 'val', 2025, tp, args.model, args.horizon)
                te = persist.prepare(run, 'test', 2026, tp, args.model, args.horizon)
            except FileNotFoundError:
                continue
            for m in SMOOTHS:
                key = name(run, m)
                for ctx, store in ((va, val_f), (te, test_f)):
                    score = persist.smoothed(ctx, m)
                    flat = score[~np.isnan(score)]
                    thr = np.quantile(flat, np.linspace(0.90, 0.99999, args.steps))
                    pts = persist.curve(score, 1, thr, ctx, args.gap)
                    got = best_at(pts, args.level * ctx['episodes'])
                    if got is not None:
                        store[key] = got
                if run == runs[0] and m == 1:
                    # прежнее состояние: та же база, но без склейки и без сглаживания
                    raw = persist.curve(te['grid'], 1,
                                        np.quantile(te['grid'][~np.isnan(te['grid'])],
                                                    np.linspace(0.90, 0.99999, args.steps)),
                                        te, 0)
                    base = best_at(raw, args.level * te['episodes'])
        common = [k for k in val_f if k in test_f]
        if not common:
            print(f'| {config.TYPE_NAMES[tp]} | — | — | — | — |')
            continue
        pick = min(common, key=lambda k: val_f[k])
        oracle = min(test_f.values())
        picked_sum += test_f[pick]
        oracle_sum += oracle
        base_sum += base if base is not None else test_f[pick]
        print(f'| {config.TYPE_NAMES[tp]} | {pick} | {test_f[pick]} | {oracle} | '
              f'+{test_f[pick] - oracle} |')
    print(f'\nИтого ложных сигналов на тесте: выбранное {picked_sum}, '
          f'лучшее возможное {oracle_sum}, цена слепого выбора '
          f'{picked_sum - oracle_sum} ({(picked_sum - oracle_sum) / max(oracle_sum, 1):.0%}).')
    print(f'Для сравнения, тот же уровень полноты на прежней базе без постобработки: {base_sum}.')


if __name__ == '__main__':
    main()
