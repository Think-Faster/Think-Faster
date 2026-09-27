"""Срочный контур на 6 ч поверх суточного. Что он добавляет к суточной модели на рабочей доле.

Суточная — CatBoost ×5 на входах A′ (stfx_main_h24*, газ — main_h24*), тревога на доле config.shares().
Срочная — CatBoost ×5 с горизонтом 6 ч (main_h6<цель>*), своя доля s6 из сетки.
Добавка: эпизоды, не пойманные суточной за 24 ч, но пойманные срочной за 6 ч (упреждение ≥ 1 ч).
Цена в двух мерах: ложные сигналы срочной вне тревог суточной (серия часов, за которой 6 ч нет
эпизода), в долях ложных сигналов суточной (серия, за которой 24 ч нет эпизода), и часы тревоги
срочной вне тревог суточной. Опора — суточная с долей, поднятой до той же цены в каждой мере.

    TF_WORK=work_pa python urgent.py _prim [зёрен]      # analytics.md, раздел 65
    TF_WORK=work_pa python urgent.py '' [зёрен]         # обычная цель
    TF_WORK=work_pa python urgent.py _prim 5 24         # вторая суточная модель на первичной цели
    TF_WORK=work_pa python urgent.py _prim 5 24 v1      # то же поверх версии 1 оборудования (сеть v1, тест)
    TF_WORK=work_pa python urgent.py _prim 5 24 v1pa    # поверх смеси 0,75/0,25 с сетью pa_all (оба года)

Опора — суточный CatBoost×5; с v1 или v1pa — только оборудование, опора — 0,75 ранга CatBoost×5 +
0,25 ранга сети, как в версии 1. С work — опора того семейства и подбора, что в пакете
(work/export/manifest.json): у датчика XGBoost с подбором, у пожара и подтопления CatBoost с подбором;
прогоны обучаются на A′ заново (`train.py --models xgb --params tuned --types sensor` и т. п.):

    TF_WORK=work_pa python urgent.py _prim 5 6 work '' cat sensor
    TF_WORK=work_pa python urgent.py _prim 5 24 v1work    # v1pa с CatBoost-частью tunedh24, как в пакете

Пятый аргумент — сеть во второй модели, `<имя>:<зёрен>:<вес>`: ранги второй модели смешиваются с
рангами сети `seqmodel.py --name <имя>_s<зерно>` того же горизонта (runs/main_h<HZ>/preds), вес 1 —
одна сеть:

    TF_WORK=work_pa python urgent.py _prim 5 24 v1pa prim_ow0:3:0.25
    TF_WORK=work_pa python urgent.py _prim 5 6 '' h6_prim_ow0:3:1

Шестой аргумент — семейства второй модели через запятую (по умолчанию cat), ранги усредняются:

    TF_WORK=work_pa python urgent.py _prim 5 24 v1pa '' cat,xgb

Седьмой — типы через запятую (по умолчанию все), например 10 зёрен срочной на 3 ч у подтопления:

    TF_WORK=work_pa python urgent.py _prim 10 3 '' '' cat flood
"""
import sys
import numpy as np
import config
import operating

TG = sys.argv[1] if len(sys.argv) > 1 else '_prim'
TYPES = config.TYPES
S = config.shares()
S6 = [0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]
# зёрна второй модели; у суточной опоры их пять — берётся не больше
SEEDS = ['' if s == 0 else f'_s{s}' for s in range(int(sys.argv[2]) if len(sys.argv) > 2 else 5)]
HZ = int(sys.argv[3]) if len(sys.argv) > 3 else 6       # горизонт второй модели, ч
YEARS = {'val': [2025], 'test': [2026]}
BASE = sys.argv[4] if len(sys.argv) > 4 else ''         # '', v1, v1pa — опора у оборудования; work
if BASE in ('v1', 'v1pa', 'v1work'):
    TYPES = ['equipment']
# рабочая суточная модель в пакете, если она не CatBoost на умолчаниях: прогон и семейство
WORK_BASE = {'sensor': ('main_h24_tuned', 'xgb'), 'fire': ('main_h24_tunedh24', 'cat'),
             'flood': ('main_h24_tuned', 'cat'), 'equipment': ('main_h24_tunedh24', 'cat')}
NET = sys.argv[5].split(':') if len(sys.argv) > 5 and sys.argv[5] else None
MODELS = sys.argv[6].split(',') if len(sys.argv) > 6 and sys.argv[6] else ['cat']
if len(sys.argv) > 7:
    TYPES = sys.argv[7].split(',')


def with_net(p6, on, tp):
    """Ранги второй модели в смеси с рангами сети; порядок строк сети — index_<on> прогона main_h<HZ>."""
    tag, n, w = NET[0], int(NET[1]), float(NET[2])
    preds = config.WORK / 'runs' / f'main_h{HZ}' / 'preds'
    net = sum(pct(np.load(preds / f'{tag}_s{s}_{tp}_{on}.npy')) for s in range(n))
    assert len(net) == len(p6)
    return (1 - w) * pct(p6) + w * pct(net)


def pct(p):
    return p.argsort().argsort() / len(p)


def score(runs, on, yy, tp, models=('cat',)):
    acc, base = 0, None
    for r in runs:
        for m in models:
            o, h, n, p = operating.split(r, on, yy, tp, m)
            if base is None:
                base = (o, h, n)
            acc = acc + pct(p)
    return base, acc / (len(runs) * len(models))


def v1(p24, on):
    """Версия 1 оборудования: 0,75 ранга бустинга + 0,25 ранга сети, в порядке operating.split."""
    tag, seeds = ('old_all', range(3)) if BASE == 'v1' else ('pa_all', range(5))
    if BASE == 'v1' and on != 'test':
        return None
    suf = '_val' if on == 'val' else ''
    net = sum(pct(np.load(config.WORK / 'roll' / f'{tag}_s{s}_2024-01-01_equipment{suf}.npy')) for s in seeds)
    assert len(net) == len(p24)
    return 0.75 * pct(p24) + 0.25 * pct(net)


def runs_of(o, h, m):
    """Серии подряд идущих часов по объекту: номер серии для каждой строки с m, иначе -1."""
    idx = np.flatnonzero(m)
    lab = np.full(len(m), -1)
    if len(idx):
        st = np.r_[True, (o[idx][1:] != o[idx][:-1]) | (h[idx][1:] - h[idx][:-1] > 1)]
        lab[idx] = np.cumsum(st) - 1
    return lab


def false_runs(o, h, m, ok):
    lab = runs_of(o, h, m)
    k = lab.max() + 1
    if k <= 0:
        return 0
    good = np.zeros(k, bool)
    np.logical_or.at(good, lab[m], ok[m])
    return int((~good).sum())


def caught(key, eps, alarm, W):
    wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(W, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    ok = key[pos] == wk
    hit = alarm[pos] & ok
    lead = np.where(hit.any(1), W - hit.argmax(1), 0)
    return hit.any(1), lead


print(f'# R1: вторая модель {HZ} ч {"+".join(MODELS)} (цель {TG or "все эпизоды"}) поверх суточного {BASE}\n')
for tp in TYPES:
    brun, bfam = WORK_BASE[tp] if BASE in ('work', 'v1work') and tp in WORK_BASE else (None, 'cat')
    base24 = [(brun or ('main_h24' if tp == 'gas' else 'stfx_main_h24')) + s for s in SEEDS[:5]]
    run6 = [f'main_h{HZ}{TG}' + s for s in SEEDS]
    print(f'## {tp}, доля суточной {S[tp]:.3f}\n')
    print('| период | s6 | первичных | пойм. суточной | +срочной | медиана упрежд., ч | всех эпизодов | +срочной '
          '| ложных сигн. суточной | +срочной (доля) | суточная с той же прибавкой: +первичных '
          '| часов тревоги +срочной | суточная с той же прибавкой часов: +первичных |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for on, yy in YEARS.items():
        (o, h, n), p24 = score(base24, on, yy, tp, (bfam,))
        if BASE in ('v1', 'v1pa', 'v1work'):
            p24 = v1(p24, on)
            if p24 is None:
                continue
        np_ = operating.split(base24[0], on, yy, tp, bfam, '_prim')[2]
        _, p6 = score(run6, on, yy, tp, MODELS)
        if NET:
            p6 = with_net(p6, on, tp)
        order = np.lexsort((h, o))
        o, h, n, np_, p24, p6 = (a[order] for a in (o, h, n, np_, p24, p6))
        key = o.astype(np.int64) * 10**7 + h
        a24 = p24 >= np.quantile(p24, 1 - S[tp])
        f24 = false_runs(o, h, a24, n <= 24)
        ep_all = np.array(sorted(zip(o[n < 168].tolist(), (h + n)[n < 168].tolist())), np.int64).reshape(-1, 2)
        ep_all = np.unique(ep_all, axis=0)
        mp = np_ < 168
        ep_pr = np.unique(np.array(list(zip(o[mp].tolist(), (h + np_)[mp].tolist())), np.int64).reshape(-1, 2), axis=0)
        c24p, _ = caught(key, ep_pr, a24, 24)
        c24a, _ = caught(key, ep_all, a24, 24)
        # опора при той же цене: поднять долю суточной так, чтобы ложных сигналов прибавилось столько же
        grid = np.unique(np.r_[S[tp], S[tp] * np.geomspace(1.02, 12, 40)])
        cost, costh, gain = [], [], []
        for sh in grid:
            a = p24 >= np.quantile(p24, 1 - sh)
            cost.append(false_runs(o, h, a, n <= 24) - f24)
            costh.append(int((a & ~a24).sum()))
            # горизонт второй модели длиннее суток — и опоре засчитывается поимка в том же окне
            gain.append(int((caught(key, ep_pr, a, max(HZ, 24))[0] & ~c24p).sum()))
        cost, costh, gain = np.maximum.accumulate(cost), np.maximum.accumulate(costh), np.maximum.accumulate(gain)
        for s6 in S6:
            a6 = p6 >= np.quantile(p6, 1 - s6)
            c6p, l6p = caught(key, ep_pr, a6, HZ)
            c6a, _ = caught(key, ep_all, a6, HZ)
            gp = c6p & ~c24p
            ga = c6a & ~c24a
            f6 = false_runs(o, h, a6 & ~a24, n <= HZ)
            x6 = int((a6 & ~a24).sum())
            med = f'{np.median(l6p[gp]):.0f}' if gp.any() else '—'
            print(f'| {on} | {s6:g} | {len(ep_pr)} | {c24p.sum()} | {gp.sum()} | {med} | {len(ep_all)} | {ga.sum()} '
                  f'| {f24} | {f6} ({f6 / max(f24, 1):.0%}) | {np.interp(f6, cost, gain):.0f} '
                  f'| {x6} | {np.interp(x6, costh, gain):.0f} |', flush=True)
    print()
