"""Дребезг: сколько «ложных тревог» — это одна и та же тревога, поднятая заново (раздел 27).

Сигналы уже считаются по-человечески: подряд идущие часы тревоги на объекте — это один сигнал, а
не десять (`metrics.signals`). Но достаточно одного спокойного часа посередине, и тот же самый
случай в отчёте распадается надвое. Диспетчер при этом видит не два случая, а один: объект уже
в работе, к нему уже съездили.

EEMUA 191 и ISA 18.2 называют это дребезгом (chattering alarm) и требуют его подавлять — считать
повтор в пределах заданного окна продолжением прежней тревоги, а не новой. Здесь меряется, сколько
это стоит и сколько приносит: окно склейки `G` меняется от нуля (как сейчас) до трёх суток.

**Чем платим.** Склейка не бесплатна. Если тревога поднялась, бригада съездила и ничего не нашла,
а через пять часов на том же объекте действительно начинается эпизод, — при склейке отдельного
оповещения об этом эпизоде не будет: тревога формально «всё ещё висит». Поэтому в таблице две
колонки про эпизоды: сколько поймано вообще и сколько поймано **свежим** сигналом, то есть таким,
который поднялся не раньше чем за горизонт до начала эпизода. Вторая колонка и есть цена склейки.

Порог берётся по лучшему F1 на проверке 2025 и без изменений применяется к тесту — как везде.

    python chatter.py
    python chatter.py --run main_h24_tuned_r12e4 --gaps 0,6,24
"""
import argparse

import numpy as np

import config
import metrics
import operating

GAPS = (0, 3, 6, 12, 24, 72)


def runs_with_gap(obj: np.ndarray, h: np.ndarray, alarm: np.ndarray, gap: int):
    """Разбить часы тревоги на сигналы, склеивая паузы не длиннее gap часов.

    Возвращает (индексы часов в порядке сортировки, номер сигнала для каждого часа).
    """
    idx = np.flatnonzero(alarm)
    if not len(idx):
        return idx, np.empty(0, np.int64)
    order = idx[np.lexsort((h[idx], obj[idx]))]
    o, hh = obj[order], h[order]
    start = np.empty(len(order), bool)
    start[0] = True
    start[1:] = (o[1:] != o[:-1]) | (hh[1:] - hh[:-1] > gap + 1)
    return order, np.cumsum(start) - 1


def table(run: str, model: str, H: int, gaps, on: str = 'test') -> str:
    out = ['| тип | склейка | сигналов | ложных | ложных в сутки | поймано эпизодов | '
           'из них свежим сигналом |', '|---|---|---:|---:|---:|---|---|']
    for tp in config.TYPES:
        ov, hv, nv, pv = operating.split(run, 'val', 2025, tp, model)
        oo, hs, ns, ps = ((ov, hv, nv, pv) if on == 'val'
                          else operating.split(run, 'test', 2026, tp, model))
        thr = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        y = (ns <= H).astype(np.int8)
        alarm = ps >= thr
        days = (hs.max() - hs.min() + 1) / 24
        eps = metrics.onsets(oo, hs, ns, metrics.RUN_CAP)
        for g in gaps:
            order, run_id = runs_with_gap(oo, hs, alarm, g)
            if not len(order):
                out.append(f'| {config.TYPE_NAMES[tp]} | {g} ч | 0 | 0 | 0,0 | 0 | 0 |')
                continue
            total = int(run_id[-1]) + 1
            true = int(np.bincount(run_id, weights=y[order], minlength=total).astype(bool).sum())
            # начало каждого сигнала и принадлежность часа сигналу
            first = np.full(total, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(first, run_id, hs[order])
            where = {(int(o), int(x)): int(r) for o, x, r in zip(oo[order], hs[order], run_id)}
            caught = fresh = 0
            for o, e in eps:
                hit = [(o, e - k) for k in range(H, 0, -1) if (o, e - k) in where]
                if not hit:
                    continue
                caught += 1
                # свежий — если сигнал, накрывший эпизод, поднялся не раньше чем за горизонт до него
                if min(first[where[w]] for w in hit) >= e - H:
                    fresh += 1
            out.append(f'| {config.TYPE_NAMES[tp]} | {g} ч | {total} | {total - true} | '
                       f'{(total - true) / days:.1f} | {caught} из {len(eps)} | {fresh} |'
                       .replace('.', ','))
    return '\n'.join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned_r100e20')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--gaps', default=','.join(map(str, GAPS)),
                    help='окна склейки в часах: 0 — как сейчас')
    ap.add_argument('--on', default='test', choices=['test', 'val'],
                    help='val — повтор на 2025: склейка не подбирается по данным, но проверить, '
                         'что доля убранных ложных та же, всё равно надо (урок раздела 22)')
    args = ap.parse_args()
    where = 'тесте 2026' if args.on == 'test' else 'проверке 2025'
    print(f'Прогон {args.run}, модель {args.model}. Порог — по лучшему F1 на проверке 2025, '
          f'таблица на {where}.\n')
    print(table(args.run, args.model, args.horizon,
                [int(x) for x in args.gaps.split(',')], args.on))


if __name__ == '__main__':
    main()
