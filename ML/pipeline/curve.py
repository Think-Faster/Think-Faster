"""Чем оплачивается полнота: доля часов под тревогой и часы ложной тревоги (раздел 31).

Зачем понадобилось. Во всех таблицах сравнения ложные считаются **сигналами** — подряд идущие
часы тревоги на одном объекте это один сигнал, а не десять. Мера правильная: диспетчер ходит на
сигнал, а не на час. Но у неё есть слепое пятно. Если опускать порог, блоки тревоги начинают
склеиваться между собой, и число сигналов, дойдя до максимума, **падает**, хотя тревога висит всё
дольше. У отказа оборудования на проверке 2025: 569 ложных при 10% часов под тревогой, 696 при
20%, 244 при 50% — и в последней точке средний сигнал длится 651 час, почти месяц.

Отсюда правило: сравнивать прогоны по числу ложных сигналов можно только выше примерно 10% часов
под тревогой. Ниже метрика немонотонна по порогу, и выбор по ней едет в её слепое пятно, а не к
лучшему прогнозу. Этот скрипт печатает обе величины рядом — число сигналов и часы, — чтобы
границу было видно, а не приходилось о ней помнить.

    python curve.py --on val --type equipment
    python curve.py --on test --type fire

Столбец «средняя длина сигнала» — главный: пока он держится в десятках часов, счёт сигналов
осмыслен; как только уходит в сотни, строка говорит уже не о качестве прогноза.
"""
import argparse

import numpy as np

import config
import metrics
import operating

# Точки перебора: сгущаются к малым долям часов, потому что рабочая область именно там.
SHARES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def curve(run: str, on: str, tp: str, model: str, H: int) -> None:
    year = operating.years(run, on)
    obj, h, nxt, p = operating.load_mix(run, on, year, tp, model, '')
    y = (nxt <= H).astype(np.int8)

    print(f'{config.TYPE_NAMES[tp]}, {on}, доля положительных часов {y.mean():.2%}')
    print('| доля часов под тревогой | полнота эпизодов | ложных сигналов | '
          'средняя длина сигнала, ч | часов ложной тревоги |')
    print('|---:|---:|---:|---:|---:|')
    for q in SHARES:
        t = float(np.quantile(p, q))
        a = p >= t
        if not a.any():
            continue
        sig, true = metrics.signals(obj, h, y, a)
        m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
        hours, false_sig = int(a.sum()), sig - true
        dur = hours / max(sig, 1)
        print(f'| {a.mean():.1%} | {m["caught"] / m["episodes"]:.1%} '
              f'({m["caught"]} из {m["episodes"]}) | {false_sig} | {dur:.0f} | '
              f'{int(false_sig * dur)} |')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='main_h24_tuned_r100e20')
    ap.add_argument('--on', default='val', choices=['val', 'test'])
    ap.add_argument('--type', dest='tp', default='', help='тип происшествия; пусто — все шесть')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=24)
    args = ap.parse_args()

    types = [args.tp] if args.tp else list(config.TYPES)
    for tp in types:
        curve(args.run, args.on, tp, args.model, args.horizon)
        print()


if __name__ == '__main__':
    main()
