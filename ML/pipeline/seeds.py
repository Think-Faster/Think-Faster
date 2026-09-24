"""Сколько зёрен нужно в смеси: цена перехода с пяти моделей на три и на одну.

В эксплуатацию по каждому типу уходит пять моделей (`export.py`), и каждая считается на каждом часе.
Пять зёрен взялись из сравнения моделей (раздел 34), где важно было убрать разброс зерна, а не
сэкономить. Здесь считается прямая цена: сколько поимок теряется, если оставить первые k зёрен
смеси при той же доле часов под тревогой.

Мера — часовая, как в разделах 34 и 46: при доле часов из настроек считаются поймано, свежие
поимки и ложные сигналы со склейкой 6 ч. Правило отклонения — своё у каждого типа, из настроек.

    python seeds.py "$MIXT"
"""
import sys

import numpy as np

import config
import smooth
from daily import alarms

MIX = sys.argv[1]
OP = config.operating()['types']
KS = [1, 2, 3, 5]


def parts(tp: str) -> list:
    table = dict(p.split('~', 1) for p in MIX.split(';') if p)
    return table.get(tp, table.get('*', '')).split('+')


def main() -> None:
    print('Доля часов — из настроек, склейка 6 ч. В клетке: поймано / свежих / ложных сигналов.\n')
    print('| тип | период | ' + ' | '.join(f'{k} зерно' if k == 1 else f'{k} зёрна' if k < 5 else f'{k} зёрен'
                                           for k in KS) + ' |')
    print('|---|---|' + '---|' * len(KS))
    for tp in config.TYPES:
        ps = parts(tp)
        for on in ('val', 'test'):
            cells = []
            for k in KS:
                smooth.MIX = '+'.join(ps[:k])          # load читает спецификацию из модульной переменной
                d = smooth.load(tp, on)
                m = smooth.measure(d, alarms(d, OP[tp]['share'], OP[tp]['reject_k']))
                cells.append(f'{m["caught"]} / {m["fresh"]} / {m["false_sig"]}')
            print(f'| {config.TYPE_NAMES[tp]} | {on} | ' + ' | '.join(cells) + ' |', flush=True)


if __name__ == '__main__':
    main()
