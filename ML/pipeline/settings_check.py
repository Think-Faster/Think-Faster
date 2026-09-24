"""Что даёт текущий `settings/operating.json` против исходной конфигурации, по типам.

«Было» — доли раздела 42 без правила отклонения; «стало» — доли и k из файла настроек. Симуляция
отклонения — `reject.simulate`: диспетчер проверяет сигнал через час и отклоняет, если ничего не
случилось. Запускать после каждой правки настроек (раздел 46).

    python settings_check.py "$MIXT"
"""
import sys
import numpy as np
import config
from smooth import SHARE, load, measure
from reject import simulate


def run(d, ons, share, k):
    return measure(d, simulate(d, ons, share, 0 if k is None else 1, 10**6, k or 0.0))


def main():
    cfg = config.operating()['types']
    head = ('ложных сигналов', 'ложных часов', 'поймано', 'свежих', 'пропущено', 'верных сигналов из всех')
    for on in ('val', 'test'):
        print(f'\n### {on}\n')
        print('| тип | эпизодов | доля / k | ' + ' | '.join(head) + ' |')
        print('|---|---|---|' + '---|' * len(head))
        tot = np.zeros((2, 6))
        for tp in config.TYPES:
            d = load(tp, on)
            by = {}
            for a, b in d['eps']:
                by.setdefault(a, []).append(b)
            ons = {a: np.array(sorted(b), np.int64) for a, b in by.items()}
            ne = len(d['W'])
            rows = []
            for i, (s, k) in enumerate(((SHARE[tp], None), (cfg[tp]['share'], cfg[tp]['reject_k']))):
                m = run(d, ons, s, k)
                v = np.array([m['false_sig'], m['fh'], m['caught'], m['fresh'], ne - m['caught'], m['sig']])
                tot[i] += v
                rows.append(v)
            a, b = rows
            k = cfg[tp]['reject_k']
            cells = [f'{int(a[j])} → {int(b[j])}' for j in range(5)]
            cells.append(f'{(a[5] - a[0]) / a[5]:.0%} → {(b[5] - b[0]) / b[5]:.0%}')
            print(f"| {config.TYPE_NAMES[tp]} | {ne} | {SHARE[tp]:g} → {cfg[tp]['share']:g} / {'нет' if k is None else k} | "
                  + ' | '.join(cells) + ' |', flush=True)
        a, b = tot
        cells = [f'{int(a[j])} → {int(b[j])} ({(b[j] / a[j] - 1) * 100:+.0f}%)' for j in range(5)]
        cells.append(f'{(a[5] - a[0]) / a[5]:.0%} → {(b[5] - b[0]) / b[5]:.0%}')
        print('| **все** | | | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
