"""PR-AUC на проверке и тесте по зёрнам и по смеси (среднее рангов) для каждого типа смеси.

argv[1] — спецификация смеси (тип~a+b+...;...), argv[2] — подпись строки, argv[3] (необязательно) —
доли часов под тревогой через запятую: для смеси печатаются поймано эпизодов и ложных часов при них.
Метки берутся из features текущей рабочей папки (TF_WORK), так что одни и те же прогнозы можно
оценить на старой и на новой разметке.
"""
import json
import sys
import numpy as np
from sklearn.metrics import average_precision_score
import config
import metrics
import operating

MIX, TAG = sys.argv[1], sys.argv[2]
SH = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else []
H = 24
CAP = json.loads((config.WORK / 'features' / 'meta.json').read_text(encoding='utf-8'))['next_cap']
table = dict(part.split('~', 1) for part in MIX.split(';') if part)
extra = ''.join(f' поймано / ложных часов при {s:g} |' for s in SH)
print(f'| разметка/модели | тип | период | эпизодных строк | PR-AUC зёрен | PR-AUC смеси |{extra}')
print('|---|---|---|---:|---|---:|' + '---:|' * len(SH))
for tp in [t for t in config.TYPES if t in table]:
    for on in ('val', 'test'):
        yr = operating.years(MIX, on)
        seeds, ranks, y = [], None, None
        for part in table[tp].split('+'):
            o, h, n, p = operating.load_mix(part, on, yr, tp, 'xgb', '')
            yy = n <= H
            if y is None:
                y, o0, h0, n0 = yy, o, h, n
            assert np.array_equal(o, o0) and np.array_equal(h, h0) and np.array_equal(y, yy)
            seeds.append(average_precision_score(y, p))
            r = p.argsort().argsort() / len(p)
            ranks = r if ranks is None else ranks + r
        ranks = ranks / len(seeds)
        cells = ''
        for s in SH:
            m = metrics.evaluate(o0, h0, n0, ranks, float(np.quantile(ranks, 1 - s)), H, CAP)
            false_h = int(round(m['alarm_rate'] * len(ranks) * (1 - m['precision'])))
            cells += f' {m["caught"]} из {m["episodes"]} / {false_h} |'
        print(f'| {TAG} | {tp} | {on} | {int(y.sum())} | {" ".join(f"{s:.3f}" for s in seeds)} | '
              f'{average_precision_score(y, ranks):.3f} |{cells}', flush=True)
