"""Шаг 13. Свой горизонт каждому типу (Ф5-3).

Основной контур учит все шесть типов на одном горизонте 24 ч, но типы разные. Пожар разгорается за
часы, отказ насоса зреет неделями: горизонт, на котором модель что-то различает, у них не обязан
совпадать. Витрина хранит «через сколько часов следующий эпизод» до 30 суток, поэтому горизонт
меняется без пересборки — только метка.

Сравнивать PR-AUC между горизонтами нельзя: с ростом H растёт и доля положительных, площадь
поднимается сама собой. Поэтому здесь два честных числа:

- **lift** — во сколько раз PR-AUC выше доли положительных: разрешающая способность модели;
- **при бюджете N тревог в сутки** (порог подобран на проверке) — какую долю эпизодов ловим на тесте
  и с какой точностью. Ложная тревога дороже пропуска, поэтому бюджет задан, а не подобран по F1.

Параметры для всех горизонтов берутся одни — подобранные на 24 ч, иначе сравнение было бы о подборе.

    python horizon.py --models xgb --hours 6,12,24,48,72
    python horizon.py --models xgb,cat --types fire,flood --budget 10
"""
import argparse
import json
import time

import numpy as np

import config
import metrics
import train

FEAT = config.WORK / 'features'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='xgb')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--hours', default='6,12,24,48,72')
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--budget', type=float, default=10, help='тревог в сутки по всем объектам')
    args = ap.parse_args()
    hours = [int(x) for x in args.hours.split(',') if x]
    models = [m for m in args.models.split(',') if m]

    meta = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))
    features, cap = meta['features'], meta['next_cap']
    keys = ['object_id', 'h']
    cols = keys + features + meta['targets']
    t = time.time()
    tr = train.load(train.SPLITS['main'][0], args.step, cols)
    va = train.load([2025], 1, cols)
    te = train.load([2026], 1, cols)
    Xt, Xv, Xs = train.matrix(tr, features), train.matrix(va, features), train.matrix(te, features)
    tr, va, te = (df.select(keys + meta['targets']) for df in (tr, va, te))
    ov, hv = va['object_id'].to_numpy(), va['h'].to_numpy()
    os_, hs = te['object_id'].to_numpy(), te['h'].to_numpy()
    days_v = (hv.max() - hv.min() + 1) / 24
    days_s = (hs.max() - hs.min() + 1) / 24
    print(f'обучение {Xt.shape}, проверка {Xv.shape}, тест {Xs.shape}: {time.time() - t:.0f} с', flush=True)

    rows = []
    for tp in config.TYPES if args.types == 'all' else args.types.split(','):
        nv, ns = va[f'next_{tp}'].to_numpy(), te[f'next_{tp}'].to_numpy()
        for H in hours:
            yt = (tr[f'next_{tp}'].to_numpy() <= H).astype(np.float32)
            yv = (nv <= H).astype(np.float32)
            for name in models:
                t1 = time.time()
                params = train.tuned(name, tp, '', config.HORIZON)
                model, predict, iters = train.FIT[name](Xt, yt, Xv, yv, params)
                pv, ps = predict(Xv).astype(np.float32), predict(Xs).astype(np.float32)
                base = float(yv.mean())
                pr_auc = float(metrics.average_precision_score(yv, pv))
                thr = metrics.threshold_for_rate(pv, args.budget, len(pv), days_v)
                m = metrics.evaluate(os_, hs, ns, ps, thr, H, cap)
                per_day = m['alarm_rate'] * len(ps) / days_s
                rows.append((tp, H, name, base, pr_auc, pr_auc / max(base, 1e-9),
                             m['precision'], m['recall_episodes'], per_day, m['lead_median_h']))
                print(f'  {config.TYPE_NAMES[tp]:18s} H={H:3d} {name} {iters:5d} дер. за '
                      f'{time.time() - t1:4.0f} с | база {base:.4f} PR-AUC {pr_auc:.3f} '
                      f'lift {pr_auc / max(base, 1e-9):5.1f} | тест P {m["precision"]:.3f} '
                      f'R(эп) {m["recall_episodes"]:.3f} тревог/сут {per_day:.1f}', flush=True)

    print(f'\nПорог — под {args.budget:.0f} тревог в сутки на проверке 2025, числа справа — тест 2026.\n')
    print('| тип | горизонт | модель | доля положительных | PR-AUC | lift | Precision | Recall (эп.) '
          '| тревог в сутки | упреждение, ч |')
    print('|---|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for tp, H, name, base, pr, lift, p, r, pd_, lead in rows:
        print(f'| {config.TYPE_NAMES[tp]} | {H} ч | {name} | {base:.4f} | {pr:.3f} | {lift:.1f} | '
              f'{p:.3f} | {r:.3f} | {pd_:.1f} | {lead:.0f} |'.replace('.', ','))
    print(f'\nготово за {time.time() - t:.0f} с')


if __name__ == '__main__':
    main()
