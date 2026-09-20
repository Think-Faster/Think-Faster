"""Шаг 16. Дообучение на новых данных: как часто, на каком окне и каким способом (ТЗ §8).

ТЗ требует модуль дообучения, но не говорит, как именно дообучать. Вариантов несколько, и они
дают разный результат: копить все данные, держать скользящее окно, взвешивать старое меньше или
достраивать деревья поверх прошлой модели. Здесь они сравниваются честно — прогоном вперёд по
времени, как это будет работать в эксплуатации.

**Прогон вперёд.** Время режется на отрезки по `--fold` суток. Перед каждым отрезком модель
переобучается на данных, доступных к этому моменту, и прогнозирует весь отрезок. Между обучением и
прогнозом остаётся разрыв в горизонт H: иначе метка обучающей строки заглядывает в отрезок, который
мы ещё не видели. Порог выбирается на последних двух неделях перед переобучением и идёт на отрезок
без изменений — ровно так, как это будет в проде.

**Стратегии.**

| ключ | что делает |
|---|---|
| `all` | всё накопленное прошлое (окно растёт) |
| `d90`, `d180`, `d365` | скользящее окно в 90/180/365 суток, старое отрезается |
| `decay90`, `decay180` | всё прошлое, но вес строки падает вдвое каждые 90/180 суток |
| `warm` | деревья достраиваются поверх модели прошлого отрезка, только на новых данных |
| `frozen` | модель обучается один раз и больше не трогается, пересчитывается только порог |
| `thrfix` | наоборот: модель переобучается как в `all`, а порог берётся один раз и держится |

`frozen` и `thrfix` — контроли, а не кандидаты в работу. Раздел 19 нашёл, что переобучение раз в
30 суток выгоднее, чем раз в 90, и объяснил это не устареванием данных (признаки не дрейфуют,
раздел 13), а тем, что вместе с моделью пересчитывается **порог**, а он привязан к частоте
происшествий, которая ходит в разы. В литературе это называется prior probability shift, и
рекомендация там та же: если признаки стабильны, а меняется только доля положительных, лечить
надо порог, а не модель.

Само по себе это объяснение проверить нельзя: в `all` обе ручки крутятся вместе. Поэтому их
разводят. `all` — двигаются обе, `frozen` — только порог, `thrfix` — только модель. Если выигрыш
даёт порог, `frozen` догонит `all`, а `thrfix` просядет; если модель — наоборот. Тогда ответ на
вопрос «переобучать раз в месяц или хватит пересчитать порог» будет измерен, а не выведен.

Считается не PR-AUC, а то, что видит диспетчер: сколько сигналов за прогон, сколько из них ложных,
сколько эпизодов поймано. Итог — таблица «какая стратегия для какого типа лучше».

    python retrain.py --types fire,equipment --fold 90
    python retrain.py --strategies all,d365,warm --model xgb
    python retrain.py --fold 30 --rounds 1500 --early 100   # быстрее, если отрезков много
    python retrain.py --fold 14 --out retrain_f14           # параллельным прогонам — разные --out
    python retrain.py --fold 30 --pw 0.1 --gap 6           # лучшее из разделов 26 и 27 сразу
"""
import argparse
import json
import time

import numpy as np
import polars as pl

import config
import metrics
import train

FEAT = config.WORK / 'features'
DAY = 24
VAL_DAYS = 14          # последние сутки перед переобучением — на раннюю остановку и порог
STRATEGIES = ['all', 'd90', 'd180', 'd365', 'decay90', 'decay180', 'warm', 'frozen', 'thrfix']


def pick(h: np.ndarray, lo: int | None, hi: int) -> np.ndarray:
    m = h < hi
    return m if lo is None else m & (h >= lo)


def weights(h: np.ndarray, now: int, half_life_days: int) -> np.ndarray:
    """Вес строки падает вдвое каждые half_life суток — старое забывается плавно."""
    age = (now - h) / (half_life_days * DAY)
    return np.exp2(-age).astype(np.float32)


def fold_fit(name, Xt, yt, wt, Xv, yv, params, prev, rounds_max, early):
    """Обучение одного отрезка. prev — модель прошлого отрезка для стратегии warm."""
    if name != 'xgb':
        return train.FIT[name](Xt, yt, Xv, yv, params)
    import xgboost as xgb
    p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist',
         'device': 'cuda', 'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8,
         'colsample_bytree': 0.6, 'min_child_weight': 5, 'max_bin': 256, 'reg_lambda': 1.0}
    p.update(params or {})
    dt = xgb.QuantileDMatrix(Xt, yt, weight=wt, max_bin=p['max_bin'])
    dv = xgb.QuantileDMatrix(Xv, yv, ref=dt)
    rounds = 300 if prev is not None else rounds_max
    booster = xgb.train(p, dt, num_boost_round=rounds, evals=[(dv, 'val')],
                        early_stopping_rounds=early, verbose_eval=False, xgb_model=prev)
    best = booster.best_iteration + 1
    return booster, lambda X: booster.inplace_predict(X, iteration_range=(0, best)), best


def dump(rows: list[dict], args) -> None:
    """Сохранить то, что уже посчитано.

    Прогон вперёд идёт часами, а писать результат только в конце нельзя: сбой или нехватка
    времени посередине оставляют пустой файл и ночь впустую. Таблица переписывается после каждой
    досчитанной стратегии, так что в файле всегда лежит всё, что успело сойтись.

    Имя файла задаётся `--out`, и это не удобство. Прогонов вперёд много (ось частоты, ось веса),
    идут они часами, и запускать их приходится параллельно. Пока имя было одно на всех, два
    одновременных прогона переписывали таблицу друг другу после каждой стратегии, и к утру в файле
    лежала мешанина из двух разных расчётов, различить которые можно было только по шапке.
    """
    out = pl.DataFrame(rows)
    out.write_csv(config.WORK / f'{args.out}.csv')
    head = (f'Прогон вперёд с {args.start}, переобучение раз в {args.fold} суток, '
            f'модель {args.model}, порог под {args.budget} тревог в сутки, '
            f'потолок {args.rounds} деревьев (остановка {args.early})'
            + (f', вес положительного класса {args.pw}' if args.pw != 1.0 else '')
            + (f', склейка дребезга {args.gap} ч.' if args.gap else '.'))
    with open(config.WORK / f'{args.out}.md', 'w', encoding='utf-8') as f:
        f.write(head + '\n\n| ' + ' | '.join(out.columns) + ' |\n')
        f.write('|' + '---|' * len(out.columns) + '\n')
        for r in out.iter_rows():
            f.write('| ' + ' | '.join(str(x).replace('.', ',') for x in r) + ' |\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--out', default='retrain',
                    help='имя файлов результата в work: <out>.md и <out>.csv. Параллельным '
                         'прогонам нужны разные имена, иначе они перепишут таблицу друг другу')
    ap.add_argument('--strategies', default=','.join(STRATEGIES))
    ap.add_argument('--fold', type=int, default=90, help='как часто переобучаем, суток')
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--start', default='2025-01-01', help='с какого дня идёт прогон вперёд')
    ap.add_argument('--step', type=int, default=3, help='прореживание обучающих часов')
    ap.add_argument('--budget', type=float, default=10.0, help='тревог в сутки для порога')
    ap.add_argument('--rounds', type=int, default=4000, help='потолок деревьев на отрезок')
    ap.add_argument('--early', type=int, default=200,
                    help='ранняя остановка: раундов без улучшения')
    ap.add_argument('--gap', type=int, default=0,
                    help='склейка дребезга: пауза не длиннее gap часов не начинает новый сигнал '
                         '(раздел 27). По умолчанию 0 — счёт как раньше, чтобы старые таблицы '
                         'оставались сопоставимыми')
    ap.add_argument('--pw', type=float, default=1.0,
                    help='вес положительного класса: <1 делает пропуск дешевле ложной тревоги '
                         '(раздел 26). Тот же ключ, что у train.py, — иначе прогон вперёд '
                         'проверял бы не ту модель, которую ставим в работу')
    args = ap.parse_args()
    H, types = args.horizon, args.types.split(',')
    meta = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))
    feats = meta['features']
    tcols = [f'next_{tp}' for tp in types]

    t0 = time.time()
    years = [2022, 2023, 2024, 2025, 2026]
    pool = train.load(years, args.step, ['object_id', 'h'] + tcols + feats)
    ev = train.load([2025, 2026], 1, ['object_id', 'h'] + tcols + feats)
    Xp, Xe = train.matrix(pool, feats), train.matrix(ev, feats)
    pool, ev = pool.select(['object_id', 'h'] + tcols), ev.select(['object_id', 'h'] + tcols)
    hp, he = pool['h'].to_numpy(), ev['h'].to_numpy()
    oe = ev['object_id'].to_numpy()
    start = int((np.datetime64(args.start) - np.datetime64('2019-01-01')) / np.timedelta64(1, 'h'))
    edges = list(range(start, int(he.max()) - args.fold * DAY // 2, args.fold * DAY))
    print(f'обучающий пул {Xp.shape}, прогон {Xe.shape}, отрезков {len(edges)}, '
          f'{round(time.time() - t0)} с', flush=True)

    rows = []
    for tp in types:
        yp, ye = (pool[f'next_{tp}'].to_numpy() <= H), (ev[f'next_{tp}'].to_numpy() <= H)
        yp, ye = yp.astype(np.int8), ye.astype(np.int8)
        nxt_e = ev[f'next_{tp}'].to_numpy()
        params = train.tuned(args.model, tp, '', H) or {}
        if args.pw != 1.0:
            params = dict(params, scale_pos_weight=args.pw)
        for st in args.strategies.split(','):
            pred = np.zeros(len(he), np.float32)
            alarm = np.zeros(len(he), bool)
            prev, trees, fitted, thr0 = None, [], 0, None
            for t in edges:
                cut = t - H                       # метка обучающей строки не должна видеть отрезок
                lo = None
                if st.startswith('d') and st[1:].isdigit():
                    lo = cut - int(st[1:]) * DAY
                elif st == 'warm' and prev is not None:
                    lo = cut - args.fold * DAY    # только то, что появилось с прошлого раза
                vlo = cut - VAL_DAYS * DAY
                tr = pick(hp, lo, vlo)
                va = pick(hp, vlo, cut)
                if tr.sum() < 5000 or va.sum() < 500 or yp[tr].sum() < 20 or yp[va].sum() < 5:
                    continue
                w = weights(hp[tr], cut, int(st[5:])) if st.startswith('decay') else None
                if st == 'frozen' and prev is not None:
                    predict, n = prev, trees[-1]          # модель не трогаем, порог ниже пересчитаем
                else:
                    model, predict, n = fold_fit(args.model, Xp[tr], yp[tr], w, Xp[va], yp[va],
                                                 params, prev if st == 'warm' else None,
                                                 args.rounds, args.early)
                    fitted += 1
                    if st == 'warm':
                        prev = model
                    elif st == 'frozen':
                        prev = predict
                trees.append(n)
                # порог по бюджету диспетчера, снятый на тех же последних двух неделях
                if st == 'thrfix' and thr0 is not None:
                    thr = thr0                            # порог заморожен, модель переобучается
                else:
                    pv = predict(Xp[va])
                    thr = metrics.threshold_for_rate(pv, args.budget, int(va.sum()),
                                                     VAL_DAYS * (args.step if args.step else 1))
                    thr0 = thr
                sl = (he >= t) & (he < t + args.fold * DAY)
                if not sl.any():
                    continue
                ps = predict(Xe[sl])
                pred[sl] = ps
                alarm[sl] = ps >= thr
            if not fitted:
                print(f'  {config.TYPE_NAMES[tp]:18} {st:9} нет данных', flush=True)
                continue
            seen = pred > 0
            m = metrics.evaluate(oe[seen], he[seen], nxt_e[seen], pred[seen],
                                 float(np.min(pred[alarm & seen])) if (alarm & seen).any() else 1.1,
                                 H, metrics.RUN_CAP)
            sig, true = metrics.signals(oe[seen], he[seen], ye[seen], alarm[seen], args.gap)
            days = float(seen.sum()) / max(len(np.unique(oe)), 1) / DAY
            rows.append({'тип': config.TYPE_NAMES[tp], 'стратегия': st, 'переобучений': fitted,
                         'деревьев': int(np.mean(trees)), 'PR-AUC': round(float(
                             metrics.average_precision_score(ye[seen], pred[seen])), 3),
                         'сигналов': sig, 'ложных': sig - true,
                         'ложных в сутки': round((sig - true) / max(days, 1), 2),
                         'доля верных': round(true / sig, 3) if sig else float('nan'),
                         'поймано эпизодов': m['caught'], 'эпизодов': m['episodes']})
            r = rows[-1]
            print(f"  {r['тип']:18} {st:9} PR-AUC {r['PR-AUC']:.3f} | сигналов {r['сигналов']:5} "
                  f"ложных {r['ложных']:5} ({r['ложных в сутки']}/сут) | доля верных "
                  f"{r['доля верных']:.3f} | эпизодов {r['поймано эпизодов']}/{r['эпизодов']} | "
                  f"{r['переобучений']} переобучений, {r['деревьев']} дер.", flush=True)
            dump(rows, args)

    print(f'готово за {round(time.time() - t0)} с → {args.out}.md', flush=True)


if __name__ == '__main__':
    main()
