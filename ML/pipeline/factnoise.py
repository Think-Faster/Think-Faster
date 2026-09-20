"""Шаг 21. Отброшенный вариант: обученный фильтр ложных объявлений канала «по факту».

**Скрипт оставлен как запись тупика, а не как часть конвейера** (раздел 18). Здесь написано, что
он делал, почему проиграл и — главное — почему задача, которую он решал, оказалась несуществующей.

Требование к каналу «по факту»: полнота около 100% и ноль ложных объявлений. Правило `clean`
(раздел 10) к нулю не дотягивало — на тесте 2026 оно объявляло 20 шумовых эпизодов пожара и 6
подтопления. Гипотеза была, что оставшиеся отличимы по объекту и его истории: шумный эпизод идёт
там, где такие же уже были, и почти всегда одинок. Отсюда фильтр на признаках объекта и короткого
окна ожидания, обученный на эпизодах 2025 и применённый к 2026; рабочая точка снимается по оценкам
вне обучения, как в `cascade.py`.

**Он проиграл простому правилу**: у пожара 8 ложных в лучшей точке против нуля, у подтопления
374 ложных при «сохранить 1,000» — шумных эпизодов на обучении слишком мало, и дерево запоминает
объекты, а не признак шума.

**А потом выяснилось, что мерить было нечего.** Метка `inc.noise` в `labels.py` строится нарезкой
эпизодов с `PARTITION BY ..., noise IS NOT NULL`: помеченные триггеры собираются в свои эпизоды,
непомеченные — в свои. Значит «эпизод шумовой» тождественно «первый триггер помечен», и любое
правило, читающее `trig_noise`, воспроизводит метку точно. При `--wait 0` это видно прямо: ноль
ложных и полнота ровно 1,000 на всех шести типах. Сами Н7/Н8 — детерминированные правила по
журналу, так что канал просто применяет их сам; учить этому модель незачем.

Что осталось полезного — две строки внизу таблицы: два простых правила, с которыми сравнивался
фильтр, и признаки, на которые он смотрел. Запускать имеет смысл только чтобы воспроизвести
раздел 18.

    python factnoise.py
    python factnoise.py --wait 0          # видно вырождение: 0 ложных, полнота 1,000
"""
import argparse

import duckdb
import numpy as np

import config

# всё, что здесь перечислено, известно к моменту t0 + wait. Длительности эпизода и итогового
# числа каналов в списке нет специально: и то и другое считается по его концу, то есть по будущему
FEATS = ['ch10', 'tr10', 'marked10', 'prev_1h', 'prev_24h', 'prev_168h',
         'ep_30d', 'ep_180d', 'noise_180d', 'hour', 'dow']


def episodes(con, tp: str, years: tuple[int, ...], wait: int) -> dict[str, np.ndarray]:
    """Эпизоды типа с признаками, известными к моменту `t0 + wait` минут."""
    yrs = ', '.join(str(y) for y in years)
    q = f"""
        WITH e AS (SELECT object_id, type, t0, noise
                   FROM inc WHERE type = '{tp}' AND year(t0) IN ({yrs})
                     AND object_id IN (SELECT object_id FROM obj3))
        SELECT e.object_id, e.t0, e.noise IS NULL AS real_,
               hour(e.t0) AS hour, dayofweek(e.t0) AS dow,
               (SELECT count(DISTINCT g.channel_id) FROM trig g
                 WHERE g.object_id = e.object_id
                   AND g.ts BETWEEN e.t0 AND e.t0 + INTERVAL {wait} MINUTE) AS ch10,
               (SELECT count(*) FROM trig g
                 WHERE g.object_id = e.object_id
                   AND g.ts BETWEEN e.t0 AND e.t0 + INTERVAL {wait} MINUTE) AS tr10,
               (SELECT count(*) FROM trig_noise n
                 WHERE n.object_id = e.object_id AND n.type = e.type
                   AND n.ts BETWEEN e.t0 AND e.t0 + INTERVAL {wait} MINUTE) AS marked10,
               (SELECT count(*) FROM trig g
                 LEFT JOIN trig_noise n ON n.object_id = g.object_id
                   AND n.channel_id = g.channel_id AND n.ts = g.ts
                 WHERE g.object_id = e.object_id AND g.type = e.type AND n.noise IS NULL
                   AND g.ts BETWEEN e.t0 AND e.t0 + INTERVAL {wait} MINUTE) AS clean10,
               (SELECT count(*) FROM trig g
                 WHERE g.object_id = e.object_id AND g.type = e.type
                   AND g.ts >= e.t0 - INTERVAL 1 HOUR AND g.ts < e.t0) AS prev_1h,
               (SELECT count(*) FROM trig g
                 WHERE g.object_id = e.object_id AND g.type = e.type
                   AND g.ts >= e.t0 - INTERVAL 24 HOUR AND g.ts < e.t0) AS prev_24h,
               (SELECT count(*) FROM trig g
                 WHERE g.object_id = e.object_id AND g.type = e.type
                   AND g.ts >= e.t0 - INTERVAL 168 HOUR AND g.ts < e.t0) AS prev_168h,
               (SELECT count(*) FROM inc i
                 WHERE i.object_id = e.object_id AND i.type = e.type
                   AND i.t0 < e.t0 AND i.t0 >= e.t0 - INTERVAL 30 DAY) AS ep_30d,
               (SELECT count(*) FROM inc i
                 WHERE i.object_id = e.object_id AND i.type = e.type
                   AND i.t0 < e.t0 AND i.t0 >= e.t0 - INTERVAL 180 DAY) AS ep_180d,
               (SELECT coalesce(avg(CASE WHEN i.noise IS NULL THEN 0.0 ELSE 1.0 END), -1.0)
                  FROM inc i
                 WHERE i.object_id = e.object_id AND i.type = e.type
                   AND i.t0 < e.t0 AND i.t0 >= e.t0 - INTERVAL 180 DAY) AS noise_180d
        FROM e"""
    d = con.sql(q).fetchnumpy()
    return {k: np.ma.getdata(v) for k, v in d.items()}


def oof(X, y, p, folds: int) -> np.ndarray:
    """Оценки вне собственного обучения: блоками подряд, как в `cascade.py`."""
    import xgboost as xgb
    out = np.full(len(y), np.nan, np.float32)
    bounds = np.linspace(0, len(y), folds + 1).astype(int)
    for a, b in zip(bounds[:-1], bounds[1:]):
        te = np.zeros(len(y), bool)
        te[a:b] = True
        if y[~te].sum() < 5 or (~y[~te].astype(bool)).sum() < 5 or not te.any():
            continue
        m = xgb.train(p, xgb.DMatrix(X[~te], y[~te], feature_names=FEATS), num_boost_round=200)
        out[te] = m.predict(xgb.DMatrix(X[te], feature_names=FEATS))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--wait', type=int, default=10, help='минут ожидания перед решением')
    ap.add_argument('--keep', default='1.0,0.999,0.995,0.99,0.98',
                    help='какую долю настоящих эпизодов соглашаемся сохранить')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    import xgboost as xgb
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)

    print(f'Фильтр ложных объявлений канала «по факту». Обучен на эпизодах 2025, применён к 2026. '
          f'Решение принимается через {args.wait} мин после первого триггера; рабочая точка выбрана '
          f'по оценкам вне обучения.\n')
    print('| тип | правило | объявлено | из них ложных | пропущено настоящих | полнота |')
    print('|---|---|---:|---:|---:|---:|')
    p = {'objective': 'binary:logistic', 'eval_metric': 'aucpr', 'tree_method': 'hist',
         'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'min_child_weight': 20, 'reg_lambda': 5.0}
    for tp in args.types.split(','):
        name = config.TYPE_NAMES[tp]
        tr = episodes(con, tp, (2025,), args.wait)
        te = episodes(con, tp, (2026,), args.wait)
        ytr = tr['real_'].astype(np.int8)
        yte = te['real_'].astype(np.int8)
        # два простых правила на том же окне ожидания, с которыми сравнивается фильтр
        for tag, said in (('есть чистый триггер (`clean` раздела 10)', te['clean10'] > 0),
                          ('за окно ожидания ни одной пометки шума', te['marked10'] == 0)):
            bad0 = int((said & (yte == 0)).sum())
            miss0 = int(((~said) & (yte == 1)).sum())
            print(f'| {name} | {tag} | {int(said.sum())} | {bad0} | {miss0} | '
                  f'{1 - miss0 / max(int(yte.sum()), 1):.3f} |'.replace('.', ','))

        noisy = int((ytr == 0).sum())
        if len(yte) < 20 or ytr.sum() < 20 or noisy < 5:
            print(f'| {name} | *шумных эпизодов на обучении {noisy} из {len(ytr)} — фильтру не на '
                  f'чем учиться, да и нечего убирать* | | | | |')
            continue

        o = np.argsort(tr['t0'], kind='stable')
        Xtr = np.column_stack([tr[f][o].astype(np.float32) for f in FEATS])
        Xte = np.column_stack([te[f].astype(np.float32) for f in FEATS])
        q = oof(Xtr, ytr[o], p, args.folds)
        m = xgb.train(p, xgb.DMatrix(Xtr, ytr[o], feature_names=FEATS), num_boost_round=200)
        qs = m.predict(xgb.DMatrix(Xte, feature_names=FEATS))
        pos = q[(ytr[o] == 1) & np.isfinite(q)]
        for keep in [float(x) for x in args.keep.split(',')]:
            t = float(np.quantile(pos, 1 - keep)) if keep < 1 else float(pos.min())
            said = qs >= t
            bad = int((said & (yte == 0)).sum())
            miss = int(((~said) & (yte == 1)).sum())
            print(f'| {name} | фильтр, сохранить {keep:.3f} | {int(said.sum())} | {bad} | {miss} | '
                  f'{1 - miss / max(int(yte.sum()), 1):.3f} |'.replace('.', ','))
        imp = sorted(m.get_score(importance_type='gain').items(), key=lambda kv: -kv[1])[:5]
        print(f"| {name} | *на что смотрит фильтр* | {', '.join(k for k, _ in imp)} | | | |")
    con.close()


if __name__ == '__main__':
    main()
