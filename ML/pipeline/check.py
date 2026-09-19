"""Сверка витрины с журналом (Ф3-1): случайные объекто-часы пересчитываются прямыми запросами к ev/inc.

Признак на момент прогноза stamp берёт события с ts < stamp; цель — первый эпизод с t0 ≥ stamp.
Совпадение значений витрины и пересчёта значит, что окна не захватывают будущее.

    python check.py [число строк]
"""
import sys

import numpy as np
import polars as pl

import config
from features import KNOWN

FEAT = config.WORK / 'features'


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    all_rows = (pl.scan_parquet(FEAT / '*.parquet').filter(pl.col('events_24h') > 0)
                .select('object_id', 'ts', 'h', 'smoke_24h', 'events_168h', 'gas_max_24h', 'temp_mean_24h',
                        'trig_fire_24h', 'onset_equipment_168h', 'since_event', 'next_fire', 'next_equipment')
                .collect())
    # половина — случайные часы, половина — рядом с пожарными срабатываниями, где окна не пустые
    busy = all_rows.filter((pl.col('trig_fire_24h') > 0) | (pl.col('next_fire') <= 24))
    rows = pl.concat([all_rows.sample(n - n // 2, seed=7), busy.sample(n // 2, seed=7)]).sort('ts')
    del all_rows, busy
    con = config.connect(read_only=True)
    checks = []
    for r in rows.iter_rows(named=True):
        o, ts = r['object_id'], r['ts']
        q = con.sql(f"""
            WITH p AS (SELECT {o} AS o, TIMESTAMP '{ts}' AS t)
            SELECT
              (SELECT count(*) FROM ev, p WHERE object_id = o AND ts >= t - INTERVAL 24 HOUR AND ts < t
                 AND stype = 'Датчик дыма' AND state = 'Обнаружен дым') AS smoke_24h,
              (SELECT count(*) FROM ev, p WHERE object_id = o AND ts >= t - INTERVAL 168 HOUR AND ts < t) AS events_168h,
              (SELECT max(num) FROM ev, p WHERE object_id = o AND ts >= t - INTERVAL 24 HOUR AND ts < t
                 AND stype = 'Газовый датчик' AND num >= 0 AND num < 327.68) AS gas_max_24h,
              (SELECT avg(num) FROM ev, p WHERE object_id = o AND ts >= t - INTERVAL 24 HOUR AND ts < t
                 AND stype = 'Датчик температуры' AND num BETWEEN -40 AND 80) AS temp_mean_24h,
              (SELECT count(*) FROM trig, p WHERE object_id = o AND type = 'fire'
                 AND ts + INTERVAL '{KNOWN['fire']}' >= date_trunc('hour', t) - INTERVAL 24 HOUR
                 AND ts + INTERVAL '{KNOWN['fire']}' < t) AS trig_fire_24h,
              (SELECT count(*) FROM inc, p WHERE object_id = o AND type = 'equipment' AND noise IS NULL
                 AND t0 >= t - INTERVAL 168 HOUR AND t0 < t) AS onset_equipment_168h,
              (SELECT date_diff('hour', max(ts), any_value(t) - INTERVAL 1 HOUR) FROM ev, p WHERE object_id = o AND ts < t
                 AND ts >= t - INTERVAL 2160 HOUR) AS since_event,
              (SELECT date_diff('hour', any_value(t) - INTERVAL 1 HOUR, min(t0)) FROM inc, p WHERE object_id = o AND type = 'fire'
                 AND noise IS NULL AND t0 >= t) AS next_fire,
              (SELECT date_diff('hour', any_value(t) - INTERVAL 1 HOUR, min(t0)) FROM inc, p WHERE object_id = o
                 AND type = 'equipment' AND noise IS NULL AND t0 >= t) AS next_equipment""").fetchone()
        names = ['smoke_24h', 'events_168h', 'gas_max_24h', 'temp_mean_24h', 'trig_fire_24h',
                 'onset_equipment_168h', 'since_event', 'next_fire', 'next_equipment']
        for name, v in zip(names, q):
            got = r[name]
            if name.startswith('next_') and (v is None or v >= 721):
                v = 721
            if name == 'since_event' and v is None:
                v = 2160
            ok = (v is None and (got is None or np.isnan(got))) or (v is not None and got is not None
                                                                   and abs(float(got) - float(v)) <= 1e-3 * max(1, abs(v)))
            checks.append(ok)
            if not ok:
                print(f'  расхождение: объект {o}, {ts}, {name}: витрина {got}, журнал {v}')
        print(f'объект {o}, прогноз на {ts}: ' + ', '.join(f'{k}={r[k]}' for k in names))
    print(f'совпало {sum(checks)} из {len(checks)}')


if __name__ == '__main__':
    main()
