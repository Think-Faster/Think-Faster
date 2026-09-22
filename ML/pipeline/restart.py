"""Разметка «перезапуск после питания»: эпизод начался не позже RESTART после возврата питания на
том же объекте (раздел 45).

Н7 из анализа журнала (`docs/dataset/анализ.md`) — скачок питания и «Затоплен» сразу после него;
в `labels.py` он размечен узко: «Затоплен» за 60 с после события питания в коллекторе. Разбор
пропусков (раздел 44) показал, что тот же след шире: 69% эпизодов загазованности и 26% отказов
оборудования начинаются в первые 10 минут после «Есть питание» / «Питание от сети» / «Включен» на
том же объекте, против 10% и 5% в тот же час суток в другие дни.

Скрипт не трогает `inc` и основную цель: он дописывает в витрину цель `next_<тип>_nopw` — через
сколько часов следующий эпизод, не считая перезапусков, — и регистрирует её в `meta.json`. Дальше
её берут `train.py --target _nopw` и `operating.load_mix(..., label='_nopw')`.

    python restart.py            # дописать цель в витрину
    python restart.py --stats    # только доли перезапусков по типам и годам
"""
import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import config
import features as ft

RESTART = 10   # мин: эпизод так скоро после возврата питания на объекте — перезапуск
POWER = """(stype = 'Состояние фазы' AND state = 'Есть питание')
        OR (stype = 'ИБП' AND state = 'Питание от сети')
        OR (stype = 'Переключатель' AND state = 'Включен')"""


def marked(con):
    con.sql(f'CREATE OR REPLACE TEMP TABLE pw AS SELECT object_id, ts FROM ev WHERE {POWER}')
    con.sql(f"""CREATE OR REPLACE TEMP TABLE inc_rs AS
        SELECT e.object_id, e.type, e.t0,
               coalesce(date_diff('second', p.ts, e.t0) <= {RESTART * 60}, false) AS restart
        FROM (SELECT * FROM inc WHERE noise IS NULL) e
        ASOF LEFT JOIN pw p ON p.object_id = e.object_id AND p.ts <= e.t0""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', action='store_true')
    args = ap.parse_args()
    con = config.connect(read_only=True)
    marked(con)
    print(con.sql("""SELECT type, year(t0) AS год, count(*) AS эпизодов,
                            round(avg(restart::INT), 3) AS перезапусков
                     FROM inc_rs WHERE year(t0) >= 2022 GROUP BY ALL ORDER BY 1, 2""").df().to_string())
    if args.stats:
        return
    objects = [r[0] for r in con.sql('SELECT object_id FROM obj3 ORDER BY object_id').fetchall()]
    oi = {o: i for i, o in enumerate(objects)}
    targets = {}
    for tp in config.TYPES:
        d = con.sql(f"""SELECT object_id, {ft.hour_of('t0')} AS h, count(*) AS n FROM inc_rs
                        WHERE type = '{tp}' AND NOT restart
                          AND object_id IN (SELECT object_id FROM obj3) GROUP BY ALL""").fetchnumpy()
        arr = np.zeros((len(objects), ft.NH), np.float32)
        h = ft.dense(d['h']).astype(np.int64)
        keep = (h >= 0) & (h < ft.NH)
        arr[np.array([oi[x] for x in d['object_id']])[keep], h[keep]] = ft.dense(d['n'])[keep]
        targets[f'next_{tp}_nopw'] = np.stack([ft.hours_to_next(a) for a in arr])
    con.close()
    out = config.WORK / 'features'
    for path in sorted(out.glob('*.parquet')):
        t = pq.read_table(path)
        t = t.drop_columns([c for c in targets if c in t.column_names])
        o = np.array([oi[x] for x in t['object_id'].to_numpy()])
        hh = t['h'].to_numpy().astype(np.int64)
        for name, v in targets.items():
            t = t.append_column(name, pa.array(v[o, hh]))
        tmp = path.with_suffix('.tmp')
        pq.write_table(t, tmp)
        del t
        for _ in range(60):   # другой процесс может держать файл открытым на чтении
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(5)
        else:
            raise RuntimeError(f'{path} занят')
        print(path.name, flush=True)
    meta_path = out / 'meta.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    meta['targets'] = [c for c in meta['targets'] if not c.endswith('_nopw')] + list(targets)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
