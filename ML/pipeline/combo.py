"""Шаг 10. Связки датчиков: сработало двое рядом, а не один где-то на объекте.

В витрине счётчики состояний идут по объекту целиком: «за сутки 3 дыма и 2 тепловых» — но дым
в одном конце коллектора и тепловой в другом к одному и тому же пожару отношения не имеют.
Имена каналов в справочнике содержат пикет (`Дым ПК204`, `Темп. ВШ ПК88,5`) — это положение
датчика вдоль коллектора. По нему можно потребовать, чтобы сработавшие датчики стояли рядом.

Связка = два разных канала разных семейств, сработали не дальше PAIR_W часов друг от друга и не
дальше PAIR_D пикетов друг от друга. Момент, когда связка становится известна, — второе из двух
срабатываний, в будущее она не смотрит.

Вторая идея из того же справочника — разброс сработавших датчиков по пикетам. Пожар местный: горит
в одном месте, сработает куст соседних датчиков. Неисправный шлейф сыпет по всему коллектору сразу.
Разброс (`span`) и число разных каналов (`nch`) в окне это различают.

    python combo.py build     # связки соседних датчиков → work/combo.parquet
    python combo.py spread    # число каналов и разброс по пикетам → work/combo_spread.parquet
    python combo.py rules     # связка как правило диспетчера против одиночного триггера

Признаки подключаются к обучению флагом `train.py --combo pairs|spread|both`, витрина не пересобирается.
"""
import sys
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
from features import NH, T0, dense, hour_of, window_sum

PAIR_W = 6   # ч: столько может пройти между двумя срабатываниями связки
PAIR_D = 5   # пикетов: дальше датчики считаются разными местами
WINDOWS = (6, 24, 168)

# Семейства срабатываний: что именно считаем сработавшим датчиком
FAM = {
    'smoke': "stype = 'Датчик дыма' AND state = 'Обнаружен дым'",
    'heat': "stype = 'Тепловой датчик' AND state = 'Не замкнут'",
    'temp_hi': "stype = 'Датчик температуры' AND state = 'Температура выше 40ºC'",
    'manual': "stype = 'Ручной извещатель' AND state IN ('Не замкнут', 'Рычаг сдернут')",
    'gas': "stype = 'Газовый датчик' AND state = 'Обнаружен газ'",
    'flood': "stype = 'Датчик затопления' AND state = 'Не замкнут'",
    'pump_bad': "stype = 'Состояние насоса' AND state IN ('Затоплен', 'Неисправен', 'Обесточен')",
    'phase_off': "stype = 'Состояние фазы' AND state = 'Обесточен'",
    'ups_batt': "stype = 'ИБП' AND state IN ('Питание от батарей', 'Батарея разряжена', 'Батарея неисправна')",
    'unit_bad': "stype IN ('Состояние вентилятора', 'Состояние насоса') AND state IN ('Неисправен', 'Обесточен')",
    'door': "stype IN ('КД Дверь', 'КД АВ') AND state = 'Не замкнут'",
    'hatch': "stype IN ('КД Люк', '9-секционный люк', 'Стекло') AND state = 'Не замкнут'",
    'motion': "state = 'Обнаружено движение'",
    'fault': "state IN ('Неисправен', 'Не определено', 'Неопределен')",
}
# Пары по типам происшествий: что с чем имеет смысл связывать
PAIRS = [
    ('fire', 'smoke', 'heat'), ('fire', 'smoke', 'temp_hi'), ('fire', 'heat', 'temp_hi'),
    ('fire', 'smoke', 'smoke'), ('fire', 'smoke', 'manual'),
    ('gas', 'gas', 'gas'), ('gas', 'gas', 'unit_bad'),
    ('flood', 'flood', 'pump_bad'), ('flood', 'flood', 'flood'), ('flood', 'pump_bad', 'unit_bad'),
    ('equipment', 'phase_off', 'ups_batt'), ('equipment', 'phase_off', 'unit_bad'),
    ('equipment', 'unit_bad', 'unit_bad'),
    ('sensor', 'fault', 'fault'),
    ('intrusion', 'door', 'motion'), ('intrusion', 'hatch', 'motion'), ('intrusion', 'door', 'door'),
]
NAMES = [f'{tp}_{a}_{b}' for tp, a, b in PAIRS]
# Семейства, для которых считаем разброс: там, где датчиков много и они стоят вдоль коллектора
SPREAD = ['smoke', 'gas', 'flood', 'fault', 'unit_bad', 'phase_off']
SPREAD_W = (6, 24)
OUT = config.WORK / 'combo.parquet'
OUT_SPREAD = config.WORK / 'combo_spread.parquet'


def channels(con) -> None:
    """Канал → пикет из имени в справочнике. Каналов без пикета около 7 %, они выпадают из связок."""
    con.sql(r"""CREATE OR REPLACE TEMP TABLE ch_pk AS
                SELECT channel_id, object_id, stype,
                       try_cast(regexp_extract(name, 'ПК\s?(\d+)', 1) AS INT) AS pk
                FROM ch WHERE object_id IN (SELECT object_id FROM obj3)""")
    n, with_pk = con.sql('SELECT count(*), count(pk) FROM ch_pk').fetchone()
    print(f'каналов {n}, из них с пикетом {with_pk} ({with_pk / n:.0%})', flush=True)


def active(con) -> None:
    """Час × канал × семейство: когда датчик сработал. Дубли внутри часа схлопнуты."""
    case = ('CASE ' + ' '.join(f"WHEN {cond} THEN '{k}'" for k, cond in FAM.items()) + ' END'
            ).replace('stype', 'e.stype').replace('state', 'e.state')
    con.sql(f"""CREATE OR REPLACE TEMP TABLE act AS
                SELECT DISTINCT c.object_id, e.channel_id, c.pk, {case} AS fam, {hour_of('e.ts')} AS h
                FROM ev e JOIN ch_pk c USING (channel_id)
                WHERE c.pk IS NOT NULL AND e.state IS NOT NULL AND ({case}) IS NOT NULL""")
    d = con.sql('SELECT fam, count(*) AS n FROM act GROUP BY 1 ORDER BY 2 DESC').df()
    print(d.to_string(index=False), flush=True)


def pair_hours(con, a: str, b: str):
    """Часы, когда сработала связка a+b, и сколько разных пар каналов в ней участвовало."""
    same = 'AND x.channel_id < y.channel_id' if a == b else 'AND x.channel_id <> y.channel_id'
    return con.sql(f"""SELECT x.object_id, greatest(x.h, y.h) AS h,
                              count(DISTINCT least(x.channel_id, y.channel_id)
                                    || '-' || greatest(x.channel_id, y.channel_id)) AS n
                       FROM act x JOIN act y ON x.object_id = y.object_id
                        AND x.fam = '{a}' AND y.fam = '{b}'
                        AND y.h BETWEEN x.h - {PAIR_W} AND x.h + {PAIR_W}
                        AND abs(x.pk - y.pk) <= {PAIR_D} {same}
                       GROUP BY ALL""").fetchnumpy()


def build() -> None:
    t = time.time()
    con = config.connect(read_only=True)
    channels(con)
    active(con)
    objects = [r[0] for r in con.sql('SELECT object_id FROM obj3 ORDER BY object_id').fetchall()]
    oi = {o: i for i, o in enumerate(objects)}
    raw = np.zeros((len(objects), NH, len(PAIRS)), np.float32)
    for j, (tp, a, b) in enumerate(PAIRS):
        d = pair_hours(con, a, b)
        h = dense(d['h']).astype(np.int64)
        o = np.array([oi[x] for x in d['object_id']], np.int64)
        keep = (h >= 0) & (h < NH)
        raw[o[keep], h[keep], j] = dense(d['n']).astype(np.float32)[keep]
        print(f'  {NAMES[j]:26s} {int(keep.sum()):8d} часов со связкой, {time.time() - t:.0f} с', flush=True)
    con.close()

    cols: dict[str, np.ndarray] = {'object_id': [], 'h': []}
    for w in WINDOWS:
        for name in NAMES:
            cols[f'co_{name}_{w}h'] = []
    for o, object_id in enumerate(objects):
        hit = raw[o].any(axis=1)
        if not hit.any():
            continue
        # строки только там, где за 7 сут была хоть одна связка: остальное — нули, их добавит join
        sums = {w: np.stack([window_sum(raw[o, :, j], w) for j in range(len(PAIRS))], 1) for w in WINDOWS}
        keep = np.flatnonzero(sums[max(WINDOWS)].any(axis=1))
        cols['object_id'].append(np.full(len(keep), object_id, np.int32))
        cols['h'].append(keep.astype(np.int32))
        for w in WINDOWS:
            for j, name in enumerate(NAMES):
                cols[f'co_{name}_{w}h'].append(sums[w][keep, j])
    table = pa.table({k: np.concatenate(v) for k, v in cols.items()})
    pq.write_table(table, OUT, compression='zstd')
    print(f'{OUT}: {table.num_rows} строк, {table.num_columns - 2} признаков за {time.time() - t:.0f} с')


def spread() -> None:
    """Сколько разных каналов семейства сработало в окне и на сколько пикетов они разошлись."""
    t = time.time()
    con = config.connect(read_only=True)
    channels(con)
    active(con)
    frames = []
    for fam in SPREAD:
        for w in SPREAD_W:
            d = con.sql(f"""WITH f AS (SELECT * FROM act WHERE fam = '{fam}'),
                                 b AS (SELECT DISTINCT object_id, h + o AS h FROM f, range(0, {w}) t(o))
                            SELECT b.object_id, b.h,
                                   count(DISTINCT f.channel_id) AS nch, max(f.pk) - min(f.pk) AS span
                            FROM b JOIN f ON f.object_id = b.object_id
                             AND f.h BETWEEN b.h - {w} + 1 AND b.h
                            GROUP BY ALL""").pl()
            d = d.rename({'nch': f'sp_{fam}_nch_{w}h', 'span': f'sp_{fam}_span_{w}h'})
            frames.append(d.filter((pl.col('h') >= 0) & (pl.col('h') < NH)))
            print(f'  {fam}_{w}h: {len(frames[-1])} строк, {time.time() - t:.0f} с', flush=True)
    con.close()
    out = frames[0]
    for d in frames[1:]:
        out = out.join(d, on=['object_id', 'h'], how='full', coalesce=True)
    out = out.fill_null(0).with_columns(pl.col('object_id').cast(pl.Int32), pl.col('h').cast(pl.Int32))
    out.write_parquet(OUT_SPREAD, compression='zstd')
    print(f'{OUT_SPREAD}: {len(out)} строк, {len(out.columns) - 2} признаков за {time.time() - t:.0f} с')


def rules() -> None:
    """Связка как правило диспетчера: сравнение с одиночным триггером на проверке и тесте."""
    import json

    import polars as pl

    import metrics
    meta = json.loads((config.WORK / 'features' / 'meta.json').read_text(encoding='utf-8'))
    co = pl.read_parquet(OUT)
    H, cap = config.HORIZON, meta['next_cap']
    need = ['object_id', 'h'] + [f'next_{tp}' for tp in config.TYPES] + [f'trig_{tp}_24h' for tp in config.TYPES]
    fams = sorted({f for _, a, b in PAIRS for f in (a, b)})
    obj_cols = {'smoke': 'smoke_24h', 'heat': 'heat_24h', 'temp_hi': 'temp_hi_24h', 'manual': 'manual_24h',
                'gas': 'gas_24h', 'flood': 'flood_sensor_24h', 'pump_bad': 'pump_flooded_24h',
                'phase_off': 'phase_off_24h', 'ups_batt': 'ups_batt_24h', 'unit_bad': 'fan_fault_24h',
                'door': 'door_24h', 'hatch': 'hatch_24h', 'motion': 'motion_24h', 'fault': 'fault_other_24h'}
    need += [obj_cols[f] for f in fams]
    rows = []
    for split, year in (('проверка', 2025), ('тест', 2026)):
        df = (pl.scan_parquet(config.WORK / 'features' / f'{year}.parquet').select(need).collect()
              .join(co, on=['object_id', 'h'], how='left').fill_null(0))
        obj, hh = df['object_id'].to_numpy(), df['h'].to_numpy()
        for tp in config.TYPES:
            nx = df[f'next_{tp}'].to_numpy()
            variants = {'одиночный триггер': df[f'trig_{tp}_24h'].to_numpy().astype(np.float32)}
            pairs = [(a, b) for t2, a, b in PAIRS if t2 == tp]
            # «сработали два семейства где-то на объекте» — без пикетов
            anyp = np.zeros(len(df), np.float32)
            for a, b in pairs:
                anyp = np.maximum(anyp, np.minimum(df[obj_cols[a]].to_numpy(), df[obj_cols[b]].to_numpy()))
            variants['два семейства на объекте'] = anyp
            near = np.zeros(len(df), np.float32)
            for a, b in pairs:
                near = np.maximum(near, df[f'co_{tp}_{a}_{b}_24h'].to_numpy())
            variants['связка рядом (пикеты)'] = near
            for name, score in variants.items():
                thr = 0.5  # правило: «сработало» — значение больше нуля
                m = metrics.evaluate(obj, hh, nx, score, thr, H, cap)
                rows.append((split, config.TYPE_NAMES[tp], name, m['pr_auc'], m['precision'],
                             m['recall_episodes'], m['alarm_rate']))
    print('\n| выборка | тип | правило | PR-AUC | Precision | Recall (эп.) | доля часов с тревогой |')
    print('|---|---|---|---:|---:|---:|---:|')
    for r in rows:
        print(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.4f} |'
              .replace('.', ','))


if __name__ == '__main__':
    {'build': build, 'spread': spread, 'rules': rules}[sys.argv[1] if len(sys.argv) > 1 else 'build']()
