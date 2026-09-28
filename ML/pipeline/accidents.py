"""Аварии и слепота по факту (INTEGRATION.md §13.11): аномальная температура, маршрут нарушителя,
потеря связи и питания объекта.

Модели здесь нет — правила по журналу, как у канала «по факту» (factalert.py). Вход — соединение
DuckDB с `ev`, `obj`, `ch` и разметкой `labels.build` (guard, obj3, inc). Одни и те же функции
считают и живой такт сервиса (service/fact.py, журнал до момента t), и ретропрогон по истории.

1. Температура. Показание датчика приходит примерно раз в сутки, поэтому база канала — медиана его
   предыдущих часовых отсчётов, а не окна по времени. Аномалия — выход за норму заказчика (+3…+40)
   и отклонение от своей базы не меньше DEVIATION: одна граница ловит каналы у входа, которые зимой
   всегда около нуля (4 108 канало-часов против 564). Эпизод — прогоны аномальных отсчётов каналов
   одного направления, перекрытые по времени с паузой до EPISODE_GAP. Объявление — со второго канала
   или со второго отсчёта подряд.
2. Маршрут. Эпизод проникновения — из разметки; маршрут — сработки охраны объекта от начала эпизода
   до снятия охраны в коллекторе или до last_at + ROUTE_TAIL, повтор датчика подряд схлопнут.
3. Слепота. Молчание по времени слепотой не считается (журнал пишет по изменению). `link` — за
   LINK_WINDOW «Неопределен» пришло от LINK_SHARE каналов объекта; `power` — в «Обесточен» стоит
   POWER_SHARE фаз объекта. Конец — RECOVER_SHARE каналов (фаз) снова в рабочем состоянии.
   Объявление — если держится BLIND_CONFIRM.

    python accidents.py
"""
import config
import labels

# 1. Температура
TEMP_STYPE = 'Датчик температуры'
STUB_LO, STUB_HI = -40, 80     # Н5: за этими границами не показание, а заглушка отказа (labels.TRIGGERS)
BASE_N = 48                    # отсчётов в базе канала: около полутора месяцев при отсчёте в сутки
BASE_MIN = 24                  # меньше — канал не оценивается: база ещё не устоялась
BAND = (3, 40)                 # °C, норма заказчика: состояние «В норме от +3 до +40»
DEVIATION = 10                 # °C от базы канала: при 10 холод — 349 объекто-суток за 2022–2026, 37 на 2+ каналах
CONFIRM_CHANNELS = 2           # объявление, когда аномалию дал второй канал объекта
CONFIRM_READINGS = 2           # … или один канал держит её столько часовых отсчётов подряд

# 2. Маршрут нарушителя
ROUTE_POINTS = f"""((stype IN {labels.OPENINGS[:-1]}, 'КД АВ') AND state = 'Не замкнут')
                   OR (stype = 'Датчик движения' AND state = 'Обнаружено движение'))"""
ROUTE_TAIL = '30 minutes'      # после последнего триггера эпизода маршрут ещё дописывается столько
ROUTE_MAX = 50                 # точек в маршруте, свежие остаются; у 90% эпизодов разных датчиков до 32

# 3. Слепота
UNDEFINED = "('Неопределен', 'Не определено')"
LINK_EXCLUDE = 'Состояние УИР-Р'   # стоит в «Неопределен» постоянно (2,2 млн строк) — не признак связи
LINK_SHARE = 0.8               # доля каналов объекта: 958 эпизодов на 67 объектах за 2022–2026
LINK_WINDOW_MIN = 10
LINK_WINDOW = f'{LINK_WINDOW_MIN} minutes'
PHASE_STYPE = 'Состояние фазы'
POWER_SHARE = 0.95             # доля фаз в «Обесточен»: штатно около половины фаз резервные и обесточены
POWER_BEFORE = '30 minutes'    # событие питания в коллекторе так незадолго до потери связи — причина power
RECOVER_SHARE = 0.5            # столько каналов (фаз) снова в рабочем состоянии — слепота кончилась
BLIND_CONFIRM_MIN = 30
BLIND_CONFIRM = f'{BLIND_CONFIRM_MIN} minutes'  # мигание короче не объявляется: четверть потерь связи проходит за 10 мин
BLIND_LOOKBACK = 7             # сут журнала для слепоты в живом такте

TYPES = ('temperature', 'blind')


def temperature(con) -> None:
    """Таблица `acc_temp`: эпизоды аномальной температуры — object_id, collector_id, direction,
    t0, t1 (часы первого и последнего аномального отсчёта), confirm_h (час, с которого эпизод
    объявляется, NULL — не подтверждён) и channels — последний аномальный отсчёт каждого канала."""
    lo, hi = BAND
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE acc_temp_r AS
        WITH r AS (SELECT object_id, channel_id, date_trunc('hour', ts) AS h, median(num) AS v
                   FROM ev WHERE stype = '{TEMP_STYPE}' AND num BETWEEN {STUB_LO} AND {STUB_HI} GROUP BY ALL),
             b AS (SELECT *, median(v) OVER w AS base, count(v) OVER w AS nb FROM r
                   WINDOW w AS (PARTITION BY channel_id ORDER BY h
                                ROWS BETWEEN {BASE_N} PRECEDING AND 1 PRECEDING)),
             a AS (SELECT *, CASE WHEN nb >= {BASE_MIN} AND v > {hi} AND v - base >= {DEVIATION} THEN 'hot'
                                  WHEN nb >= {BASE_MIN} AND v < {lo} AND base - v >= {DEVIATION} THEN 'cold'
                             END AS direction FROM b),
             -- прогон канала: отсчёты подряд одного направления
             k AS (SELECT *, row_number() OVER (PARTITION BY channel_id ORDER BY h)
                             - row_number() OVER (PARTITION BY channel_id, direction ORDER BY h) AS run FROM a)
        SELECT object_id, channel_id, direction, run, h, v, base,
               row_number() OVER (PARTITION BY channel_id, direction, run ORDER BY h) AS kth,
               min(h) OVER (PARTITION BY channel_id, direction, run) AS r0,
               max(h) OVER (PARTITION BY channel_id, direction, run) AS r1
        FROM k WHERE direction IS NOT NULL""")
    # прогоны объекта одного направления, перекрытые по времени с паузой до EPISODE_GAP, — эпизод
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE acc_temp AS
        WITH runs AS (SELECT DISTINCT object_id, channel_id, direction, run, r0, r1 FROM acc_temp_r),
             m AS (SELECT *, CASE WHEN r0 > max(r1) OVER w + INTERVAL {labels.EPISODE_GAP}
                                       OR max(r1) OVER w IS NULL THEN 1 ELSE 0 END AS new_ep
                   FROM runs WINDOW w AS (PARTITION BY object_id, direction ORDER BY r0, channel_id
                                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)),
             e AS (SELECT *, sum(new_ep) OVER (PARTITION BY object_id, direction ORDER BY r0, channel_id) AS ep FROM m),
             x AS (SELECT a.*, e.ep FROM acc_temp_r a JOIN e USING (object_id, channel_id, direction, run)),
             f AS (SELECT object_id, direction, ep, channel_id, min(h) AS first_h FROM x GROUP BY ALL),
             c2 AS (SELECT object_id, direction, ep, first_h AS h2 FROM f
                    QUALIFY row_number() OVER (PARTITION BY object_id, direction, ep ORDER BY first_h, channel_id)
                            = {CONFIRM_CHANNELS}),
             last AS (SELECT object_id, direction, ep,
                             list({{'sensor_id': channel_id, 'value': round(v, 1), 'baseline': round(base, 1),
                                   'at': h}} ORDER BY channel_id) AS channels
                      FROM (SELECT * FROM x QUALIFY row_number() OVER (
                                PARTITION BY object_id, direction, ep, channel_id ORDER BY h DESC) = 1)
                      GROUP BY ALL),
             g AS (SELECT object_id, direction, ep, min(h) AS t0, max(h) AS t1,
                          min(h) FILTER (WHERE kth = {CONFIRM_READINGS}) AS hr FROM x GROUP BY ALL)
        SELECT g.object_id, o.collector_id, g.direction, g.t0, g.t1,
               CASE WHEN c2.h2 IS NULL THEN g.hr WHEN g.hr IS NULL THEN c2.h2 ELSE least(g.hr, c2.h2) END AS confirm_h,
               l.channels
        FROM g JOIN obj3 o USING (object_id)
        LEFT JOIN c2 USING (object_id, direction, ep) JOIN last l USING (object_id, direction, ep)""")


def routes(con, src: str) -> dict:
    """(object_id, t0) → маршрут: для эпизодов проникновения из `src` (SQL с object_id, collector_id,
    t0, t1) — сработки охраны объекта по времени. Конец — снятие охраны в коллекторе после начала или
    t1 + ROUTE_TAIL; журнал `ev` уже обрезан по моменту такта."""
    rows = con.sql(f"""
        WITH e AS (SELECT DISTINCT object_id, collector_id, t0, t1 FROM ({src})),
             w AS (SELECT e.*, least(coalesce(d.ts, e.t1 + INTERVAL {ROUTE_TAIL}),
                                     e.t1 + INTERVAL {ROUTE_TAIL}) AS te
                   FROM e ASOF LEFT JOIN (SELECT * FROM guard WHERE NOT armed) d
                        ON d.collector_id = e.collector_id AND e.t0 < d.ts),
             p AS (SELECT w.object_id, w.t0, v.channel_id, v.stype, v.ts
                   FROM w JOIN ev v ON v.object_id = w.object_id AND v.ts >= w.t0 AND v.ts <= w.te
                   WHERE {ROUTE_POINTS}),
             q AS (SELECT *, lag(channel_id) OVER (PARTITION BY object_id, t0 ORDER BY ts, channel_id) AS prev FROM p),
             r AS (SELECT *, row_number() OVER (PARTITION BY object_id, t0 ORDER BY ts DESC, channel_id DESC) AS k
                   FROM q WHERE prev IS DISTINCT FROM channel_id)
        SELECT object_id, t0, list({{'sensor_id': channel_id, 'stype': stype, 'at': ts}} ORDER BY ts, channel_id)
        FROM r WHERE k <= {ROUTE_MAX} GROUP BY ALL""").fetchall()
    return {(int(o), t0): pts for o, t0, pts in rows}


def blind(con, since=None) -> None:
    """Таблица `acc_blind`: эпизоды слепоты — object_id, collector_id, cause, share, t0, t1 (NULL —
    ещё идёт). Точка слепоты — момент, когда доля каналов (фаз) достигла порога; точка возврата —
    RECOVER_SHARE каналов в рабочем состоянии. Эпизод — от первой точки слепоты после возврата до
    следующего возврата. `since` — журнал не раньше этого момента (живой такт: слепота дольше
    BLIND_LOOKBACK видна и по молчанию каналов в воронке)."""
    lim = '' if since is None else f"AND ts >= TIMESTAMP '{since}'"
    # возврат связи — пачка рабочих состояний; окно не скользящее, а два сдвинутых на половину
    # ведра: скользящий count(DISTINCT) по всему журналу (250 млн строк) не считается за разумное время
    buckets = ' UNION ALL '.join(
        f"""SELECT object_id, max(ts) AS ts, count(DISTINCT channel_id) AS k FROM ev
            WHERE stype <> '{LINK_EXCLUDE}' {lim}
              AND (state NOT IN {UNDEFINED} OR state IS NULL AND num IS NOT NULL)
            GROUP BY object_id, time_bucket(INTERVAL {LINK_WINDOW}, ts, INTERVAL '{off} minutes')"""
        for off in (0, LINK_WINDOW_MIN // 2))
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE acc_blind_pts AS
        WITH n AS (SELECT object_id, count(*) AS n FROM ch WHERE stype <> '{LINK_EXCLUDE}' GROUP BY 1),
             u AS (SELECT object_id, channel_id, ts FROM ev
                   WHERE stype <> '{LINK_EXCLUDE}' AND state IN {UNDEFINED} {lim}),
             c AS (SELECT *, count(DISTINCT channel_id) OVER (PARTITION BY object_id ORDER BY ts
                                 RANGE BETWEEN INTERVAL {LINK_WINDOW} PRECEDING AND CURRENT ROW) AS k FROM u),
             lost AS (SELECT c.object_id, c.ts, false AS back, k / n.n AS share
                      FROM c JOIN n USING (object_id) WHERE k >= {LINK_SHARE} * n.n),
             wk AS (SELECT w.object_id, w.ts, true AS back, k / n.n AS share
                    FROM ({buckets}) w JOIN n USING (object_id) WHERE k >= {RECOVER_SHARE} * n.n),
             -- фаза: последнее известное состояние; число обесточенных — сумма переходов
             ph AS (SELECT object_id, channel_id, ts, (state = 'Обесточен')::INT AS off FROM ev
                    WHERE stype = '{PHASE_STYPE}' AND state IN ('Обесточен', 'Есть питание') {lim}),
             np AS (SELECT object_id, count(DISTINCT channel_id) AS n FROM ph GROUP BY 1),
             d AS (SELECT *, off - coalesce(lag(off) OVER (PARTITION BY channel_id ORDER BY ts), 0) AS delta FROM ph),
             s AS (SELECT object_id, ts, sum(delta) OVER (PARTITION BY object_id ORDER BY ts, channel_id) AS k FROM d),
             pw AS (SELECT s.object_id, s.ts, k / np.n <= {RECOVER_SHARE} AS back, k / np.n AS share
                    FROM s JOIN np USING (object_id)
                    WHERE k >= {POWER_SHARE} * np.n OR k <= {RECOVER_SHARE} * np.n)
        SELECT object_id, ts, back, share, 'link' AS cause FROM lost
        UNION ALL SELECT object_id, ts, back, share, 'link' FROM wk
        UNION ALL SELECT object_id, ts, back, share, 'power' FROM pw""")
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE acc_blind AS
        WITH q AS (SELECT *, lag(back) OVER (PARTITION BY object_id, cause ORDER BY ts, back) AS prev
                   FROM acc_blind_pts),
             st AS (SELECT object_id, cause, ts AS t0, share FROM q WHERE NOT back AND prev IS DISTINCT FROM false),
             e AS (SELECT st.*, b.ts AS t1 FROM st ASOF LEFT JOIN (SELECT * FROM acc_blind_pts WHERE back) b
                        ON b.object_id = st.object_id AND b.cause = st.cause AND st.t0 < b.ts),
             -- событие питания в коллекторе незадолго до потери связи — причина питание
             pe AS (SELECT o.collector_id, e.ts FROM ev e JOIN obj3 o USING (object_id)
                    WHERE ((e.stype = '{PHASE_STYPE}' AND e.state = 'Обесточен')
                           OR (e.stype = 'ИБП' AND e.state = 'Питание от батарей')) {lim.replace('ts', 'e.ts')})
        SELECT e.object_id, o.collector_id,
               CASE WHEN e.cause = 'link' AND p.ts >= e.t0 - INTERVAL {POWER_BEFORE} THEN 'power' ELSE e.cause END AS cause,
               round(e.share, 2) AS share, e.t0, e.t1
        FROM e JOIN obj3 o USING (object_id)
        ASOF LEFT JOIN pe p ON p.collector_id = o.collector_id AND e.t0 >= p.ts""")


def fire_temp_only(con) -> None:
    """Таблица `acc_fire_temp`: эпизоды пожара из одних датчиков температуры. Метка для модели прежняя
    (labels.TRIGGERS), а по факту такой эпизод объявляется как `temperature`, а не `fire`."""
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE acc_fire_temp AS
        SELECT i.object_id, i.t0 FROM inc i JOIN trig g
          ON g.object_id = i.object_id AND g.type = 'fire' AND g.ts BETWEEN i.t0 AND i.t1
        WHERE i.type = 'fire' GROUP BY ALL HAVING bool_and(g.stype = '{TEMP_STYPE}')""")


def main() -> None:
    con = config.connect(read_only=True)   # разметка — из последнего `python labels.py` в этой базе
    temperature(con)
    blind(con)
    fire_temp_only(con)
    print('температура — объявлений в год (эпизоды с подтверждением):')
    con.sql("""SELECT year(t0) AS год, direction, count(*) AS эпизодов,
                            count(confirm_h) AS объявлено, count(DISTINCT object_id) AS объектов
                     FROM acc_temp GROUP BY ALL ORDER BY 1, 2""").show(max_rows=100)
    print('слепота — объявлений в год (держится не меньше BLIND_CONFIRM):')
    con.sql(f"""SELECT year(t0) AS год, cause, count(*) AS эпизодов,
                             count(*) FILTER (WHERE t1 IS NULL OR t1 - t0 >= INTERVAL {BLIND_CONFIRM}) AS объявлено,
                             count(DISTINCT object_id) AS объектов,
                             round(median(epoch(t1 - t0) / 60)) AS медиана_мин
                      FROM acc_blind GROUP BY ALL ORDER BY 1, 2""").show(max_rows=100)
    print(con.sql("""SELECT count(*) AS пожар_только_температура,
                            (SELECT count(*) FROM inc WHERE type = 'fire') AS эпизодов_пожара FROM acc_fire_temp"""))
    r = routes(con, "SELECT object_id, collector_id, t0, t1 FROM inc WHERE type = 'intrusion' AND noise IS NULL")
    sizes = sorted(len({p['sensor_id'] for p in v}) for v in r.values())
    if sizes:
        print(f'маршрут: эпизодов {len(sizes)}, разных датчиков медиана {sizes[len(sizes) // 2]}, '
              f'90% — до {sizes[int(len(sizes) * 0.9)]}, максимум {sizes[-1]}')


if __name__ == '__main__':
    main()
