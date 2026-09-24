"""Шаг 2. Разметка: триггеры → эпизоды инцидентов по типам, след выезда бригады.

Подтверждений инцидентов в данных нет (ответ 12), поэтому разметка косвенная:
1. Триггер — строка журнала, которая по смыслу состояния относится к одному из типов (TRIGGERS).
2. Эпизод — триггеры одного типа на одном объекте с паузой не больше EPISODE_GAP. Так дребезг (Н3)
   и пачки по многим каналам схлопываются в один инцидент; начало эпизода — момент инцидента.
3. Шум размечается, а не удаляется: массовый дым (Н8) и «Затоплен» в пачке с событием питания (Н7).
   В цель модели шумные эпизоды не входят. Метка Н9 — газ при снятой охране (плановая проверка
   баллоном, раздел 59) — только мера оценки газа: в рабочей разметке выключена.
4. Проникновение — дверь или люк открылись, когда объект на охране, за ними сработало движение или
   объёмный датчик (ответ 20), и охрану в коллекторе не сняли в течение ARRIVAL. Если сняли — это
   персонал, который открыл дверь раньше, чем снял охрану: в 2022–2026 таких 24% срабатываний под охраной.
5. След выезда — «охрана снята → дверь → движение» в том же коллекторе (объект 2-го уровня:
   подсистемы одного коллектора висят на разных объектах 3-го уровня). Эпизод подтверждён, если
   выезд начался в пределах CONFIRM после начала эпизода. Окно выбрано по приросту доли выездов
   над фоном (results/labels.md).

    python labels.py
"""
import config

EPISODE_GAP = '1 hour'
MASS_SMOKE = 10              # Н8: столько извещателей одного объекта за ±10 мин — массовое срабатывание
POWER_BEFORE_FLOOD = 60      # с, Н7: «Затоплен» так скоро после события питания в коллекторе — побочный эффект
VISIT_STEP = '30 minutes'    # след выезда: дверь и движение не позже чем через столько после снятия охраны
INTRUSION_STEP = '10 minutes'  # проникновение: движение не позже чем через столько после открытия
ARRIVAL = '15 minutes'       # охрану сняли так скоро после открытия — это персонал, а не нарушитель
CONFIRM = '2 hours'          # окно сопоставления эпизода и выезда
PRIMARY = '7 days'           # эпизод первичный, если такого же типа на объекте не было столько времени
CHECK_TYPES = ""             # Н9 выключена; "('gas')" — разметка для оценки газа (work_ck), раздел 59

EQUIPMENT = "('Состояние насоса', 'Состояние вентилятора', 'Состояние фазы', 'ИБП', 'Переключатель')"
OPENINGS = "('КД Дверь', 'КД Люк', 'Стекло', '9-секционный люк')"
TRIGGERS = f"""
    CASE
        WHEN stype = 'Датчик дыма' AND state = 'Обнаружен дым' THEN 'fire'
        WHEN stype = 'Тепловой датчик' AND state = 'Не замкнут' THEN 'fire'
        WHEN stype = 'Ручной извещатель' AND state IN ('Не замкнут', 'Рычаг сдернут') THEN 'fire'
        WHEN stype = 'Датчик температуры' AND (state = 'Температура выше 40ºC' OR num BETWEEN 40 AND 80) THEN 'fire'
        WHEN stype = 'Газовый датчик' AND state = 'Обнаружен газ' THEN 'gas'
        WHEN stype = 'Состояние насоса' AND state IN ('Затоплен', 'Работают все насосы в АНС') THEN 'flood'
        WHEN stype = 'Датчик затопления' AND state = 'Не замкнут' THEN 'flood'
        -- «Обесточен» у фазы штатно чередуется с «Есть питание» (по 415 тыс. строк), это не отказ
        WHEN stype = 'Состояние фазы' AND state IN ('Неисправен', 'Отключено устройство') THEN 'equipment'
        -- «Батарея неисправна» не тревожное ни в справочнике состояний заказчика, ни во флаге журнала (0%)
        WHEN stype IN {EQUIPMENT} AND stype <> 'Состояние фазы' AND state IN ('Неисправен', 'Обесточен',
             'Отключено устройство', 'Питание от батарей', 'Батарея разряжена') THEN 'equipment'
        WHEN stype NOT IN {EQUIPMENT} AND stype <> 'Состояние охраны'
             AND state IN ('Неисправен', 'Отключено устройство', 'Не определено') THEN 'sensor'
        -- Н5: заглушки вместо показаний
        WHEN stype = 'Датчик температуры' AND (num < -40 OR num > 80) THEN 'sensor'
        -- газ — % объёма метана, тревога с 1% (ответ заказчика); выше 15% и ниже нуля — неисправность
        WHEN stype = 'Газовый датчик' AND (num < 0 OR num > 15) THEN 'sensor'
    END"""


def build(con) -> None:
    """Таблицы разметки из ev и obj текущего соединения; retro.py подставляет журнал, обрезанный по моменту."""
    con.sql("""CREATE OR REPLACE TABLE obj3 AS
               SELECT object_id, parent_id AS collector_id, kind, name FROM obj WHERE level = 3""")
    # Постановка и снятие в одну секунду (28 раз за все годы) — мигание: во всех 17 случаях с известными
    # соседями чередование сходится, только если состояние после пары то же, что до неё. Пару выкидываем,
    # иначе ASOF JOIN и arg_max выбирают между её строками как придётся, и разметка не воспроизводится
    con.sql("""
        CREATE OR REPLACE TABLE guard AS
        SELECT g.object_id, o.collector_id, g.ts, g.state = 'На охране' AS armed
        FROM ev g JOIN obj3 o USING (object_id)
        WHERE g.stype = 'Состояние охраны' AND g.state IN ('На охране', 'Снято с охраны')
        QUALIFY min(armed::INT) OVER (PARTITION BY g.object_id, g.ts) = max(armed::INT) OVER (PARTITION BY g.object_id, g.ts)""")

    # 1. Триггеры по состояниям
    con.sql(f"""
        CREATE OR REPLACE TABLE trig AS
        SELECT object_id, channel_id, ts, stype, coalesce(state, num::VARCHAR) AS what, {TRIGGERS} AS type
        FROM ev WHERE type IS NOT NULL""")

    # 2. Проникновение. ASOF JOIN берёт ближайшее событие до (охрана) или после (движение, снятие охраны)
    con.sql(f"""
        CREATE OR REPLACE TABLE armed_open AS
        WITH openings AS (
            SELECT e.object_id, o.collector_id, e.channel_id, e.ts, e.stype
            FROM ev e JOIN obj3 o USING (object_id) WHERE e.stype IN {OPENINGS} AND e.state = 'Не замкнут'
        ), motion AS (
            SELECT object_id, ts FROM ev
            WHERE (stype = 'Датчик движения' AND state = 'Обнаружено движение') OR (stype = 'КД АВ' AND state = 'Не замкнут')
        ), armed AS (
            SELECT p.* FROM openings p ASOF JOIN guard g ON g.object_id = p.object_id AND p.ts >= g.ts WHERE g.armed
        ), moved AS (
            SELECT a.* FROM armed a ASOF JOIN motion m ON m.object_id = a.object_id AND a.ts < m.ts
            WHERE m.ts <= a.ts + INTERVAL {INTRUSION_STEP}
        )
        SELECT v.*, d.ts - v.ts <= INTERVAL {ARRIVAL} AS arrival
        FROM moved v ASOF LEFT JOIN (SELECT * FROM guard WHERE NOT armed) d
             ON d.collector_id = v.collector_id AND v.ts <= d.ts""")
    con.sql("""INSERT INTO trig SELECT object_id, channel_id, ts, stype, 'Не замкнут под охраной', 'intrusion'
               FROM armed_open WHERE arrival IS NOT TRUE""")

    # 3. Шум: массовый дым (Н8) и «Затоплен» сразу после события питания в том же коллекторе (Н7)
    con.sql(f"""
        CREATE OR REPLACE TABLE trig_noise AS
        SELECT object_id, channel_id, ts, type, 'Н8' AS noise FROM (
            SELECT *, count(DISTINCT channel_id) OVER (
                PARTITION BY object_id ORDER BY ts
                RANGE BETWEEN INTERVAL 10 MINUTE PRECEDING AND INTERVAL 10 MINUTE FOLLOWING) AS n
            FROM trig WHERE stype = 'Датчик дыма' AND type = 'fire')
        WHERE n >= {MASS_SMOKE}""")
    con.sql(f"""
        INSERT INTO trig_noise
        WITH f AS (SELECT t.*, o.collector_id FROM trig t JOIN obj3 o USING (object_id) WHERE t.what = 'Затоплен'),
             p AS (SELECT t.ts, o.collector_id FROM trig t JOIN obj3 o USING (object_id) WHERE t.type = 'equipment')
        SELECT f.object_id, f.channel_id, f.ts, f.type, 'Н7'
        FROM f ASOF JOIN p ON p.collector_id = f.collector_id AND f.ts >= p.ts
        WHERE f.ts - p.ts <= INTERVAL {POWER_BEFORE_FLOOD} SECOND""")

    # 4. След выезда: снятие охраны → дверь → движение в том же коллекторе
    con.sql(f"""
        CREATE OR REPLACE TABLE visit AS
        WITH d AS (SELECT ts, collector_id FROM guard WHERE NOT armed),
             x AS (SELECT e.ts, o.collector_id FROM ev e JOIN obj3 o USING (object_id)
                   WHERE e.stype IN ('КД Дверь', 'КД Люк') AND e.state = 'Не замкнут'),
             m AS (SELECT e.ts, o.collector_id FROM ev e JOIN obj3 o USING (object_id)
                   WHERE e.stype = 'Датчик движения' AND e.state = 'Обнаружено движение'),
             dx AS (SELECT d.*, x.ts AS door FROM d ASOF JOIN x ON x.collector_id = d.collector_id AND d.ts < x.ts)
        SELECT dx.collector_id, dx.ts AS t0
        FROM dx ASOF JOIN m ON m.collector_id = dx.collector_id AND dx.ts < m.ts
        WHERE dx.door <= dx.ts + INTERVAL {VISIT_STEP} AND m.ts <= dx.ts + INTERVAL {VISIT_STEP}""")

    # 5. Эпизоды: объект × тип, пауза ≤ EPISODE_GAP. Из шумовых триггеров собираются свои эпизоды —
    #    чтобы показать масштаб шума
    con.sql(f"""
        CREATE OR REPLACE TABLE inc AS
        WITH t AS (
            SELECT tr.object_id, tr.ts, tr.type, tr.channel_id, n.noise
            FROM trig tr LEFT JOIN trig_noise n USING (object_id, channel_id, ts, type)
        ), marked AS (
            SELECT *, CASE WHEN ts - lag(ts) OVER w > INTERVAL {EPISODE_GAP} OR lag(ts) OVER w IS NULL
                           THEN 1 ELSE 0 END AS new_ep
            FROM t WINDOW w AS (PARTITION BY object_id, type, noise IS NOT NULL ORDER BY ts)
        ), numbered AS (
            SELECT *, sum(new_ep) OVER (PARTITION BY object_id, type, noise IS NOT NULL ORDER BY ts) AS ep
            FROM marked
        ), eps AS (
            SELECT object_id, type, min(ts) AS t0, max(ts) AS t1, count(*) AS rows,
                   count(DISTINCT channel_id) AS channels, any_value(noise) AS noise
            FROM numbered GROUP BY object_id, type, noise IS NOT NULL, ep
        ), prev AS (
            SELECT *, t0 - lag(t1) OVER (PARTITION BY object_id, type, noise IS NOT NULL ORDER BY t0) AS since_prev
            FROM eps
        )
        SELECT p.object_id, o.collector_id, p.type, p.t0, p.t1, p.rows, p.channels, p.noise,
               coalesce(v.t0 <= p.t0 + INTERVAL {CONFIRM}, false) AS confirmed,
               coalesce(p.since_prev > INTERVAL {PRIMARY}, true) AS primary_
        FROM prev p JOIN obj3 o USING (object_id)
        ASOF LEFT JOIN visit v ON v.collector_id = o.collector_id AND p.t0 < v.t0""")

    # 6. Н9 — ВРЕМЕННО, до таблицы графика плановых работ (раздел 59, INTEGRATION.md §1.6): эпизод
    #    типа из CHECK_TYPES, начавшийся при снятой охране коллектора, считается проверкой. Меняется
    #    только цель: признаки видят такие эпизоды как раньше (features.py). Модель газа, обученная
    #    без проверок, хуже, поэтому в рабочей разметке блок выключен и служит только мерой. Придёт
    #    таблица — этот блок заменить окнами из неё.
    if CHECK_TYPES:
        con.sql(f"""
            UPDATE inc SET noise = 'Н9'
            FROM (SELECT i.object_id, i.type, i.t0
                  FROM inc i ASOF JOIN guard g ON g.collector_id = i.collector_id AND i.t0 >= g.ts
                  WHERE i.type IN {CHECK_TYPES} AND i.noise IS NULL AND NOT g.armed) c
            WHERE inc.object_id = c.object_id AND inc.type = c.type AND inc.t0 = c.t0 AND inc.noise IS NULL""")


def main() -> None:
    con = config.connect()
    build(con)
    print(con.sql("""SELECT type, year(t0) AS год, count(*) FILTER (WHERE noise IS NULL) AS эпизодов,
                            count(*) FILTER (WHERE noise IS NOT NULL) AS шумовых,
                            round(avg(confirmed::INT) FILTER (WHERE noise IS NULL), 3) AS подтв,
                            round(avg(primary_::INT) FILTER (WHERE noise IS NULL), 3) AS первичных
                     FROM inc GROUP BY ALL ORDER BY 1, 2""").show(max_rows=100))
    print(con.sql("""SELECT count(*) AS срабатываний_под_охраной, count(*) FILTER (WHERE arrival) AS персонал
                     FROM armed_open WHERE year(ts) >= 2022"""))
    print(con.sql("SELECT year(t0) AS год, count(*) AS выездов FROM visit GROUP BY 1 ORDER BY 1"))


if __name__ == '__main__':
    main()
