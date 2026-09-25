"""Шаг 2. Разметка: триггеры → эпизоды инцидентов по типам, след выезда бригады.

Подтверждений инцидентов в данных нет (ответ 12), поэтому разметка косвенная:
1. Триггер — строка журнала, которая по смыслу состояния относится к одному из типов (TRIGGERS).
2. Эпизод — триггеры одного типа на одном объекте с паузой не больше EPISODE_GAP. Так дребезг (Н3)
   и пачки по многим каналам схлопываются в один инцидент; начало эпизода — момент инцидента.
3. Шум размечается, а не удаляется: массовый дым (Н8) и «Затоплен» в пачке с событием питания (Н7).
   В цель модели шумные эпизоды не входят. Метка Н10 — газ и отказ газового датчика в окне
   планово-предупредительных работ по графику организатора (settings/works_2026.csv, раздел 62).
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
WORKS = config.ML / 'settings' / 'works_2026.csv'  # таблица графика работ (INTEGRATION.md §1.6), раздел 62
WORKS_PAD = 7                # сут: в год без строки графика окно переносится на те же дни года ± столько

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

    # 6. Н10 — плановые работы по графику (раздел 62, INTEGRATION.md §1.6): эпизод типа из строки
    #    графика, начавшийся на её объекте в окне работ (works_windows), — шум. Сюда же отказ датчика,
    #    начавшийся с датчика, который работы снимают (`removed_sensor`; у ППР газа — газовый, его
    #    увозят в метрологию). Как прочий шум, эпизод Н10 не входит ни в цель, ни в счётчики эпизодов
    #    в признаках (features.py); срабатывания его датчиков в признаках остаются.
    works_windows(con)
    con.sql("""
        UPDATE inc SET noise = 'Н10'
        FROM (SELECT DISTINCT i.object_id, i.type, i.t0
              FROM inc i JOIN works_win w
                ON w.object_id IN (i.object_id, i.collector_id) AND i.t0 >= w.a AND i.t0 < w.b
              WHERE i.noise IS NULL
                AND (list_contains(w.types, i.type) OR i.type = 'sensor' AND EXISTS (
                     SELECT 1 FROM trig t WHERE t.object_id = i.object_id AND t.ts = i.t0
                                          AND t.type = 'sensor' AND t.stype = w.sensor))) c
        WHERE inc.object_id = c.object_id AND inc.type = c.type AND inc.t0 = c.t0 AND inc.noise IS NULL""")


def works_windows(con, years: list[int] | None = None) -> None:
    """Окна графика работ — временная таблица `works_win` (object_id, types, sensor, a, b), час входит
    в окно при a <= ts < b. Строка графика — объект любого уровня (коллектор накрывает все свои
    объекты), вид работ, типы происшествий и время начала и конца с точностью до часа; это та же
    таблица, которую правит главный диспетчер (INTEGRATION.md §1.6). В свой год окно берётся как есть.
    В год, где у объекта нет строки того же вида работ, окно переносится со строки ближайшего года на
    те же дни с запасом WORKS_PAD: по журналу график из года в год почти не сдвигается (раздел 62).
    Одно правило на разметку (Н10) и на молчание M7 (maintenance.py --mode works); годы по умолчанию —
    все, где есть эпизоды."""
    if years is None:
        years = [r[0] for r in con.sql('SELECT DISTINCT year(t0) FROM inc ORDER BY 1').fetchall()]
    con.sql(f"""
        CREATE OR REPLACE TEMP TABLE works_win AS
        WITH w AS (SELECT object_id::BIGINT AS object_id, work_kind, string_split(incident_types, ',') AS types,
                          removed_sensor AS sensor, starts_at::TIMESTAMP AS s, ends_at::TIMESTAMP AS e,
                          year(starts_at::TIMESTAMP) AS wy
                   FROM read_csv('{WORKS.as_posix()}', delim=';', header=true, all_varchar=true)
                   WHERE object_id IS NOT NULL),
             src AS (SELECT k.object_id, k.work_kind, y, arg_min(k.wy, abs(k.wy - y)) AS wy
                     FROM (SELECT DISTINCT object_id, work_kind, wy FROM w) k, (SELECT unnest({list(years)}) AS y)
                     GROUP BY ALL HAVING NOT bool_or(k.wy = y))
        SELECT object_id, types, sensor, s AS a, e AS b FROM w
        UNION ALL
        SELECT w.object_id, w.types, w.sensor,
               w.s + to_years((c.y - w.wy)::INT) - INTERVAL {WORKS_PAD} DAY,
               w.e + to_years((c.y - w.wy)::INT) + INTERVAL {WORKS_PAD} DAY
        FROM w JOIN src c USING (object_id, work_kind, wy)""")


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
