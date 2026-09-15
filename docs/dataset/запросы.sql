-- Запросы к датасету: загрузка и расчёты для анализ.md
-- Диалект DuckDB 1.5. Запуск из папки с распакованными CSV:
--   pip install duckdb
--   duckdb tf.duckdb < запросы.sql        (или по частям из Python: duckdb.connect('tf.duckdb').sql(...))
-- Данные только читаются из CSV, исходные файлы не меняются.

------------------------------------------------------------------------------
-- 1. Загрузка
------------------------------------------------------------------------------

SET preserve_insertion_order = false;

-- Журнал за все годы: ~313,5 млн строк, ~40 с
CREATE OR REPLACE TABLE ev AS
SELECT try_cast(ид_события AS BIGINT)                   AS event_id,
       try_cast(ид_канала_данных AS BIGINT)             AS channel_id,
       try_cast(дата || ' ' || время AS TIMESTAMP)      AS ts,
       тревожное = 't'                                  AS alarm,
       значение_датчика                                 AS value,
       regexp_extract(filename, '(\d{4})')::INT         AS src_year
FROM read_csv('ext-journal-*.csv', header = true, filename = true,
              columns = {'ид_события': 'VARCHAR', 'ид_канала_данных': 'VARCHAR', 'дата': 'VARCHAR',
                         'время': 'VARCHAR', 'тревожное': 'VARCHAR', 'значение_датчика': 'VARCHAR'});

-- Повтор заголовка внутри ext-journal-2025.csv даёт 1 строку с ts IS NULL
DELETE FROM ev WHERE ts IS NULL;

CREATE OR REPLACE TABLE ch AS
SELECT ид_канала_данных AS channel_id, тип_инж_системы AS system_type, тип_датчика AS sensor_type,
       тег_инженерной_системы AS tag, название_датчика AS sensor_name,
       split_part(тег_инженерной_системы, '-', 1) AS pfx
FROM read_csv('справочник_каналов_датчиков.csv', header = true, types = {'тег_инженерной_системы': 'VARCHAR'});

CREATE OR REPLACE TABLE obj AS
SELECT ид_объект AS object_id, иерархия_уровень AS level, родитель AS parent_id,
       вид_объекта AS object_kind, диспетчерское_название_объекта AS object_name
FROM read_csv('справочник_объектов_диспетчер.csv', header = true);

------------------------------------------------------------------------------
-- 2. Объём и деградация
------------------------------------------------------------------------------

-- По годам
SELECT src_year, count(*) AS n_rows, count(DISTINCT channel_id) AS n_channels, sum(alarm::INT) AS n_alarms
FROM ev GROUP BY 1 ORDER BY 1;

-- По месяцам
SELECT strftime(ts, '%Y-%m') AS month, count(*) AS n_rows, count(DISTINCT channel_id) AS n_channels, sum(alarm::INT) AS n_alarms
FROM ev GROUP BY 1 ORDER BY 1;

-- Дни без записей
CREATE OR REPLACE TABLE daily AS
SELECT ts::DATE AS d, count(*) AS n_rows, count(DISTINCT channel_id) AS n_channels FROM ev GROUP BY 1;

SELECT d FROM (SELECT unnest(generate_series(DATE '2019-01-01', DATE '2026-06-30', INTERVAL 1 DAY))::DATE AS d)
ANTI JOIN daily USING (d) ORDER BY d;

-- Первое и последнее появление каналов
WITH l AS (SELECT channel_id, min(ts) AS first_ts, max(ts) AS last_ts FROM ev GROUP BY 1)
SELECT year(first_ts) AS first_year, count(*) AS n_channels FROM l GROUP BY 1 ORDER BY 1;

-- Интервал между записями канала по типам, за месяц
WITH m AS (SELECT e.*, ch.sensor_type FROM ev e JOIN ch USING (channel_id)
           WHERE ts >= '2026-03-01' AND ts < '2026-04-01'),
x AS (SELECT sensor_type, epoch(ts - lag(ts) OVER (PARTITION BY channel_id ORDER BY ts, event_id)) AS dt_s FROM m)
SELECT sensor_type, count(*) AS n, approx_quantile(dt_s, 0.5) AS median_s, approx_quantile(dt_s, 0.9) AS p90_s
FROM x WHERE dt_s IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;

------------------------------------------------------------------------------
-- 3. Целостность
------------------------------------------------------------------------------

-- Каналы журнала, которых нет в справочнике
SELECT count(*) AS n_rows, count(DISTINCT channel_id) AS n_channels, sum(alarm::INT) AS n_alarms, max(ts) AS last_ts
FROM ev ANTI JOIN ch USING (channel_id);

-- Повторы ид_события: в разных файлах / полные дубли. Тяжёлый запрос, ~5 мин
WITH d AS (SELECT event_id, count(*) AS k, count(DISTINCT (channel_id, ts, alarm, value)) AS kv, count(DISTINCT src_year) AS ky
           FROM ev GROUP BY 1 HAVING count(*) > 1)
SELECT count(*) AS repeated_ids, sum(k - 1) AS extra_rows,
       count(*) FILTER (WHERE ky > 1) AS ids_in_several_files, count(*) FILTER (WHERE kv = 1) AS full_duplicate_ids
FROM d;

-- Несколько значений у канала в одну секунду
SELECT count(*) FROM (SELECT channel_id, ts FROM ev WHERE src_year = 2026 GROUP BY 1, 2 HAVING count(DISTINCT value) > 1);

------------------------------------------------------------------------------
-- 4. Состояния, флаг тревоги, числа
------------------------------------------------------------------------------

-- Матрица «тип × состояние → доля тревог»
SELECT ch.sensor_type, value, count(*) AS n, round(100.0 * avg(alarm::INT), 1) AS pct_alarm
FROM ev JOIN ch USING (channel_id)
WHERE try_cast(value AS DOUBLE) IS NULL
GROUP BY 1, 2 HAVING count(*) >= 20 ORDER BY 1, 3 DESC;

-- Флаг тревоги у «Обнаружен дым» по годам
SELECT year(ts) AS y, count(*) AS n, round(100.0 * avg(alarm::INT), 1) AS pct_alarm
FROM ev JOIN ch USING (channel_id) WHERE value = 'Обнаружен дым' GROUP BY 1 ORDER BY 1;

-- Распределение числовых значений и заглушки
SELECT ch.sensor_type, count(*) AS n, min(v), approx_quantile(v, 0.01) AS p01, approx_quantile(v, 0.5) AS median,
       approx_quantile(v, 0.99) AS p99, max(v), count(*) FILTER (WHERE alarm) AS n_alarm
FROM (SELECT channel_id, try_cast(value AS DOUBLE) AS v, alarm FROM ev) e JOIN ch USING (channel_id)
WHERE v IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;

-- Показания газа в окне ±5 мин вокруг «Обнаружен газ»
WITH g AS (SELECT channel_id, ts FROM ev JOIN ch USING (channel_id)
           WHERE sensor_type = 'Газовый датчик' AND value = 'Обнаружен газ' AND alarm AND ts >= '2025-01-01'),
n AS (SELECT channel_id, ts, try_cast(value AS DOUBLE) AS v FROM ev JOIN ch USING (channel_id)
      WHERE sensor_type = 'Газовый датчик' AND ts >= '2025-01-01' AND try_cast(value AS DOUBLE) IS NOT NULL)
SELECT count(DISTINCT (g.channel_id, g.ts)) AS gas_alarms, approx_quantile(n.v, 0.5) AS median_v, max(n.v) AS max_v
FROM g LEFT JOIN n ON n.channel_id = g.channel_id AND n.ts BETWEEN g.ts - INTERVAL 5 MINUTE AND g.ts + INTERVAL 5 MINUTE;

------------------------------------------------------------------------------
-- 5. Эпизоды тревог и шум
------------------------------------------------------------------------------

-- Эпизод: подряд идущие тревоги канала с одним значением и паузой ≤ 1 ч
CREATE OR REPLACE TABLE alarms AS
SELECT e.channel_id, e.ts, e.value, ch.sensor_type, ch.tag, ch.sensor_name, ch.pfx
FROM ev e LEFT JOIN ch USING (channel_id) WHERE alarm;

CREATE OR REPLACE TABLE episodes AS
WITH x AS (SELECT *, CASE WHEN lag(ts) OVER w IS NULL OR epoch(ts - lag(ts) OVER w) > 3600 OR value <> lag(value) OVER w
                          THEN 1 ELSE 0 END AS brk
           FROM alarms WINDOW w AS (PARTITION BY channel_id ORDER BY ts)),
y AS (SELECT *, sum(brk) OVER (PARTITION BY channel_id ORDER BY ts ROWS UNBOUNDED PRECEDING) AS ep FROM x)
SELECT row_number() OVER () AS eid, channel_id, ep, any_value(sensor_type) AS sensor_type, any_value(tag) AS tag,
       any_value(pfx) AS pfx, any_value(sensor_name) AS sensor_name, any_value(value) AS st,
       min(ts) AS t0, max(ts) AS t1, count(*) AS n
FROM y GROUP BY channel_id, ep;

-- Эпизоды по типам и состояниям
SELECT sensor_type, st, count(*) AS episodes, sum(n) AS alarm_rows, round(sum(n) / count(*), 1) AS rows_per_episode,
       count(*) FILTER (WHERE n = 1) AS single_row_episodes
FROM episodes GROUP BY 1, 2 HAVING count(*) >= 30 ORDER BY 3 DESC;

-- Концентрация тревог: доля топ-10 и топ-100 каналов
WITH a AS (SELECT channel_id, count(*) AS n FROM ev WHERE alarm GROUP BY 1),
r AS (SELECT n, row_number() OVER (ORDER BY n DESC) AS rn, sum(n) OVER () AS total FROM a)
SELECT round(100.0 * sum(n) FILTER (WHERE rn <= 10) / max(total), 1) AS top10_pct,
       round(100.0 * sum(n) FILTER (WHERE rn <= 100) / max(total), 1) AS top100_pct
FROM r;

-- Н2. Массовое «Неопределен»: доля в пачках ≥ 20 каналов одного префикса за минуту
WITH x AS (SELECT date_trunc('minute', ts) AS mi, pfx, count(DISTINCT channel_id) AS k, count(*) AS n
           FROM ev JOIN ch USING (channel_id) WHERE value = 'Неопределен' GROUP BY 1, 2)
SELECT round(100.0 * sum(n) FILTER (WHERE k >= 20) / sum(n), 1) AS pct_in_bursts FROM x;

-- Н8. «Обнаружен дым»: сколько соседних эпизодов дыма в том же префиксе за ±10 мин
WITH s AS (SELECT * FROM episodes WHERE st = 'Обнаружен дым' AND sensor_type IS NOT NULL),
k AS (SELECT a.eid, count(b.eid) AS neighbours FROM s a LEFT JOIN s b
        ON a.pfx = b.pfx AND b.eid <> a.eid AND b.t0 BETWEEN a.t0 - INTERVAL 10 MINUTE AND a.t0 + INTERVAL 10 MINUTE
      GROUP BY a.eid)
SELECT CASE WHEN neighbours = 0 THEN '0' WHEN neighbours < 3 THEN '1-2' WHEN neighbours < 10 THEN '3-9' ELSE '10+' END AS b,
       count(*) FROM k GROUP BY 1 ORDER BY 1;

-- Инциденты по годам
SELECT year(t0) AS y,
       count(*) FILTER (WHERE st = 'Затоплен')                        AS flood_pump,
       count(*) FILTER (WHERE sensor_type = 'Датчик затопления')      AS flood_sensor,
       count(*) FILTER (WHERE st = 'Обнаружен дым')                   AS smoke,
       count(*) FILTER (WHERE st = 'Обнаружен газ')                   AS gas,
       count(*) FILTER (WHERE st = 'Температура выше 40ºC')           AS hot,
       count(*) FILTER (WHERE st = 'Питание от батарей')              AS ups_battery,
       count(*) FILTER (WHERE st = 'Обесточен')                       AS deenergized,
       count(*) FILTER (WHERE st = 'Рычаг сдернут')                   AS lever,
       count(*) FILTER (WHERE sensor_type IN ('КД Люк', '9-секционный люк')) AS hatch
FROM episodes GROUP BY 1 ORDER BY 1;

------------------------------------------------------------------------------
-- 6. Параллельные события: всё в префиксе за окно времени
------------------------------------------------------------------------------

-- Случай 1: затопление, 889, 05.05.2024. Для других случаев поменять префикс и окно
SELECT ch.sensor_type, e.value, e.alarm, count(*) AS n, count(DISTINCT e.channel_id) AS n_channels, min(e.ts), max(e.ts)
FROM ev e JOIN ch USING (channel_id)
WHERE ch.pfx = '889' AND e.ts BETWEEN '2024-05-05 03:00' AND '2024-05-05 06:00'
  AND ch.sensor_type <> 'Газовый датчик'
GROUP BY ALL ORDER BY min(e.ts);
-- Случай 2: pfx = '847', 2026-03-05 10:50 — 12:30, дополнительно исключить value IN ('Движения нет', 'Обнаружено движение')
-- Случай 3: pfx = '645', 2025-09-16 14:00 — 16:00
