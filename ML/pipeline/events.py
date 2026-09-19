"""Шаг 1. Журнал → таблица ev: события 2019–2026 без 2021 года, привязанные к объектам.

- Дубли (Н6) схлопываем: совпадают канал, время и значение. Флаг «тревожное» не берём —
  он меняет смысл по годам (анализ.md), поэтому и в сравнении дублей не участвует.
- Строку заголовка внутри файла 2025 года выкидываем. У канала «Состояние охраны» выкидываем
  значения-даты: нулевую `01.01.1970` (Н9), замаскированную `##.##.####` и обычные даты.
- Каналы, которых нет в справочнике, выкидываем: их не к чему привязать.
- Числовые показания (газ, температура, заряд ИБП) — в num, текстовые состояния — в state.

    python events.py
"""
import time

import config


def main() -> None:
    con = config.connect()
    t = time.time()
    con.sql(f"""
        CREATE OR REPLACE TABLE obj AS
        SELECT ид_объект AS object_id, иерархия_уровень AS level, родитель AS parent_id,
               вид_объекта AS kind, диспетчерское_название_объекта AS name
        FROM read_csv('{(config.DICT / 'справочник_объектов_диспетчер.csv').as_posix()}', header=true)""")
    con.sql(f"""
        CREATE OR REPLACE TABLE ch AS
        SELECT ид_канала_данных::INTEGER AS channel_id, тип_инж_системы AS system, тип_датчика AS stype,
               тег_инженерной_системы AS tag, название_датчика AS name, ид_объект AS object_id
        FROM read_csv('{(config.DICT / 'справочник_каналов_датчиков.csv').as_posix()}', header=true,
                      types={{'тег_инженерной_системы': 'VARCHAR'}})""")
    files = [(config.JOURNAL / f'ext-journal-{y}.csv').as_posix() for y in config.YEARS]
    con.sql(f"""
        CREATE OR REPLACE TABLE ev AS
        WITH raw AS (
            SELECT try_cast(ид_канала_данных AS INTEGER) AS channel_id,
                   try_cast(дата || ' ' || время AS TIMESTAMP) AS ts,
                   значение_датчика AS v
            FROM read_csv({files}, header=true, quote='"', escape='"', parallel=true,
                          columns={{'ид_события': 'VARCHAR', 'ид_канала_данных': 'VARCHAR', 'дата': 'VARCHAR',
                                   'время': 'VARCHAR', 'тревожное': 'VARCHAR', 'значение_датчика': 'VARCHAR'}})
        ), uniq AS (
            SELECT DISTINCT channel_id, ts, v FROM raw
            WHERE ts >= '2019-01-01' AND ts < '{config.DATA_END}' AND year(ts) <> 2021
        )
        SELECT c.object_id::SMALLINT AS object_id, u.channel_id, u.ts, c.stype,
               CASE WHEN try_cast(u.v AS DOUBLE) IS NULL THEN u.v END AS state,
               try_cast(u.v AS DOUBLE) AS num
        FROM uniq u JOIN ch c USING (channel_id)
        WHERE NOT (c.stype = 'Состояние охраны' AND regexp_matches(u.v, '^[0-9#]{{2}}\\.[0-9#]{{2}}\\.'))""")
    print('ev', con.sql('SELECT count(*) FROM ev').fetchone()[0], 'строк за', round(time.time() - t), 'с')
    print(con.sql("""SELECT year(ts) AS год, count(*) AS строк, count(DISTINCT channel_id) AS каналов,
                            count(DISTINCT object_id) AS объектов FROM ev GROUP BY 1 ORDER BY 1"""))


if __name__ == '__main__':
    main()
