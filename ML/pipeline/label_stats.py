"""Числа для обоснования разметки (Ф3-2): окно подтверждения выездом, шум Н7/Н8, персонал под охраной,
и 20 подтверждённых эпизодов для ручного просмотра. Печатает markdown.

Фон для выездов — те же коллекторы и то же время суток через неделю до и после эпизода: доля
«выезд в окне W» у эпизодов против той же доли у сдвинутых моментов. Прирост над фоном и есть
сигнал, что бригада ехала на этот эпизод.

    python label_stats.py > label_stats.md
"""
import config

WINDOWS = ['30 minutes', '1 hour', '2 hours', '4 hours', '8 hours', '24 hours']


def md(rel, floats: int = 3) -> str:
    cols, rows = rel.columns, rel.fetchall()
    fmt = lambda v: f'{v:.{floats}f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join(fmt(v) for v in r) + ' |' for r in rows])


def main() -> None:
    con = config.connect(read_only=True)
    con.sql("""
        CREATE TEMP TABLE d AS
        WITH e AS (SELECT type, collector_id, t0 FROM inc WHERE noise IS NULL AND year(t0) BETWEEN 2022 AND 2025),
             pts AS (SELECT type, collector_id, t0 AS t, 'эпизод' AS kind FROM e
                     UNION ALL SELECT type, collector_id, t0 + INTERVAL 7 DAY, 'фон' FROM e
                     UNION ALL SELECT type, collector_id, t0 - INTERVAL 7 DAY, 'фон' FROM e)
        SELECT p.type, p.kind, v.t0 - p.t AS gap
        FROM pts p ASOF LEFT JOIN visit v ON v.collector_id = p.collector_id AND p.t < v.t0""")
    share = ', '.join(f"round(avg((gap <= INTERVAL '{w}')::INT) FILTER (WHERE kind = 'эпизод'), 3) AS \"эп {w}\", "
                      f"round(avg((gap <= INTERVAL '{w}')::INT) FILTER (WHERE kind = 'эпизод') / "
                      f"nullif(avg((gap <= INTERVAL '{w}')::INT) FILTER (WHERE kind = 'фон'), 0), 2) AS \"×фон {w}\""
                      for w in WINDOWS)
    print('## Выезд после начала эпизода, 2022–2025\n')
    print('Доля эпизодов, после которых в коллекторе начался выезд в пределах окна, и прирост над фоном.\n')
    print(md(con.sql(f"SELECT type AS тип, count(*) FILTER (WHERE kind = 'эпизод') AS эпизодов, {share} "
                     "FROM d GROUP BY 1 ORDER BY 1")))

    print('\n## Шум Н7: «Затоплен» рядом с событием питания в коллекторе\n')
    print(md(con.sql("""
        WITH f AS (SELECT t.ts, o.collector_id FROM trig t JOIN obj3 o USING (object_id) WHERE t.what = 'Затоплен'),
             p AS (SELECT t.ts, o.collector_id FROM trig t JOIN obj3 o USING (object_id) WHERE t.type = 'equipment'),
             j AS (SELECT year(f.ts) AS y, f.ts - p.ts AS gap FROM f ASOF LEFT JOIN p
                   ON p.collector_id = f.collector_id AND f.ts >= p.ts)
        SELECT y AS год, count(*) AS «Затоплен», round(avg((gap = INTERVAL 0 SECOND)::INT), 3) AS в_ту_же_секунду,
               round(avg((gap <= INTERVAL 60 SECOND)::INT), 3) AS за_60_с
        FROM j GROUP BY 1 ORDER BY 1""")))

    print('\n## Шум Н8: массовый дым (≥ 10 извещателей объекта за ±10 мин)\n')
    print(md(con.sql("""
        SELECT year(t0) AS год, count(*) FILTER (WHERE type = 'fire' AND noise IS NULL) AS пожар_эпизодов,
               count(*) FILTER (WHERE noise = 'Н8') AS массовый_дым,
               count(*) FILTER (WHERE noise = 'Н7') AS затоплен_от_питания
        FROM inc GROUP BY 1 ORDER BY 1""")))

    print('\n## Открытия под охраной с движением\n')
    print(md(con.sql("""
        SELECT year(ts) AS год, count(*) AS срабатываний, count(*) FILTER (WHERE arrival) AS персонал_снял_охрану,
               round(avg(arrival::INT), 3) AS доля_персонала
        FROM armed_open GROUP BY 1 ORDER BY 1""")))

    print('\n## 20 подтверждённых эпизодов для просмотра\n')
    print(md(con.sql("""
        WITH c AS (SELECT * FROM inc WHERE confirmed AND noise IS NULL AND year(t0) BETWEEN 2022 AND 2025
                   QUALIFY row_number() OVER (PARTITION BY type ORDER BY hash(object_id, t0))
                           <= CASE WHEN type IN ('fire', 'equipment') THEN 4 ELSE 3 END),
             first_trig AS (
                SELECT c.object_id, c.type, c.t0, arg_min(t.stype || ': ' || t.what, t.ts) AS первый_триггер
                FROM c JOIN trig t ON t.object_id = c.object_id AND t.type = c.type AND t.ts = c.t0 GROUP BY ALL)
        SELECT c.type AS тип, c.object_id AS объект, o.name AS название, c.t0 AS начало,
               round(epoch(c.t1 - c.t0) / 60)::INT AS мин, c.rows AS строк, c.channels AS каналов,
               f.первый_триггер, round(epoch(v.t0 - c.t0) / 60)::INT AS выезд_через_мин
        FROM c JOIN obj3 o USING (object_id) JOIN first_trig f USING (object_id, type, t0)
        ASOF JOIN visit v ON v.collector_id = c.collector_id AND c.t0 < v.t0
        ORDER BY c.type, c.t0""")))


if __name__ == '__main__':
    main()
