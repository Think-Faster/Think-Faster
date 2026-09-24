"""Горячий журнал (M2): куда приём пишет события, из чего собираются признаки.

Схема совпадает с внутренней таблицей `ev` исследования (INTEGRATION §1.1), но называется
`ev_all` — ровно то имя, которое ждёт `retro.snapshot()` из main, пересоздающий вид `ev` с окном
и полной историей охраны (П2, INTEGRATION2 §10.1). events.py чистит журнал на этапе загрузки —
здесь та же чистка применяется к каждой строке Kafka-потока. Разница одна: у исследования журнал
статичный, у нас он кольцевой — глубина 100 суток, строки охраны не удаляются никогда (для
проникновения нужна вся история режима).

Событие входит в чтения только после чистки; чистка должна совпадать с events.py строка в строку,
иначе витрина в проде разойдётся с той, на которой модели считаны. Дедупликация — первичным ключом
(канал, время, значение), как SELECT DISTINCT в events.py; флаг «тревожное» в дублях не участвует.
"""
import re
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

import svc as config

GUARD_DATE = re.compile(r'^[0-9#]{2}\.[0-9#]{2}\.')

# Каналы вне справочника выкидываются (events.py): их не к чему привязать. Числовые показания — в
# num, текстовые состояния — в state, сырое значение оставлено в value как ключ дедупликации.
SCHEMA = """
CREATE TABLE IF NOT EXISTS ev_all(
    object_id INT, channel_id INT, ts TIMESTAMP, stype VARCHAR,
    state VARCHAR, num DOUBLE, value VARCHAR, PRIMARY KEY(channel_id, ts, value));
CREATE INDEX IF NOT EXISTS ev_all_ts ON ev_all(ts);
CREATE TABLE IF NOT EXISTS obj(
    object_id INT PRIMARY KEY, level INT, parent_id INT, kind VARCHAR, name VARCHAR);
CREATE TABLE IF NOT EXISTS ch(
    channel_id INT PRIMARY KEY, system VARCHAR, stype VARCHAR, tag VARCHAR, name VARCHAR, object_id INT);
"""


def _load_csv(con: duckdb.DuckDBPyConnection, obj_csv: Path, ch_csv: Path) -> None:
    con.sql(f"""CREATE OR REPLACE TABLE obj AS
            SELECT ид_объект AS object_id, иерархия_уровень AS level, родитель AS parent_id,
                   вид_объекта AS kind, диспетчерское_название_объекта AS name
            FROM read_csv('{(obj_csv).as_posix()}', header=true)""")
    con.sql(f"""CREATE OR REPLACE TABLE ch AS
            SELECT ид_канала_данных::INTEGER AS channel_id, тип_инж_системы AS system,
                   тип_датчика AS stype, тег_инженерной_системы AS tag, название_датчика AS name,
                   ид_объект AS object_id
            FROM read_csv('{(ch_csv).as_posix()}', header=true,
                          types={{'тег_инженерной_системы': 'VARCHAR'}})""")


class HotStore:
    """Кольцевой журнал сервиса на DuckDB. Один процесс пишет, N читают."""

    def __init__(self, path: Path | None = None, read_only: bool = False):
        self.path = Path(path) if path else config.HOT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        self.con.sql(f"SET memory_limit='{os_mem()}'")
        if not read_only:
            for stmt in SCHEMA.split(';'):
                if stmt.strip():
                    self.con.execute(stmt)
        self._ch = None

    def close(self) -> None:
        self.con.close()

    # ----- справочники ---------------------------------------------------
    def load_reference(self, obj_csv: Path, ch_csv: Path) -> None:
        _load_csv(self.con, obj_csv, ch_csv)
        self._ch = None

    def import_reference(self) -> None:
        _load_csv(self.con, config.DICT / 'справочник_объектов_диспетчер.csv',
                  config.DICT / 'справочник_каналов_датчиков.csv')

    def reference_df(self) -> pl.DataFrame:
        return self.con.sql('SELECT * FROM ch').pl()

    # ----- чистка (совпадает с pipeline/events.py) ---------------------------------
    def clean_event(self, channel_id: int, ts, value) -> dict | None:
        ch = self._channels()
        row = ch.get(channel_id)
        if row is None:                                  # каналов нет в справочнике — не к чему привязать
            return None
        value = str(value)
        if row['stype'] == config.GUARD_STYPE and GUARD_DATE.match(value):
            return None                                 # «01.01.1970», «##.##.####», обычные даты
        num = None
        try:
            num = float(value)
        except ValueError:
            pass
        return {'object_id': row['object_id'], 'channel_id': channel_id, 'ts': ts,
                'stype': row['stype'], 'state': None if num is not None else value,
                'num': num, 'value': value}

    def _channels(self) -> dict:
        if self._ch is None:
            rows = self.con.sql('SELECT channel_id, stype, object_id FROM ch').fetchall()
            self._ch = {r[0]: {'stype': r[1], 'object_id': r[2]} for r in rows}
        return self._ch

    # ----- запись ----------------------------------------------------------
    def append(self, rows: list[dict]) -> int:
        """INSERT OR IGNORE: дубль «канал + время + значение» схлопывается (Н6)."""
        if not rows:
            return 0
        self.con.executemany(
            'INSERT OR IGNORE INTO ev_all VALUES (?,?,?,?,?,?,?)',
            [(r['object_id'], r['channel_id'], r['ts'], r['stype'], r['state'], r['num'], r['value'])
             for r in rows])
        return len(rows)

    RAW_COLS = "{'ид_события': 'VARCHAR', 'ид_канала_данных': 'VARCHAR', 'дата': 'VARCHAR', " \
               "'время': 'VARCHAR', 'тревожное': 'VARCHAR', 'значение_датчика': 'VARCHAR'}"

    def bulk_import(self, until: datetime | None = None, since: datetime | None = None,
                    years: list[int] | None = None) -> int:
        """Загрузка журнала из csv одним COPY-подобным INSERT (П9, INTEGRATION2 §10.1).

        Дата — параметр, не константа: `until` (по умолчанию config.DATA_END) режет хвост, `since`
        — левый край. Годы вне YEARS и 2021 выкидываются, как в events.py. Чистка и дедупликация —
        внутри того же запроса: разбор CSV — через read_csv поверхности (семейство COPY в DuckDB,
        кавычки и провалы не ломают строку), массовая вставка объявлена одним INSERT OR IGNORE.
        """
        until = until or config.DATA_END
        since = since or datetime(2019, 1, 1)
        years = years or config.YEARS
        files = [(config.JOURNAL / f'ext-journal-{y}.csv').as_posix() for y in years]
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f'нет журнала: {missing}')
        before = int(self.con.sql('SELECT count(*) FROM ev_all').fetchone()[0])
        self.con.sql(f"""
            INSERT OR IGNORE INTO ev_all
            WITH raw AS (
                SELECT try_cast(ид_канала_данных AS INTEGER) AS channel_id,
                       try_cast(дата || ' ' || время AS TIMESTAMP) AS ts, значение_датчика AS v
                FROM read_csv({files}, header=true, quote='"', escape='"', parallel=true,
                              columns={self.RAW_COLS})
                WHERE ts >= TIMESTAMP '{since}' AND ts < TIMESTAMP '{until}' AND year(ts) <> 2021)
            SELECT c.object_id::INT, r.channel_id, r.ts, c.stype,
                   CASE WHEN try_cast(r.v AS DOUBLE) IS NULL THEN r.v END, try_cast(r.v AS DOUBLE), r.v
            FROM raw r JOIN ch c USING (channel_id)
            WHERE NOT (c.stype = '{config.GUARD_STYPE}' AND regexp_matches(r.v, '^[0-9#]{{2}}\\.[0-9#]{{2}}\\.'))
            """)
        return int(self.con.sql('SELECT count(*) FROM ev_all').fetchone()[0]) - before

    # ----- чтение (образец — retro.open_db) ----------------------------------------
    def ev_view(self, t: datetime) -> None:
        """Представление ev на момент t: только журнал до t, окно 100 сут + охрана целиком."""
        lo = t - timedelta(days=config.HOT_RETENTION_DAYS)
        self.con.sql(f"""CREATE OR REPLACE VIEW ev AS
            SELECT object_id, channel_id, ts, stype, state, num FROM ev_all
            WHERE ts < TIMESTAMP '{t}' AND (ts >= TIMESTAMP '{lo}' OR stype = '{config.GUARD_STYPE}')""")

    def query(self, sql: str) -> pl.DataFrame:
        return self.con.sql(sql).pl()

    def objects_3(self) -> pl.DataFrame:
        return self.con.sql("SELECT * FROM obj WHERE level = 3").pl()

    # ----- обслуживание -----------------------------------------------------------
    def retention_sweep(self, now: datetime) -> int:
        """Удалить строки старше глубины, кроме охраны — это M2 в работе."""
        cut = now - timedelta(days=config.HOT_RETENTION_DAYS)
        return self.con.execute(
            f"""DELETE FROM ev_all WHERE ts < TIMESTAMP '{cut}' AND stype <> '{config.GUARD_STYPE}'"""
        ).fetchone()[0]


def os_mem() -> str:
    import os
    return os.environ.get('TF_MEMORY', '2GB')