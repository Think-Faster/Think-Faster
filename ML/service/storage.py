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
from datetime import datetime, timedelta, timezone
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
CREATE TABLE IF NOT EXISTS ch_status(
    channel_id INT PRIMARY KEY, status VARCHAR, since TIMESTAMP, changed_at TIMESTAMP);
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


def _msk(s: str) -> datetime:
    """Время из сообщения (ISO, возможно с поясом) → наивное московское, как в журнале."""
    t = datetime.fromisoformat(s)
    if t.tzinfo is not None:
        t = t.astimezone(timezone(timedelta(hours=3))).replace(tzinfo=None)
    return t


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

    def apply_reference(self, d: dict, con=None) -> str:
        """Сообщение `tf.ingest.reference` (Н26). Возвращает, что применено; непонятное — ValueError.

        - `channel.status` от воронки: канал замолчал (`silent`, с `since`) или заговорил (`ok`);
        - `object` и `channel` — строка справочника с колонками датасета (`ид_объект`, `родитель`, …,
          `ид_канала_данных`, `тип_датчика`, …): объект или канал добавлен или изменён на ходу.
        """
        con = con or self.con
        kind = d.get('kind')
        if kind == 'channel.status':
            status = d['status']
            if status not in ('silent', 'ok'):
                raise ValueError(f'статус канала {status!r}')
            con.execute('INSERT OR REPLACE INTO ch_status VALUES (?,?,?,?)',
                        (int(d['ид_канала_данных']), status, _msk(d.get('since') or d['at']), _msk(d['at'])))
        elif kind == 'object':
            # справочники из csv пересоздаются CREATE … AS SELECT, без первичного ключа: замена — удалить и вставить
            con.execute('DELETE FROM obj WHERE object_id = ?', (int(d['ид_объект']),))
            con.execute('INSERT INTO obj VALUES (?,?,?,?,?)',
                        (int(d['ид_объект']), int(d['иерархия_уровень']),
                         None if d.get('родитель') in (None, '') else int(d['родитель']),
                         d.get('вид_объекта'), d.get('диспетчерское_название_объекта')))
        elif kind == 'channel':
            con.execute('DELETE FROM ch WHERE channel_id = ?', (int(d['ид_канала_данных']),))
            con.execute('INSERT INTO ch VALUES (?,?,?,?,?,?)',
                        (int(d['ид_канала_данных']), d.get('тип_инж_системы'), d['тип_датчика'],
                         None if d.get('тег_инженерной_системы') is None else str(d['тег_инженерной_системы']),
                         d.get('название_датчика'), int(d['ид_объект'])))
            self._ch = None                              # новый канал сразу проходит чистку
        else:
            raise ValueError(f'неизвестный вид справочника {kind!r}')
        return kind

    def silent_channels(self) -> pl.DataFrame:
        """Молчащие каналы по воронке: object_id, channel_id, stype, since."""
        return self.con.sql("""SELECT c.object_id, s.channel_id, c.stype, s.since FROM ch_status s
                               JOIN ch c USING (channel_id) WHERE s.status = 'silent'""").pl()

    def last_events(self, objects: list[int], t, hours: int = 24) -> pl.DataFrame:
        """Последнее событие каждого канала объектов `objects` за `hours` ч до t (свидетели тревоги, §9.7)."""
        if not objects:
            return pl.DataFrame(schema={'object_id': pl.Int32, 'channel_id': pl.Int32, 'ts': pl.Datetime,
                                        'stype': pl.Utf8, 'value': pl.Utf8})
        ids = ','.join(str(int(o)) for o in objects)
        return self.con.sql(f"""
            SELECT object_id, channel_id, ts, stype, value FROM ev_all
            WHERE ts < TIMESTAMP '{t}' AND ts >= TIMESTAMP '{t}' - INTERVAL {int(hours)} HOUR
              AND object_id IN ({ids}) AND stype <> '{config.GUARD_STYPE}'
            QUALIFY row_number() OVER (PARTITION BY channel_id ORDER BY ts DESC) = 1""").pl()

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
    def append(self, rows: list[dict], con=None) -> int:
        """INSERT OR IGNORE: дубль «канал + время + значение» схлопывается (Н6). `con` — курсор потока
        приёма (core._IngestStore): такт в это время читает основным соединением."""
        if not rows:
            return 0
        (con or self.con).executemany(
            'INSERT OR IGNORE INTO ev_all VALUES (?,?,?,?,?,?,?)',
            [(r['object_id'], r['channel_id'], r['ts'], r['stype'], r['state'], r['num'], r['value'])
             for r in rows])
        return len(rows)

    RAW_COLS = "{'ид_события': 'VARCHAR', 'ид_канала_данных': 'VARCHAR', 'дата': 'VARCHAR', " \
               "'время': 'VARCHAR', 'тревожное': 'VARCHAR', 'значение_датчика': 'VARCHAR'}"

    def bulk_import(self, until: datetime | None = None, since: datetime | None = None,
                    years: list[int] | None = None, chunk_days: int = 7) -> int:
        """Загрузка журнала из csv (П9, INTEGRATION2 §10.1): окно горячего журнала плюс вся охрана.

        Дата — параметр, не константа: `until` (по умолчанию config.DATA_END) режет хвост, `since`
        (по умолчанию `until` минус глубина M2, 100 сут) — левый край; строки охраны берутся за всю
        историю, как их хранит `retention_sweep`. Годы вне YEARS и 2021 выкидываются, как в events.py.
        Разбор CSV — через read_csv поверхности (семейство COPY в DuckDB, кавычки и провалы не ломают
        строку), чистка — в том же запросе, во временную таблицу без индексов. В ev_all строки идут
        порциями по `chunk_days` суток через INSERT OR IGNORE: первичный ключ (Н6) растёт в памяти, и
        одна вставка 100 суток парка (~16 млн строк) не укладывается в TF_MEMORY=2GB, а по неделе — да.
        """
        until = until or config.DATA_END
        since = since or until - timedelta(days=config.HOT_RETENTION_DAYS)
        years = years or config.YEARS
        files = [(config.JOURNAL / f'ext-journal-{y}.csv').as_posix() for y in years]
        missing = [f for f in files if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(f'нет журнала: {missing}')
        before = int(self.con.sql('SELECT count(*) FROM ev_all').fetchone()[0])
        self.con.sql(f"""
            CREATE OR REPLACE TEMP TABLE ev_stage AS
            WITH raw AS (
                SELECT try_cast(ид_канала_данных AS INTEGER) AS channel_id,
                       try_cast(дата || ' ' || время AS TIMESTAMP) AS ts, значение_датчика AS v
                FROM read_csv({files}, header=true, quote='"', escape='"', parallel=true,
                              columns={self.RAW_COLS})
                WHERE ts < TIMESTAMP '{until}' AND year(ts) <> 2021)
            SELECT c.object_id::INT AS object_id, r.channel_id, r.ts, c.stype,
                   CASE WHEN try_cast(r.v AS DOUBLE) IS NULL THEN r.v END AS state,
                   try_cast(r.v AS DOUBLE) AS num, r.v AS value
            FROM raw r JOIN ch c USING (channel_id)
            WHERE (r.ts >= TIMESTAMP '{since}' OR c.stype = '{config.GUARD_STYPE}')
              AND NOT (c.stype = '{config.GUARD_STYPE}' AND regexp_matches(r.v, '^[0-9#]{{2}}\\.[0-9#]{{2}}\\.'))
            ORDER BY r.ts""")
        cols = 'object_id, channel_id, ts, stype, state, num, value'
        self.con.execute(f"INSERT OR IGNORE INTO ev_all SELECT {cols} FROM ev_stage WHERE ts < TIMESTAMP '{since}'")
        a = since
        while a < until:
            b = min(a + timedelta(days=chunk_days), until)
            self.con.execute(f"""INSERT OR IGNORE INTO ev_all SELECT {cols} FROM ev_stage
                                 WHERE ts >= TIMESTAMP '{a}' AND ts < TIMESTAMP '{b}'""")
            a = b
        self.con.execute('DROP TABLE ev_stage')
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