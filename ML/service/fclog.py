"""Журнал прогнозов на томе сервиса — источник `/history` (INTEGRATION §13.1, история главного диспетчера §9.2).

Каждый такт пишет оценку всех объектов по шести типам, порог часа и статус тех пар, что не просто
«ниже порога»: ALARM (ушло диспетчеру), MUTED (молчание по графику работ или решению), REJECTED
(отклонено диспетчером и погашено правилом). Так главный диспетчер видит, что модель говорила и
почему это не дошло до смены. Глубина — как у горячего журнала (HOT_RETENTION_DAYS).

Отдельный файл DuckDB, а не горячая база: такт пишет сюда раз в час, ручки читают своим курсором
и не ждут тяжёлых запросов сборки признаков.
"""
import threading
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

import svc as config

STATUSES = ('ALARM', 'MUTED', 'REJECTED')


class ForecastLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.OUT_DIR / 'forecasts.duckdb')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self._lock = threading.Lock()
        cols = ', '.join(f'{tp} REAL' for tp in config.TYPES)
        self.con.sql(f'CREATE TABLE IF NOT EXISTS fc_score (hour_end TIMESTAMP, object_id BIGINT, {cols})')
        self.con.sql("""CREATE TABLE IF NOT EXISTS fc_status (hour_end TIMESTAMP, object_id BIGINT, type VARCHAR,
                        status VARCHAR, reason VARCHAR, ref VARCHAR)""")
        self.con.sql("""CREATE TABLE IF NOT EXISTS fc_thr (hour_end TIMESTAMP, type VARCHAR, threshold REAL,
                        model_version VARCHAR, settings_version INTEGER)""")

    def write(self, hour_end: datetime, objects: np.ndarray, scores: dict, thresholds: dict,
              statuses: list[tuple], model_version: str, settings_version: int) -> None:
        """statuses — [(object_id, type, status, reason, ref)]. Час переписывается целиком (повтор такта)."""
        score = pl.DataFrame({'hour_end': [hour_end] * len(objects), 'object_id': np.asarray(objects, np.int64),
                              **{tp: np.asarray(scores[tp], np.float32) for tp in config.TYPES}})
        st = pl.DataFrame(statuses, schema={'object_id': pl.Int64, 'type': pl.Utf8, 'status': pl.Utf8,
                                            'reason': pl.Utf8, 'ref': pl.Utf8}, orient='row') \
               .with_columns(pl.lit(hour_end).alias('hour_end'))
        thr = pl.DataFrame({'hour_end': [hour_end] * len(config.TYPES), 'type': list(config.TYPES),
                            'threshold': [float(thresholds[tp]) for tp in config.TYPES],
                            'model_version': [model_version] * len(config.TYPES),
                            'settings_version': [int(settings_version)] * len(config.TYPES)})
        with self._lock:
            self.con.execute('BEGIN')
            for t in ('fc_score', 'fc_status', 'fc_thr'):
                self.con.execute(f'DELETE FROM {t} WHERE hour_end = ?', [hour_end])
            self.con.sql(f'INSERT INTO fc_score SELECT hour_end, object_id, {", ".join(config.TYPES)} FROM score')
            self.con.sql('INSERT INTO fc_status SELECT hour_end, object_id, type, status, reason, ref FROM st')
            self.con.sql('INSERT INTO fc_thr SELECT hour_end, type, threshold, model_version, settings_version FROM thr')
            self.con.execute('COMMIT')

    def history(self, object_id: int, tp: str, a: datetime, b: datetime) -> list[dict]:
        """Часы [a, b) пары: оценка, порог часа, статус (None — ниже порога), причина и ссылка."""
        assert tp in config.TYPES
        cur = self.con.cursor()
        rows = cur.execute(f"""
            SELECT s.hour_end, s.{tp}, t.threshold, t.model_version, x.status, x.reason, x.ref
            FROM fc_score s JOIN fc_thr t ON t.hour_end = s.hour_end AND t.type = ?
            LEFT JOIN fc_status x ON x.hour_end = s.hour_end AND x.object_id = s.object_id AND x.type = ?
            WHERE s.object_id = ? AND s.hour_end >= ? AND s.hour_end < ? ORDER BY s.hour_end""",
                           [tp, tp, int(object_id), a, b]).fetchall()
        cur.close()
        return [{'hour_end': h.isoformat(timespec='minutes') + config.TZ, 'score': round(float(p), 4),
                 'threshold': round(float(thr), 4), 'model_version': mv, 'status': st, 'reason': rs, 'ref': ref}
                for h, p, thr, mv, st, rs, ref in rows]

    def last_hour(self) -> datetime | None:
        cur = self.con.cursor()
        r = cur.execute('SELECT max(hour_end) FROM fc_thr').fetchone()[0]
        cur.close()
        return r

    def sweep(self, now: datetime, days: int = config.HOT_RETENTION_DAYS) -> None:
        edge = now - timedelta(days=days)
        with self._lock:
            for t in ('fc_score', 'fc_status', 'fc_thr'):
                self.con.execute(f'DELETE FROM {t} WHERE hour_end < ?', [edge])

    def close(self) -> None:
        self.con.close()
