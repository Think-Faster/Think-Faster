"""Скользящий порог (M5): квантиль оценок парка за 90 суток на долю из настроек.

Порог не свойство модели (INTEGRATION §2.1, §9.4): при смене версии истории оценок принадлежат
старой модели, и порог пересчитывается заново — пока истории нет, берётся из ретропрогона новой
версии (таблица `rebase`), это часть M9.

История хранится в OUT_DIR/history.parquet по часам (тип × объект × score); окно — последние
threshold_window_days суток. estimate() — та же квантиль, но в числа «тревог в сутки» для
ползунков админ-панели (§9.3, ручка /api/ml/estimate).
"""
from datetime import datetime, timedelta

import numpy as np
import polars as pl

import svc as config
from settings import OperatingSettings, estimate


class ScoreHistory:
    """Оценки парка по часам; append каждую границу часа, окно срезается в момент чтения."""

    def __init__(self, settings: OperatingSettings, path=None):
        self.settings = settings
        self.path = path or config.HISTORY
        self._by_type: dict[str, list[tuple[int, np.ndarray]]] = {t: [] for t in config.TYPES}
        self.threshold_override: dict[str, float] = {}   # из ретропрогона новой версии (M9)
        self.load()

    def load(self):
        if not self.path.exists():
            return
        df = pl.read_parquet(self.path)
        for tp in config.TYPES:
            sub = df.filter(pl.col('type') == tp).sort('h')
            if sub.height:
                self._by_type[tp] = [(int(h), arr) for h, arr in zip(
                    sub['h'].to_list(), row_chunks(sub['scores'].to_list()))]

    def update(self, hour_end: int, scores: dict) -> None:
        for tp, arr in scores.items():
            self._by_type[tp].append((hour_end, np.asarray(arr, np.float32)))
        self._trim()

    def _trim(self):
        win = self.settings.threshold_window_days * 24
        for tp in self._by_type:
            arrs = self._by_type[tp]
            if not arrs:
                continue
            last = arrs[-1][0]
            self._by_type[tp] = [(h, a) for h, a in arrs if h > last - win]

    def window(self, tp: str) -> np.ndarray:
        return np.concatenate([a for _, a in self._by_type[tp]]) if self._by_type[tp] else np.empty(0)

    def threshold(self, tp: str, share: float | None = None) -> float:
        s = self.settings.share(tp) if share is None else share
        w = self.window(tp)
        if w.size == 0:
            return float('nan')
        return float(np.quantile(w, 1.0 - s))

    def estimate(self, tp: str, share: float) -> dict:
        return estimate(self.window(tp), share, self.settings.threshold_window_days)

    def rebase(self, tp: str, thr: float, days: int = 7) -> None:
        """Пока истории новой версии нет, держим порог из ретропрогона (M9/§9.4).

        Ставим одно «доисторическое» значение-заглушку, которое просто не влияет на окно при
        первых 7 сутках сбора — на порог оно не заменяется, для этого есть поле threshold_override.
        """
        w = self.window(tp)
        if w.size == 0:
            self._by_type[tp].append((0, np.array([thr], np.float32)))

    def save(self) -> None:
        rows = []
        for tp, arrs in self._by_type.items():
            for h, a in arrs:
                rows.append(pl.DataFrame({'type': [tp], 'h': [h], 'scores': [serialize(a)]}))
        if not rows:
            return
        pl.concat(rows).write_parquet(self.path)


def serialize(a: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


def row_chunks(b64s: list[str]) -> list[np.ndarray]:
    import base64
    return [np.frombuffer(base64.b64decode(b), np.float32) for b in b64s]


def hour_index(t: datetime) -> int:
    import features as ft
    return int((t - ft.T0).total_seconds() // 3600)