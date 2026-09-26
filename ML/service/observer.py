"""Наблюдение (M10): дрейф, поток тревог против ожидаемой доли, свежесть, DLQ.

Собирает показатели на границе часа в OUT_DIR/observe.json. Это не отчёт для человека, а вход
для алертов: состояние ухудшилось — есть flag, по которому дежурный смотрит подробности. PSI —
как в pipeline/drift.py (децили эталона против нового окна); границы 0.10 / 0.25.
"""
import json
from pathlib import Path

import numpy as np

import svc as config


def psi(ref: np.ndarray, win: np.ndarray, bins: int = 10,
        grades: tuple = (0.10, 0.25)) -> tuple[float, int]:
    """Population Stability Index распределения win против ref: индекс и грейд 0/1/2."""
    if ref.size < bins or win.size == 0:
        return float('nan'), 0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    if edges.size < 2:
        return float('nan'), 0
    r = np.clip(np.histogram(ref, bins=edges)[0] / max(ref.size, 1), 1e-6, None)
    w_ = np.clip(np.histogram(win, bins=edges)[0] / max(win.size, 1), 1e-6, None)
    val = float(np.sum((r - w_) * np.log(r / w_)))
    grade = 0 if val < grades[0] else 1 if val < grades[1] else 2
    return val, grade


class Observer:
    def __init__(self, path: Path | None = None):
        self.path = path or config.OBS_LOG
        self.buf: list[dict] = []

    def observe(self, *, alarms: dict, history: dict, freshness: dict,
                stale: bool, dlq: int, now) -> dict:
        """history: тип → (часы, оценки парка) истории порога. PSI — оценки парка за последние сутки
        против истории старше недели (строки истории — объект-часы, а не часы). Сводка копится в buf."""
        per = {}
        for tp in config.TYPES:
            hh, pp = history.get(tp, (np.empty(0), np.empty(0)))
            a = np.asarray(alarms.get(tp, []), float)
            per[tp] = {'window_size': int(pp.size), 'alarm_share': float(a.mean()) if a.size else 0.0,
                       'psi_day_vs_history': None, 'psi_grade': None}
            if pp.size:
                ref, win = pp[hh <= hh[-1] - 168], pp[hh > hh[-1] - 24]
                val, grade = psi(ref[::max(1, ref.size // 200_000)], win)
                per[tp].update({'psi_day_vs_history': None if np.isnan(val) else round(val, 4),
                                'psi_grade': grade})
        summary = {'hour': now.isoformat(), 'freshness_hours': freshness,
                   'stale': stale, 'dlq': dlq, 'types': per}
        self.buf = self.buf[-199:] + [summary]
        return summary

    def dump(self):
        import json
        self.path.write_text(json.dumps(self.buf[-200:], ensure_ascii=False, indent=1),
                             encoding='utf-8')