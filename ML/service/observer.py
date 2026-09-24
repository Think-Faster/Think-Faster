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
        """history: тип → массив оценок окна. Отдаёт сводку и копит в buf."""
        per = {}
        for tp in config.TYPES:
            h = history.get(tp, np.empty(0))
            exp_share = (alarms.get(tp, 0) / max(len(alarms.get(tp, []) or []), 1))
            per[tp] = {'window_size': int(h.size), 'alarm_share': float(exp_share),
                       'psi_vs_week': -1}
            if h.size:
                last_week = h[-24:]
                per[tp]['psi_vs_week'] = psi(h[: max(h.size - 168, 1)], last_week)[0]
        summary = {'hour': now.isoformat(), 'freshness_hours': freshness,
                   'stale': stale, 'dlq': dlq, 'types': per}
        self.buf.append(summary)
        return summary

    def dump(self):
        import json
        self.path.write_text(json.dumps(self.buf[-200:], ensure_ascii=False, indent=1),
                             encoding='utf-8')