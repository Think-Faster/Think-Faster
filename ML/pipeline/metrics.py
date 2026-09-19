"""Метрики прогноза инцидентов (ТЗ §6, Ф5-1).

Два взгляда на одну и ту же разметку:
- по объекто-часам: предупреждение в час h верно, если эпизод начался в часах h+1 … h+H;
  Precision здесь — доля предупреждений, за которыми пришёл инцидент;
- по эпизодам: эпизод пойман, если в какой-то из H часов перед его началом было предупреждение.
  Упреждение — сколько часов от первого такого предупреждения до начала эпизода (не больше H).
  Отдельно — сколько часов тревога горела непрерывно к началу эпизода (до 7 сут): длинная серия
  значит, что объект давно в зоне риска, а не что модель заметила что-то новое.
"""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

RUN_CAP = 168


def onsets(obj: np.ndarray, h: np.ndarray, nxt: np.ndarray, cap: int) -> set[tuple[int, int]]:
    """Эпизоды (объект, час начала), которые видны из строк витрины."""
    m = nxt < cap
    return set(zip(obj[m].tolist(), (h[m] + nxt[m]).tolist()))


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Порог с наибольшим F1 — выбирается на проверке (2025) и без изменений идёт на тест."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    return float(th[np.argmax(f1)]) if len(th) else 0.5


def threshold_for_precision(y: np.ndarray, p: np.ndarray, target: float,
                            min_alarms: int = 50) -> float | None:
    """Порог под заданную точность: наибольшая полнота среди точек, где Precision не ниже target.

    Ложная тревога дороже пропуска по-разному для разных типов, поэтому рабочую точку выбирает
    заказчик, а не F1. Точки, где тревог меньше min_alarms, отбрасываются: там точность случайна.
    """
    pr, rc, th = precision_recall_curve(y, p)
    pos = max(int(y.sum()), 1)
    alarms = rc[:-1] * pos / np.maximum(pr[:-1], 1e-12)
    ok = np.where((pr[:-1] >= target) & (alarms >= min_alarms))[0]
    return float(th[ok[0]]) if len(ok) else None


def threshold_for_rate(p: np.ndarray, per_day: float, rows: int, days: float) -> float:
    """Порог под бюджет диспетчера: сколько тревог в сутки он готов разбирать по всем объектам."""
    rate = min(per_day * days / max(rows, 1), 1.0)
    return float(np.quantile(p, 1 - rate))


def signals(obj: np.ndarray, h: np.ndarray, y: np.ndarray, alarm: np.ndarray) -> tuple[int, int]:
    """Сигналы диспетчеру: подряд идущие часы тревоги на объекте — это один сигнал, а не десять.

    Возвращает (всего сигналов, из них подтвердившихся). Сигнал подтверждён, если хотя бы в один из
    его часов инцидент действительно начался в горизонте. Остальные — ложные: то, что диспетчер
    сходил и ничего не нашёл.
    """
    if not alarm.any():
        return 0, 0
    o, hh, yy = obj[alarm], h[alarm], y[alarm]
    order = np.lexsort((hh, o))
    o, hh, yy = o[order], hh[order], yy[order]
    start = np.empty(len(o), bool)
    start[0] = True
    start[1:] = (o[1:] != o[:-1]) | (hh[1:] != hh[:-1] + 1)
    run = np.cumsum(start) - 1
    total = int(run[-1]) + 1
    true = int(np.bincount(run, weights=yy, minlength=total).astype(bool).sum())
    return total, true


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def evaluate(obj: np.ndarray, h: np.ndarray, nxt: np.ndarray, p: np.ndarray, thr: float,
             horizon: int, cap: int) -> dict:
    y = (nxt <= horizon).astype(np.int8)
    alarm = p >= thr
    out = {'rows': int(len(y)), 'base_rate': float(y.mean()),
           'pr_auc': float(average_precision_score(y, p)) if y.any() else float('nan'),
           'roc_auc': float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float('nan'),
           'threshold': float(thr), 'alarm_rate': float(alarm.mean()),
           'precision': float(y[alarm].mean()) if alarm.any() else float('nan'),
           'recall_rows': float(alarm[y == 1].mean()) if y.any() else float('nan')}
    # эпизоды и упреждение
    alarmed = set(zip(obj[alarm].tolist(), h[alarm].tolist()))
    eps = onsets(obj, h, nxt, cap)
    seen = set(zip(obj.tolist(), h.tolist()))
    leads, runs, caught, total = [], [], 0, 0
    for o, e in eps:
        window = [(o, e - k) for k in range(horizon, 0, -1)]   # от самого раннего часа к позднему
        if not any(w in seen for w in window):
            continue
        total += 1
        first = next((k for k, w in zip(range(horizon, 0, -1), window) if w in alarmed), None)
        if first is not None:
            caught += 1
            leads.append(first)
            # сколько часов тревога уже горела к началу эпизода: непрерывная серия назад, до 7 сут
            k = first
            while k < RUN_CAP and (o, e - k - 1) in alarmed:
                k += 1
            runs.append(k)
    leads, runs = np.array(leads), np.array(runs)
    sig, sig_true = signals(obj, h, y, alarm)
    out.update({'signals': sig, 'signals_true': sig_true, 'signals_false': sig - sig_true})
    q = lambda a, x: float(np.percentile(a, x)) if len(a) else float('nan')
    out.update({'episodes': total, 'caught': caught, 'recall_episodes': caught / total if total else float('nan'),
                'lead_median_h': q(leads, 50), 'lead_p25_h': q(leads, 25), 'lead_p75_h': q(leads, 75),
                'lead_ge_12h': float((leads >= 12).mean()) if len(leads) else float('nan'),
                'alarm_run_median_h': q(runs, 50), 'alarm_run_capped': float((runs >= RUN_CAP).mean())
                if len(runs) else float('nan')})
    return out
