"""Состояние правил (M6): склейка дребезга, отклонение диспетчера, молчание (П5/П6).

Три правила из INTEGRATION §2.2 с состоянием «между часами», но по интерфейсам §10.1:

- склейка повторов — ходячий `chatter.runs_with_gap()`: разрыв ≤ 6 ч не начинает новый сигнал.
  `since_hours` (П6) — часы С КОНЦА последнего эпизода пары (0, пока сигнал жив), а не с последней
  тревоги, чтобы карточка показывала, сколько прошло от события, а не дребезг;
- REJECT (П5) — оценка гасится, пока пара не поднимется выше не-аварийного порога:
  `thr_mute = квантиль(90-суточного парка, 1 - share·k)` (k>0), как в reject.py: k=0 — полное
  молчание до конца эпизода/REOPEN, k=None — правило выключено и действие идёт только в историю;
  в режиме N молчание дополнительно ограничено REJECT_N_HOURS;
- MUTE — временное молчание по паре (админ-панель); снимается REOPEN или настоящим эпизодом.

Настоящий эпизод (факт из labels.build: `inc`) снимает и отклонение, и молчание — «до следующего
настоящего эпизода», как в reject.simulate. Состояние — память одного процесса-писателя; при
рестарте склейка стартует заново, это честнее, чем поднимать её из истории 90 суток.
"""
import json
from pathlib import Path

import numpy as np

import svc as config
from settings import OperatingSettings


class RuleState:
    """Часовое состояние: сигнал пары, отклонение, молчание, стакан решений."""

    def __init__(self, settings: OperatingSettings, bootstrap: Path | None = None,
                 history: dict | None = None):
        self.settings = settings
        self.history = history or {}              # тип -> (h_hist, p_hist) для thr_mute
        self.signal: dict[tuple[int, str], dict] = {}   # (объект,тип) -> {start, last_on, open}
        self.rejections: dict[tuple[int, str], int] = {}  # (объект,тип) -> час решения (N-режим)
        self.mutes: dict[tuple[int, str], int] = {}       # (объект,тип) -> час окончания MUTE
        self.suppressed: dict[tuple[int, str], int] = {}  # час последнего подавления (для журнала)
        self.log: list[dict] = []
        if bootstrap and bootstrap.exists():
            self.log = json.loads(bootstrap.read_text(encoding='utf-8'))

    def _thr_mute(self, tp: str) -> float | None:
        k = self.settings.reject_k(tp)
        if k is None or k == 0.0:
            # не-режектируемый тип или k=0 («навсегда»): полное молчание, пока не снято эпизодом
            return 1.0 if k == 0.0 else None
        if tp not in self.history:
            return None
        hh, pp = self.history[tp]
        lo, hi = np.searchsorted(hh, [hh[-1] - 90 * 24 + 1, hh[-1] + 1])
        return float(np.quantile(pp[lo:hi], 1 - self.settings.share(tp) * k))

    def apply(self, scores: dict, objects: np.ndarray, hour_end: int,
              thresholds: dict) -> dict:
        """Применить правила к сырым оценкам. Возврат: тип → (alarm, since_hours).

        scores/thresholds/history — по типу; действия REJECT/MUTE приходят в on_decision.
        """
        gap = config.CHATTER_GAP_HOURS
        out = {}
        for tp, s in scores.items():
            s = np.asarray(s, np.float32)
            thr = float(thresholds[tp])
            raw = s >= thr                       # сырая тревога retro (доля) не меняется
            mute_thr = self._thr_mute(tp)
            alarm = raw.copy()
            since = np.zeros(len(alarm), np.int32)
            for i, oid in enumerate(objects):
                key = (int(oid), tp)
                now = raw[i]
                st = self.signal.get(key)
                if now:                          # дребезг: тот же сигнал, когда пауза ≤ gap
                    if st and hour_end - st['last_on'] <= gap + 1:
                        st['last_on'] = hour_end
                    else:
                        self.signal[key] = st = {'start': hour_end, 'last_on': hour_end}
                    since[i] = 0                 # П6: эпизод ещё не закончился
                elif st:
                    since[i] = int(hour_end - st['last_on'])   # часы С КОНЦА эпизода (П6)
                if key in self.rejections and mute_thr is not None:
                    # П5: пока пара не поднялась выше не-аварийного порога — тревога гасится
                    alarm[i] = False
                    if mute_thr < 1.0 and s[i] >= mute_thr:
                        alarm[i] = True          # сигнал вернулся выше thr_mute — оголяем
                    self.suppressed[key] = hour_end
                until = self.mutes.get(key)
                if until is not None and hour_end < until:
                    alarm[i] = False
                    self.suppressed[key] = hour_end
            if self.settings.is_rejectable(tp):
                for key in list(self.rejections):
                    if hour_end - self.rejections[key] > config.REJECT_N_HOURS:
                        self.rejections.pop(key)      # N-режим: молчание ограничено N (§10.1)
            out[tp] = (alarm, since)
        return out

    def on_decision(self, object_id: int, tp: str, action: str, ts: int,
                    mute_hours: int | None = None) -> None:
        key = (object_id, tp)
        self.log.append({'object_id': object_id, 'type': tp, 'action': action, 'h': ts})
        if action == 'REJECT':
            if self.settings.is_rejectable(tp):
                self.rejections[key] = ts
            # для не-режектируемых типов отклонение — только запись в историю (§9.1)
        elif action == 'MUTE':
            self.mutes[key] = ts + (mute_hours if mute_hours is not None else config.REJECT_N_HOURS)
        elif action in ('REOPEN', 'CONFIRMED'):
            self.rejections.pop(key, None)
            if action == 'REOPEN':
                self.mutes.pop(key, None)

    def on_fact(self, facts: set[tuple[int, str]], hour_end: int) -> None:
        """«Настоящий эпизод пришёл» — снимает и отклонение, и молчание (reject.simulate)."""
        for key in list(self.rejections):
            if key in facts:
                self.rejections.pop(key, None)
        for key in list(self.mutes):
            if key in facts:
                self.mutes.pop(key, None)

    def reasons(self, object_id: int, tp: str) -> list[str]:
        """Почему сейчас отклонение/молчание — текст для карточки и журнала решений."""
        key = (object_id, tp)
        if key in self.rejections:
            return ['REJECT']
        if key in self.mutes:
            return ['MUTE']
        return []