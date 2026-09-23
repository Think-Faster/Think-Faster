"""Состояние правил (M6): склейка дребезга, отклонение диспетчера, молчание.

Три правила из INTEGRATION §2.2, и у каждого есть состояние «между часами»:

- склейка повторов 6 ч: сколько часов тревога пары объект-тип уже держится (since_hours, §2.3);
- отклонение диспетчера: для газа и подтопления REJECT гасит оценку в reject_k раз, пока не
  придёт настоящий эпизод (факт) или REOPEN; для остальных типов — только в историю (§9.1);
- MUTE: временное молчание пары на срок (админ-панель).

Состояние — только в памяти (один процесс пишет); при рестарте склейка начинается заново, это
дешевле и честнее, чем поднимать её из истории 90 суток.
"""
import json
from pathlib import Path

import numpy as np

import svc as config
from settings import OperatingSettings


class RuleState:
    def __init__(self, settings: OperatingSettings, bootstrap: Path | None = None):
        self.settings = settings
        self.last_alarm: dict[tuple[int, str], int] = {}
        self.rejections: dict[tuple[int, str], int] = {}   # (объект,тип) -> час решения
        self.mutes: dict[tuple[int, str], int] = {}        # (объект,тип) -> час окончания
        self.log: list[dict] = []
        if bootstrap and bootstrap.exists():
            self.log = json.loads(bootstrap.read_text(encoding='utf-8'))

    def apply(self, scores: dict, objects: np.ndarray, hour_end: int,
              thresholds: dict) -> dict:
        """Применить правила к сырым оценкам. Возврат: тип → (alarm, since_hours) массивы.

        scores/thresholds — по типу. События REJECT/MUTE приходят через on_decision до вызова.
        """
        gap = self.settings.chatter_gap_hours
        k = self.settings.reject_k
        out = {}
        for tp, s in scores.items():
            thr = thresholds[tp]
            s = np.asarray(s, np.float32)
            weight = np.ones(len(s), np.float32)
            for i, oid in enumerate(objects):
                key = (int(oid), tp)
                if key in self.rejections and hour_end >= self.rejections[key] \
                        and tp in self.settings.reject_types:
                    weight[i] = 1.0 - k                   # §2.2: газ/подтопление, k = 0.2
            raw = s * weight >= thr
            alarm = raw.copy()
            since = np.ones(len(alarm), np.int32)
            for i, oid in enumerate(objects):
                key = (int(oid), tp)
                if raw[i]:
                    last = self.last_alarm.get(key)
                    since[i] = 1 if last is None else int(hour_end - last)
                until = self.mutes.get(key)
                if until is not None and hour_end < until:
                    alarm[i] = False
                if alarm[i]:
                    self.last_alarm[key] = hour_end
            out[tp] = (alarm, since)
        return out

    def on_decision(self, object_id: int, tp: str, action: str, ts: int,
                    mute_hours: int | None = None) -> None:
        key = (object_id, tp)
        self.log.append({'object_id': object_id, 'type': tp, 'action': action, 'h': ts})
        if action == 'REJECT':
            # для не-режектируемых типов отклонение остаётся только записью в историю (§9.1)
            self.rejections[key] = ts
        elif action == 'MUTE':
            if mute_hours is None:
                mute_hours = self.settings.mute_max_hours
            self.mutes[key] = ts + mute_hours
        elif action in ('REOPEN', 'CONFIRMED'):
            self.rejections.pop(key, None)
            if action == 'REOPEN':
                self.mutes.pop(key, None)

    def on_fact(self, facts: set[tuple[int, str]], hour_end: int) -> None:
        """«До следующего настоящего эпизода»: эпизод пришёл — отклонение снято."""
        for key in list(self.rejections):
            if key in facts:
                self.rejections.pop(key, None)