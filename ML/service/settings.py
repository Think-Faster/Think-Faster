"""Рабочие настройки: ML/settings/operating.json (INTEGRATION §2.1, §9.3).

Настройки — не константа кода, а файл с версией: интерфейс главного диспетчера меняет его и
обязан оставлять старые версии в истории, сервис перечитывает его без перезапуска. Ползунки в
админ-панели — это ровно поля `shares` и `reject_k`; пересчёт «доля → тревог в сутки» —
функция estimate() (ручка /api/ml/estimate).
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import svc as config

KNOWN_TYPES = list(config.TYPES)


class SettingError(ValueError):
    pass


@dataclass
class OperatingSettings:
    version: int
    changed: str
    changed_by: str
    reason: str
    horizon_hours: int
    shares: dict
    reject_k: float
    reject_types: list
    max_share: float
    threshold_window_days: int
    chatter_gap_hours: int
    mute_max_hours: int

    @classmethod
    def load(cls, path: Path | None = None) -> 'OperatingSettings':
        p = Path(path) if path else config.SETTINGS
        raw = json.loads(p.read_text(encoding='utf-8'))
        return cls._from(raw, p)

    @classmethod
    def _from(cls, raw: dict, p: Path) -> 'OperatingSettings':
        bad = [k for k in ('version', 'changed', 'changed_by', 'reason',
                           'horizon_hours', 'shares', 'reject_k', 'reject_types',
                           'max_share', 'threshold_window_days', 'chatter_gap_hours',
                           'mute_max_hours') if k not in raw]
        if bad:
            raise SettingError(f'{p.name}: нет полей {bad}')
        shares = raw['shares']
        if set(shares) != set(KNOWN_TYPES):
            raise SettingError(f'{p.name}: shares должны покрывать все 6 типов, есть {sorted(shares)}')
        for tp, s in shares.items():
            if s <= 0 or s >= 1:
                raise SettingError(f'{p.name}: доля типа {tp} вне (0, 1): {s}')
            if s > raw['max_share']:
                raise SettingError(f'{p.name}: доля {tp}={s} выше максимума {raw["max_share"]} '
                                   '(§9.3: больше семи тревог в сутки интерфейс не должен давать)')
        if not (0 < raw['reject_k'] <= 1):
            raise SettingError(f'{p.name}: reject_k вне (0, 1]')
        if not set(raw['reject_types']) <= set(KNOWN_TYPES):
            raise SettingError(f'{p.name}: reject_types вне известных типов: {raw["reject_types"]}')
        return cls(**raw)

    def share(self, tp: str) -> float:
        return self.shares[tp]

    def is_rejectable(self, tp: str) -> bool:
        return tp in self.reject_types

    def check(self, share: float) -> None:
        if share > self.max_share:
            raise SettingError(f'доля {share} выше потолка {self.max_share}')

    def rebase(self, updates: dict, by: str, reason: str, now: str | None = None) -> 'OperatingSettings':
        """Новая версия поверх текущих настроек (для записи из интерфейса §9.3)."""
        raw = self.as_dict()
        raw.update({k: v for k, v in updates.items() if k in raw})
        raw['version'] = self.version + 1
        raw['changed'] = now or datetime.now().astimezone().isoformat(timespec='seconds')
        raw['changed_by'] = by
        raw['reason'] = reason
        return self._from(raw, config.SETTINGS)

    def as_dict(self) -> dict:
        return json.loads(json.dumps(self.__dict__))

    def save(self, path: Path | None = None) -> None:
        p = Path(path) if path else config.SETTINGS
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=1), encoding='utf-8')


def estimate(history_scores: np.ndarray, share: float, days: int = 90) -> dict:
    """«Доля часов → сколько тревог в сутки» для ползунка (§9.3) и порога (M5).

    history_scores — плоский массив оценок парка за историю, их место в распределении и есть
    доля: порог = (1 - share)-квантиль; ожидаемых тревог в сутки — столько объекто-часов,
    сколько ляжет выше порога за сутки истории.
    """
    t = np.quantile(history_scores, 1.0 - share)
    above = int((history_scores >= t).sum())
    per_day = above / max(len(history_scores) / 24.0, 1e-9)
    return {'threshold': float(t), 'alarms_per_day': per_day,
            'window_days': round(len(history_scores) / 24.0, 1)}


if __name__ == '__main__':
    st = OperatingSettings.load()
    for tp, s in st.shares.items():
        print(tp, s)
    print('version', st.version, '| rejectable', st.reject_types)