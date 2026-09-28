"""Рабочие настройки: ML/settings/operating.json (INTEGRATION §2.1, §9.3).

Формат — ровно тот, что в main: `version/changed/changed_by/reason` и `types{tp:{share, reject_k}}`
с per-типовым reject_k (None — правило отклонения выключено). Валидация по operating.schema.json —
та же `config.operating()`, что читает исследование, — сервис не держит своей копии и не расширяет
файл своими полями (П5/П6, INTEGRATION2 §10.1). Интерфейс диспетчера меняет только долю и k;
пересчёт «доля → тревог в сутки» — функция estimate() (ручка /api/ml/estimate).
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import svc as config
import config as pipe  # noqa: F401  pipeline/config: operating()

KNOWN_TYPES = list(config.TYPES)


class SettingError(ValueError):
    pass


@dataclass
class OperatingSettings:
    version: int
    changed: str
    changed_by: str
    reason: str
    types: dict

    @classmethod
    def load(cls, path: Path | None = None) -> 'OperatingSettings':
        return cls._from(pipe.operating(Path(path) if path else config.SETTINGS))

    @classmethod
    def _from(cls, raw: dict) -> 'OperatingSettings':
        # config.operating() уже проверил схему (границы share и reject_k) и наличие reason
        return cls(version=int(raw['version']), changed=str(raw['changed']),
                   changed_by=str(raw['changed_by']), reason=str(raw['reason']),
                   types={tp: dict(v) for tp, v in raw['types'].items()})

    def share(self, tp: str) -> float:
        return float(self.types[tp]['share'])

    def reject_k(self, tp: str) -> float | None:
        return self.types[tp]['reject_k']

    def is_rejectable(self, tp: str) -> bool:
        return self.types[tp]['reject_k'] is not None

    def check(self, share: float) -> None:
        if not (0.0 < share < 1.0):
            raise SettingError(f'доля {share} вне (0, 1)')

    def rebase(self, updates: dict, by: str, reason: str, now: str | None = None) -> 'OperatingSettings':
        """Новая версия поверх текущих настроек: меняется только types{share, reject_k} (§9.3)."""
        types = json.loads(json.dumps(self.types))
        for tp, u in updates.items():
            if tp in types and isinstance(u, dict):
                for k in ('share', 'reject_k'):
                    if k in u:
                        types[tp][k] = u[k]
        raw = {'version': self.version + 1,
               'changed': now or datetime.now(config.MSK).isoformat(timespec='seconds'),
               'changed_by': by, 'reason': reason, 'types': types}
        _validate(raw)
        return self._from(raw)

    def as_dict(self) -> dict:
        return {'version': self.version, 'changed': self.changed, 'changed_by': self.changed_by,
                'reason': self.reason, 'types': self.types}

    def save(self, path: Path | None = None) -> None:
        p = Path(path) if path else config.SETTINGS
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(p)


def bounds() -> dict:
    """Границы из operating.schema.json для формы админ-панели (/status): доля по типам и reject_k."""
    import json as _json
    schema = _json.loads((pipe.SETTINGS.parent / 'operating.schema.json').read_text(encoding='utf-8'))
    kb = schema['$defs']['reject_k']
    return {'share': {tp: [v['properties']['share']['minimum'], v['properties']['share']['maximum']]
                      for tp, v in schema['properties']['types']['properties'].items()},
            'reject_k': [kb['minimum'], kb['maximum']]}


def _validate(raw: dict) -> None:
    """Проверка по operating.schema.json — та, что в config.operating(), но без записи файла."""
    import json as _json
    schema = _json.loads((pipe.SETTINGS.parent / 'operating.schema.json').read_text(encoding='utf-8'))
    props = schema['properties']['types']['properties']
    kb = schema['$defs']['reject_k']
    assert set(raw['types']) == set(KNOWN_TYPES), 'нужны все 6 типов'
    for tp, v in props.items():
        b, s, k = v['properties']['share'], raw['types'][tp]['share'], raw['types'][tp]['reject_k']
        assert b['minimum'] <= s <= b['maximum'], f'{tp}: share {s} вне [{b["minimum"]}, {b["maximum"]}]'
        assert k is None or kb['minimum'] <= k <= kb['maximum'], f'{tp}: reject_k {k} вне границ'
    assert raw.get('reason'), 'не указана причина изменения'


def estimate(history_scores: np.ndarray, share: float, days: int) -> dict:
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
    for tp, v in st.types.items():
        print(tp, v['share'], v['reject_k'])
    print('version', st.version, '| changed_by', st.changed_by)