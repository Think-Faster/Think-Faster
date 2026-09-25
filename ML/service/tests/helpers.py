"""Общие заготовки тестов: operating.json в формате main (types{share,reject_k}, §10.1)."""
import json
from pathlib import Path

from settings import OperatingSettings

TYPES = ('fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion')
DEFAULT_SHARE = {'fire': 0.030, 'gas': 0.026, 'flood': 0.018, 'equipment': 0.072,
                 'sensor': 0.009, 'intrusion': 0.005}
DEFAULT_K = {'fire': None, 'gas': 0.2, 'flood': 0.2, 'equipment': None,
             'sensor': None, 'intrusion': None}


def write_operating(tmp: Path, share: float | None = None,
                    reject_k: dict | None = None) -> Path:
    """Рабочие настройки: все шесть типов с долями в схемных границах."""
    p = tmp / 'operating.json'
    raw = {'version': 1, 'changed': '2026-09-23', 'changed_by': 'ml-test',
           'reason': 'тест',
           'types': {t: {'share': share or DEFAULT_SHARE[t], 'reject_k': DEFAULT_K[t]}
                     for t in TYPES}}
    for t, k in (reject_k or {}).items():
        raw['types'][t]['reject_k'] = k
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    return p


def make_settings(tmp: Path, **kwargs) -> OperatingSettings:
    return OperatingSettings.load(write_operating(tmp, **kwargs))