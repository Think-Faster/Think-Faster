"""Игнорируемые периоды — брак данных (INTEGRATION §1.5, команда settings.gaps).

В исследовании это константа `config.GAPS`; в сервисе — таблица главного диспетчера с версиями, как
график работ. Правило у неё обратное графику: брак портит то, чему модель научилась, поэтому правка
таблицы ставит флаг «нужно переобучение», а до него сервис только не пускает эти часы в признаки
(`retro.snapshot(holes=...)`) и в историю порога. Переобучение получает ту же таблицу через TF_GAPS.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import svc as config

NAME = 'gaps.json'
YEAR_2021 = (datetime(2021, 1, 1), datetime(2022, 1, 1))   # две системы мониторинга (plan.md §5)


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(str(s).strip().replace('T', ' '))


def validate(rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows, 1):
        try:
            a, b = _ts(r['a']), _ts(r['b'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'строка {i}: начало и конец — ГГГГ-ММ-ДД ЧЧ:ММ') from None
        if not a < b:
            raise ValueError(f'строка {i}: начало не раньше конца')
        out.append({'a': a.strftime('%Y-%m-%d %H:%M'), 'b': b.strftime('%Y-%m-%d %H:%M'),
                    'comment': str(r.get('comment') or '')})
    return sorted(out, key=lambda r: r['a'])


def _write(path: Path, raw: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, path)


class Gaps:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.path = self.folder / NAME
        self.versions = self.folder / 'gaps_versions'
        if not self.path.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            rows = [{'a': a.strftime('%Y-%m-%d %H:%M'), 'b': b.strftime('%Y-%m-%d %H:%M'),
                     'comment': 'потеря данных (config.GAPS)'} for a, b in config.GAPS]
            _write(self.path, {'version': 1, 'changed': None, 'changed_by': 'поставка',
                               'reason': 'из pipeline/config.py', 'rows': rows})
        self.raw = json.loads(self.path.read_text(encoding='utf-8'))

    @property
    def version(self) -> int:
        return int(self.raw.get('version', 1))

    @property
    def rows(self) -> list[dict]:
        return self.raw.get('rows', [])

    def intervals(self) -> list[tuple[datetime, datetime]]:
        return [(_ts(r['a']), _ts(r['b'])) for r in self.rows]

    def covers(self, ts: datetime) -> bool:
        return any(a <= ts < b for a, b in self.intervals())

    def apply(self) -> None:
        """Константы исследования на месте: их читают labels/features/sensor в этом процессе."""
        import features as ft
        iv = self.intervals()
        config.GAPS[:] = iv
        ft.HOLES[:] = iv + [YEAR_2021]

    def replace(self, rows: list[dict], version: int, changed_by: str, reason: str | None,
                now: datetime) -> dict:
        rows = validate(rows)
        if version <= self.version:
            raise ValueError(f'версия {version} не новее текущей {self.version}')
        self.versions.mkdir(parents=True, exist_ok=True)
        _write(self.versions / f'v{self.version}.json', self.raw)
        old = {(r['a'], r['b']) for r in self.rows}
        new = {(r['a'], r['b']) for r in rows}
        self.raw = {'version': int(version), 'changed': now.isoformat(), 'changed_by': changed_by,
                    'reason': reason, 'rows': rows}
        _write(self.path, self.raw)
        self.apply()
        return {'version': int(version), 'added': sorted(' — '.join(x) for x in new - old),
                'removed': sorted(' — '.join(x) for x in old - new)}
