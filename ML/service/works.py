"""Молчание по графику работ (M7, INTEGRATION §1.6): таблица главного диспетчера поверх модели, без переобучения.

Окна — то же правило, что у разметки Н10 и отчёта `maintenance.py` (`labels.works_windows`): строка
накрывает свой объект и все объекты его коллектора, час входит в окно при a <= ts < b. В год, где у
пары (объект, вид работ) строки нет, окно переносится со строки ближайшего года на столько же лет с
запасом WORKS_PAD суток (при равном удалении — с более раннего года; 29 февраля → 28-е). Сервису
сверх разметки нужен номер строки (`work_id`) — для причины молчания, пометки факта и аудита.

Таблица версионна (§13.3, settings.works): каждая правка — новая версия, файл версии кладётся в
`works_versions/`, текущая — `works_2026.csv` в папке настроек сервиса; разметка (`labels.WORKS`)
смотрит туда же. Правка действует со следующего часа: такт читает окна на час своей границы.
"""
import csv
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import svc as config

COLUMNS = ['work_id', 'object_id', 'work_kind', 'incident_types', 'removed_sensor', 'starts_at', 'ends_at',
           'source', 'comment']
NAME = 'works_2026.csv'
PAD_DAYS = 7          # = labels.WORKS_PAD
REASON = 'плановые работы по графику'
# §1.6: прогноз отказа датчика в окне не глушится — он общий на объект, и молчание теряет настоящие
# отказы других датчиков; снятый работами датчик глушится только по факту (fact_window)
NOT_MUTED = ('sensor',)
FACT_NOTE = 'идёт ППР по графику'


@dataclass(frozen=True)
class Window:
    work_id: str
    object_id: int
    work_kind: str
    types: tuple
    sensor: str | None
    a: datetime
    b: datetime
    shifted: bool

    def covers(self, obj: int, collector: int | None, ts: datetime) -> bool:
        return self.object_id in (obj, collector) and self.a <= ts < self.b


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip().replace('T', ' '))


def _shift_years(t: datetime, n: int) -> datetime:
    try:
        return t.replace(year=t.year + n)
    except ValueError:                                   # 29 февраля в невисокосный год
        return t.replace(year=t.year + n, day=28)


def read_rows(path: Path) -> list[dict]:
    text = Path(path).read_text(encoding='utf-8-sig')
    return [{k: (v or '').strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(text), delimiter=';')]


def write_rows(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, delimiter=';', lineterminator='\n', extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, '') if r.get(k) is not None else '' for k in COLUMNS})
    tmp = path.with_suffix('.tmp')
    tmp.write_text(buf.getvalue(), encoding='utf-8')
    os.replace(tmp, path)


def validate(rows: list[dict]) -> list[dict]:
    """Строки таблицы из команды settings.works: обязательные поля, типы, порядок времени, уникальный номер."""
    seen, out = set(), []
    for i, r in enumerate(rows, 1):
        r = {k: ('' if r.get(k) is None else str(r.get(k)).strip()) for k in COLUMNS}
        if not r['work_id'] or r['work_id'] in seen:
            raise ValueError(f'строка {i}: номер работы пуст или повторяется')
        seen.add(r['work_id'])
        if r['object_id'] and not r['object_id'].isdigit():
            raise ValueError(f'строка {i}: object_id не число')
        types = [t for t in r['incident_types'].split(',') if t]
        if not types or any(t not in config.TYPES for t in types):
            raise ValueError(f'строка {i}: типы происшествий {r["incident_types"]!r} не из {config.TYPES}')
        try:
            a, b = _ts(r['starts_at']), _ts(r['ends_at'])
        except ValueError:
            raise ValueError(f'строка {i}: время начала или конца не в формате ГГГГ-ММ-ДД ЧЧ:ММ') from None
        if not a < b:
            raise ValueError(f'строка {i}: начало не раньше конца')
        r['starts_at'], r['ends_at'] = a.strftime('%Y-%m-%d %H:%M'), b.strftime('%Y-%m-%d %H:%M')
        out.append(r)
    return out


def windows(rows: list[dict], years, pad_days: int = PAD_DAYS) -> list[Window]:
    """Окна таблицы на годы `years` — как `labels.works_windows`: свои строки как есть плюс перенос."""
    w = []
    for r in rows:
        if not r.get('object_id'):
            continue
        w.append(Window(r['work_id'], int(r['object_id']), r['work_kind'], tuple(r['incident_types'].split(',')),
                        r.get('removed_sensor') or None, _ts(r['starts_at']), _ts(r['ends_at']), False))
    out = list(w)
    keys: dict[tuple, set] = {}
    for x in w:
        keys.setdefault((x.object_id, x.work_kind), set()).add(x.a.year)
    pad = timedelta(days=pad_days)
    for (obj, kind), wys in keys.items():
        for y in years:
            if y in wys:
                continue
            wy = min(wys, key=lambda v: (abs(v - y), v))
            for x in w:
                if x.object_id == obj and x.work_kind == kind and x.a.year == wy:
                    out.append(Window(x.work_id, obj, kind, x.types, x.sensor,
                                      _shift_years(x.a, y - wy) - pad, _shift_years(x.b, y - wy) + pad, True))
    return out


class Works:
    """Текущая таблица работ в папке настроек сервиса и её версии."""

    def __init__(self, folder: Path, seed: Path | None = None):
        self.folder = Path(folder)
        self.path = self.folder / NAME
        self.meta_path = self.folder / 'works.json'
        self.versions = self.folder / 'works_versions'
        if not self.path.exists():
            self.folder.mkdir(parents=True, exist_ok=True)
            src = seed or config.ML / 'settings' / NAME
            write_rows(self.path, read_rows(src))
            self.meta_path.write_text(json.dumps({'version': 1, 'changed': None, 'changed_by': 'поставка',
                                                  'reason': f'из {src.name}'}, ensure_ascii=False), encoding='utf-8')
        self.meta = json.loads(self.meta_path.read_text(encoding='utf-8')) if self.meta_path.exists() else {'version': 1}
        self.rows = read_rows(self.path)
        self._cache: dict[tuple, list[Window]] = {}

    @property
    def version(self) -> int:
        return int(self.meta.get('version', 1))

    def windows(self, years) -> list[Window]:
        key = tuple(sorted(set(years)))
        if key not in self._cache:
            self._cache[key] = windows(self.rows, key)
        return self._cache[key]

    def active(self, ts: datetime) -> list[Window]:
        """Окна, в которые попадает час ts (перенос — из прошлого и этого года)."""
        return [w for w in self.windows((ts.year - 1, ts.year)) if w.a <= ts < w.b]

    def mute(self, ts: datetime, objects, collectors: dict) -> dict[tuple, Window]:
        """(объект, тип) → окно, в котором прогноз этого типа молчит в час ts (M7)."""
        act = self.active(ts)
        if not act:
            return {}
        out = {}
        for o in objects:
            o = int(o)
            c = collectors.get(o)
            for w in act:
                if w.object_id in (o, c):
                    for tp in w.types:
                        if tp not in NOT_MUTED:
                            out.setdefault((o, tp), w)
        return out

    def fact_window(self, obj: int, collector: int | None, tp: str, t0: datetime,
                    stype: str | None = None) -> Window | None:
        """Окно, где начался факт: тип из строки или отказ снятого датчика (как Н10 в labels.build)."""
        for w in self.active(t0):
            if w.object_id in (obj, collector) and (tp in w.types or tp == 'sensor' and stype and stype == w.sensor):
                return w
        return None

    def covers(self, obj: int, collector: int | None, ts: datetime) -> Window | None:
        """Окно любого вида работ на объекте или его коллекторе в момент ts. Для аварий и слепоты
        §13.11 это полное молчание: без объявления, MUTED и события аудита."""
        for w in self.active(ts):
            if w.object_id in (obj, collector):
                return w
        return None

    def replace(self, rows: list[dict], version: int, changed_by: str, reason: str | None,
                now: datetime) -> dict:
        """Новая версия таблицы (settings.works): прежняя уходит в works_versions/. Возврат — разница для аудита."""
        rows = validate(rows)
        if version <= self.version:
            raise ValueError(f'версия {version} не новее текущей {self.version}')
        self.versions.mkdir(parents=True, exist_ok=True)
        old = {r['work_id']: r for r in self.rows}
        new = {r['work_id']: r for r in rows}
        write_rows(self.versions / f'v{self.version}.csv', self.rows)
        write_rows(self.path, rows)
        self.meta = {'version': int(version), 'changed': now.isoformat(), 'changed_by': changed_by, 'reason': reason}
        tmp = self.meta_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, self.meta_path)
        self.rows, self._cache = rows, {}
        return {'version': int(version), 'added': sorted(set(new) - set(old)),
                'removed': sorted(set(old) - set(new)),
                'changed': sorted(k for k in set(new) & set(old) if new[k] != old[k])}

    def point_labels(self) -> None:
        """Разметка Н10 живого журнала (`labels.build`) читает ту же текущую таблицу."""
        import labels
        labels.WORKS = self.path
