"""Предиктор (M4): выгрузка по манифесту и счёт шести типов (П1).

Выгрузка — `work/export` при разработке или пакет модели `TF_MODEL_BUNDLE` (/models/current, §13.6).
Манифест main: `models[tp][ключ] = {file, family, from_run, ...}`; ключ — номер зерна (`'0'`) или
`<семейство>_s<зерно>` (выгрузки версий, `export.py --equipment-version`). Версия модели типа (M9a,
раздел 60) — папка рядом: `work/export_<тип>_v<N>` или `<пакет>/<тип>_v<N>`; её `models` и `blend`
заменяют тип основного манифеста, остальные типы общие.

Оценка зерна кладётся на отсортированную шкалу проверки 2025, как `retro.load_mix_models()`:
`searchsorted(sorted_2025, p, side='right') / len`. Смесь типа — по частям `blend`: среднее рангов
зёрен семейства, части — со своими весами; без `blend` (main) — среднее всех зёрен.

Шкала зерна (оценки проверки 2025 в порядке витрины) ищется так:
1. `<папка выгрузки>/scales/<тип>_<семейство>_s<зерно>.npy` — пакет модели;
2. сеть прогона вперёд: `work/roll/<from_run>_<тип>_val.npy` (`seqmodel.py --score-val`, Н20);
3. `work/runs/<прогон>/preds/<семейство>_<тип>_val.npy` — как `operating.split`/`retro.load_mix_models`.
Те же файлы дают начальную историю порога (`bootstrap_history`).

Пункты 2–3 — только для разработки: они ищут по имени прогона, а имя модель не удостоверяет.
Бустинги выгрузки переобучены на 2022–2026 (`export.py`) и с моделями прогонов не совпадают, а в
work под теми же именами лежат прежние модели. Пакет (`bundle.locate`) считает шкалу бустинга
заново — оценки самой модели выгрузки на входах сервиса, шкалу сети проверяет по весам — и
записывает файл в `e['scale']`: он берётся первым.
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import polars as pl

import svc as config

CALIBRATION = 'calibration.json'
log = logging.getLogger('tf-model')


def load_manifest(export: Path | None = None) -> dict:
    p = (export or config.EXPORT) / 'manifest.json'
    assert p.exists(), f'нет выгрузки {p} — соберите её pipeline/export.py (INTEGRATION §3)'
    return json.loads(p.read_text(encoding='utf-8'))


def seed_of(key) -> int:
    """Зерно по ключу манифеста: `'3'` (main) или `'tcn_s3'` (версии)."""
    key = str(key)
    if key.isdigit():
        return int(key)
    m = re.search(r'_s(\d+)$', key)
    assert m, f'ключ модели без зерна: {key}'
    return int(m.group(1))


def version_dir(export: Path, tp: str, n: int) -> Path | None:
    """Папка версии `n` типа `tp`: в пакете — `<пакет>/<тип>_v<n>`, при разработке — `work/export_<тип>_v<n>`."""
    for d in (export / f'{tp}_v{n}', export.parent / f'export_{tp}_v{n}'):
        if (d / 'manifest.json').exists():
            return d
    return None


def available_versions(export: Path | None = None) -> dict[str, list[dict]]:
    """Тип → собранные версии `[{number, name, about, built}]` — для /status и model.switch."""
    export = Path(export or config.EXPORT)
    out: dict[str, list[dict]] = {}
    pats = [(export, re.compile(r'^([a-z]+)_v(\d+)$')), (export.parent, re.compile(r'^export_([a-z]+)_v(\d+)$'))]
    for root, pat in pats:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = pat.match(d.name)
            if not m or not (d / 'manifest.json').exists():
                continue
            v = json.loads((d / 'manifest.json').read_text(encoding='utf-8')).get('version', {})
            row = {'number': int(m.group(2)), 'name': v.get('name'), 'about': v.get('about'),
                   'built': json.loads((d / 'manifest.json').read_text(encoding='utf-8')).get('built')}
            if all(r['number'] != row['number'] for r in out.get(m.group(1), [])):
                out.setdefault(m.group(1), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r['number'])
    return out


def _is_iso_date(tail: str) -> bool:
    return (len(tail) == 10 and tail[4] == '-' and tail[7] == '-'
            and tail[:4].isdigit() and tail[5:7].isdigit() and tail[8:].isdigit())


def _candidate_runs(from_run: str) -> list[str]:
    """Рабочие прогоны, в preds которых может лежать шкала зерна: сам `from_run` и его ветка без даты."""
    runs = [from_run]
    if _is_iso_date(from_run[-10:]):
        stem = from_run[:-11]
        runs.append(stem)
        if stem.endswith('_s0'):
            runs.append(stem[:-3])
    return runs


def val_scale_file(from_run: str, family: str, seed: int, tp: str, work: Path | None,
                   root: Path | None = None) -> Path | None:
    """Файл оценок проверки 2025 зерна (порядок строк витрины 2025) или None."""
    if root is not None:
        p = root / 'scales' / f'{tp}_{family}_s{seed}.npy'
        if p.exists():
            return p
    if work is None:
        return None
    if family == 'tcn':
        # сеть прогона вперёд называется своим прогоном — это точный файл именно этой сети
        p = work / 'roll' / f'{from_run}_{tp}_val.npy'
        if p.exists():
            return p
        names = [f'tcn_s{seed}_{tp}_val.npy', f'tcn_{tp}_val.npy']
    else:
        names = [f'{family}_{tp}_val.npy']
    for run in _candidate_runs(from_run):
        for n in names:
            p = work / 'runs' / run / 'preds' / n
            if p.exists():
                return p
    return None


class Predictor:
    """Модели выгрузки + смесь зёрен на шкале проверки 2025.

    `versions` — выбранная версия по типу (`{'equipment': 1}`), переключается командой model.switch.
    Модели грузятся лениво (xgb/cat/torch — опциональные зависимости), шкалы кэшируются на процесс.
    """

    def __init__(self, export: Path | None = None, work: Path | None = None,
                 versions: dict[str, int] | None = None):
        self.export = Path(export) if export else config.EXPORT
        self.bundle = (self.export / 'scales').exists()
        self.work = None if self.bundle and work is None else Path(work) if work else config.WORK
        self.manifest = load_manifest(self.export)
        meta = self.export / 'meta.json' if self.bundle else self.work / 'features' / 'meta.json'
        self.meta = json.loads(meta.read_text(encoding='utf-8'))
        self.features = self.manifest['features']
        self.versions: dict[str, int] = {}
        self.entries: dict[str, list[dict]] = {}
        self.blends: dict[str, list[dict]] = {}
        self.manifests: dict[str, dict] = {}
        for tp, models in self.manifest['models'].items():
            self._set_type(tp, models, self.manifest.get('blend'), self.export, self.manifest)
        for tp, n in (versions or {}).items():
            self.use_version(tp, n)
        self._boosters: dict[Path, object] = {}
        self._nets: dict[Path, object] = {}
        self._scales: dict[tuple, np.ndarray] = {}
        self._importance: dict[str, dict] = {}
        self._calib: dict[tuple, dict | None] = {}

    def _set_type(self, tp: str, models: dict, blend, root: Path, manifest: dict) -> None:
        self.entries[tp] = [{'key': k, 'seed': seed_of(k), 'family': v['family'], 'path': root / v['file'],
                             'from_run': v.get('from_run', ''), 'root': root} for k, v in models.items()]
        self.blends[tp] = blend or []
        self.manifests[tp] = manifest
        getattr(self, '_importance', {}).pop(tp, None)     # важность — от модели выбранной версии

    def use_version(self, tp: str, n: int | None) -> None:
        """Версия `n` типа `tp` вместо модели main; `None` или 0 — вернуть модель main."""
        if not n:
            self._set_type(tp, self.manifest['models'][tp], self.manifest.get('blend'), self.export,
                           self.manifest)
            self.versions.pop(tp, None)
            return
        d = version_dir(self.export, tp, int(n))
        if d is None:
            raise FileNotFoundError(f'нет собранной версии {n} типа {tp} (export.py --{tp}-version {n})')
        m = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
        if m.get('features', self.features) != self.features:
            raise ValueError(f'версия {n} типа {tp} собрана на других признаках, чем выгрузка main')
        self._set_type(tp, m['models'][tp], m.get('blend'), d, m)
        self.versions[tp] = int(n)

    @property
    def version(self) -> str:
        """Версия выгрузки: дата сборки main и выбранные версии типов (`2026-09-25+equipment_v1`)."""
        tail = ''.join(f'+{tp}_v{n}' for tp, n in sorted(self.versions.items()))
        return f"{self.manifest.get('built', '')}{tail}"

    # ----- модели --------------------------------------------------------
    def has_nets(self) -> bool:
        return any(e['family'] == 'tcn' for es in self.entries.values() for e in es)

    def _ensure(self):
        need_nets = [e for es in self.entries.values() for e in es
                     if e['family'] == 'tcn' and e['path'] not in self._nets]
        if need_nets:
            import torch
            from seqmodel import Net
            for e in need_nets:
                state = torch.load(e['path'], map_location='cpu', weights_only=True)
                width, c_in = state['inp.weight'].shape[:2]
                net = Net(c_in, state['head.0.weight'].shape[1] - 2 * width, len(config.TYPES), width, 0.0)
                net.load_state_dict(state)
                self._nets[e['path']] = net.eval()
        for es in self.entries.values():
            for e in es:
                if e['family'] != 'tcn' and e['path'] not in self._boosters:
                    self._boosters[e['path']] = _load_booster(e['path'], e['family'])

    def matrix(self, frame: pl.DataFrame) -> np.ndarray:
        return frame.select(self.features).to_numpy().astype(np.float32)

    def scale_file(self, tp: str, e: dict) -> Path:
        p = e.get('scale') or val_scale_file(e['from_run'], e['family'], e['seed'], tp, self.work, e['root'])
        if p is None:
            raise FileNotFoundError(
                f'нет шкалы 2025 для {tp}/{e["key"]} ({e["from_run"]}): нужен scales/{tp}_{e["family"]}_s'
                f'{e["seed"]}.npy в пакете, roll/{e["from_run"]}_{tp}_val.npy (seqmodel.py --score-val) '
                f'или runs/<прогон>/preds/*_{tp}_val.npy')
        return p

    def scale(self, tp: str, e: dict) -> np.ndarray:
        key = (tp, e['path'])
        if key not in self._scales:
            self._scales[key] = np.sort(np.load(self.scale_file(tp, e)))
        return self._scales[key]

    def check(self) -> list[str]:
        """Чего не хватает для счёта: модели и шкалы всех зёрен (пустой список — всё на месте)."""
        miss = []
        for tp, es in self.entries.items():
            for e in es:
                if not e['path'].exists():
                    miss.append(f'модель {e["path"]}')
                if not e.get('scale') and val_scale_file(e['from_run'], e['family'], e['seed'], tp, self.work,
                                                         e['root']) is None:
                    miss.append(f'шкала {tp}/{e["key"]} ({e["from_run"]})')
        return miss

    def _netcols(self, pack: tuple) -> dict[Path, np.ndarray]:
        """Оценки сетей на входе pack: файл сети → массив по строкам (сигмоида по всем типам)."""
        import torch
        x, s = (torch.from_numpy(a) for a in pack)
        cols = {}
        with torch.no_grad():
            for path, net in self._nets.items():
                cols[path] = torch.sigmoid(net(x, s).float()).cpu().numpy()
        return cols

    def _blend(self, tp: str, parts: dict[str, list[np.ndarray]]) -> np.ndarray:
        if not self.blends.get(tp):
            return np.mean([p for ps in parts.values() for p in ps], axis=0).astype(np.float32)
        total = sum(b['weight'] for b in self.blends[tp])
        mix = sum(b['weight'] / total * np.mean(parts[b['family']], axis=0) for b in self.blends[tp])
        return np.asarray(mix, dtype=np.float32)

    def predict(self, frame: pl.DataFrame, seqdata: tuple | None = None) -> dict:
        """Тип → смесь оценок по порядку строк frame (0..1, шкала проверки 2025)."""
        self._ensure()
        X = self.matrix(frame)
        netcols = self._netcols(seqdata) if seqdata is not None and self.has_nets() else {}
        out = {}
        for tp, es in self.entries.items():
            parts: dict[str, list] = {}
            for e in es:
                if e['family'] == 'tcn':
                    p = netcols[e['path']][:, config.TYPES.index(tp)]
                else:
                    p = self._boosters[e['path']](X)
                base = self.scale(tp, e)
                parts.setdefault(e['family'], []).append(np.searchsorted(base, p, side='right') / len(base))
            out[tp] = self._blend(tp, parts)
        return out

    # ----- история и шкала -------------------------------------------------
    def hours_2025(self) -> np.ndarray:
        """Час каждой строки витрины 2025 — порядок, в котором лежат шкалы."""
        if self.bundle:
            return np.load(self.export / 'history' / 'h_2025.npy')
        return pl.scan_parquet(self.work / 'features' / '2025.parquet').select('h').collect()['h'].to_numpy()

    def bootstrap_history(self, year: int = 2025) -> dict:
        """(h_hist, p_hist) смеси по строкам витрины 2025 — начальная история порога.

        Как `retro.mix_history`: оценка зерна переводится в долю его же оценок на 2025
        (searchsorted/len), смесь — как в `predict`, массив парка сортируется по часу.
        """
        assert year == 2025, 'шкалы и история — проверка 2025'
        h = self.hours_2025()
        order = np.argsort(h, kind='stable')
        out = {}
        for tp, es in self.entries.items():
            parts: dict[str, list] = {}
            for e in es:
                raw = np.load(self.scale_file(tp, e)).astype(np.float32)
                assert len(raw) == len(h), f'шкала {tp}/{e["key"]}: {len(raw)} строк, витрина 2025 — {len(h)}'
                parts.setdefault(e['family'], []).append(np.searchsorted(np.sort(raw), raw, side='right') / len(raw))
            if parts:
                out[tp] = (h[order], self._blend(tp, parts)[order])
        return out

    # ----- уверенность (analytics §6, INTEGRATION §2.1) -----------------------
    def labels_2025(self, tp: str) -> np.ndarray:
        """Метка строки витрины 2025: эпизод типа в ближайшие `HORIZON` ч (как у обучения и `confidence.py`)."""
        assert self.work is not None, 'метки 2025 есть только в рабочей папке: пакет везёт готовую калибровку'
        nxt = pl.scan_parquet(self.work / 'features' / '2025.parquet').select(f'next_{tp}').collect()
        return (nxt[f'next_{tp}'].fill_null(np.inf).to_numpy() <= config.HORIZON).astype(np.int8)

    def mix_2025(self, tp: str) -> np.ndarray:
        """Смесь типа на строках витрины 2025 в их порядке — те же ранги, что `bootstrap_history`."""
        parts: dict[str, list] = {}
        for e in self.entries[tp]:
            raw = np.load(self.scale_file(tp, e)).astype(np.float32)
            parts.setdefault(e['family'], []).append(np.searchsorted(np.sort(raw), raw, side='right') / len(raw))
        return self._blend(tp, parts)

    def calibration(self, tp: str) -> dict | None:
        """Точки изотонической калибровки типа `{x, y}`: `calibration.json` пакета (или папки версии),
        при разработке — обучается на проверке 2025 рабочей папки. None — калибровать не на чем."""
        es = self.entries.get(tp) or []
        key = (tp, es[0]['root'] if es else None)
        if key in self._calib:
            return self._calib[key]
        cal = None
        f = key[1] / CALIBRATION if key[1] is not None else None
        if f is not None and f.exists():
            cal = json.loads(f.read_text(encoding='utf-8')).get(tp)
        elif self.work is not None and es:
            try:
                cal = fit_calibration(self.mix_2025(tp), self.labels_2025(tp))
            except (FileNotFoundError, AssertionError, pl.exceptions.PolarsError) as e:
                log.warning('калибровки %s нет, confidence не выдаётся: %s', tp, e)
                cal = None
        self._calib[key] = cal
        return cal

    def confidence(self, tp: str, score) -> np.ndarray | None:
        """Доля подтвердившихся тревог с такой оценкой на проверке 2025 — калиброванная вероятность."""
        cal = self.calibration(tp)
        if not cal:
            return None
        return np.interp(np.asarray(score, dtype=np.float64), cal['x'], cal['y'])

    # ----- основания тревоги (ТЗ §5) -----------------------------------------
    def importance(self, tp: str) -> dict[str, float]:
        """Важность признаков типа: `importance.json` пакета или отчёты прогонов-родителей."""
        if tp in self._importance:
            return self._importance[tp]
        es = self.entries.get(tp, [])
        imp: dict[str, float] = {}
        for root in {e['root'] for e in es}:
            f = root / 'importance.json'
            if f.exists():
                imp = json.loads(f.read_text(encoding='utf-8')).get(tp, {})
        if not imp and self.work is not None:
            for e in es:
                for f in sorted((self.work / 'runs' / e['from_run']).glob('report_*.json')):
                    im = json.loads(f.read_text(encoding='utf-8')).get(tp, {}).get('importance', {}).get(e['family'])
                    for name, w in dict(im or {}).items():      # train.importance — список пар [признак, вес]
                        imp[name] = imp.get(name, 0.0) + w / max(len(es), 1)
        self._importance[tp] = imp
        return imp

    def reasons(self, frame: pl.DataFrame, idx: int, tp: str, k: int = 5) -> list[dict]:
        """Топ признаков строки по важности модели типа.

        Не SHAP и не вклад в конкретное решение: важность прогона-родителя (у сетей её нет — пусто).
        """
        imp = self.importance(tp)
        feats = sorted((f for f in self.features if f in imp), key=lambda f: -abs(imp[f]))
        vals = [(f, frame[idx, f]) for f in feats[:k]]
        import features as ft  # pipeline/features.py: подписи признаков для карточки
        stypes = self.meta.get('stypes')
        return [{'feature': f, 'label': ft.describe(f, stypes), 'value': None if v is None else float(v)}
                for f, v in vals]


def fit_calibration(mix: np.ndarray, y: np.ndarray) -> dict:
    """Изотоническая регрессия «смесь → эпизод в ближайшие 24 ч» (analytics §6, `confidence.py`).

    Наружу — точки излома ступенчатой кривой: сервис переводит оценку в вероятность `np.interp`,
    без sklearn. Шкала монотонна: у оценки выше вероятность не ниже.
    """
    from sklearn.isotonic import IsotonicRegression
    mix, y = np.asarray(mix, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert len(mix) == len(y) and len(y), (len(mix), len(y))
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(mix, y)
    return {'x': [round(float(v), 6) for v in iso.X_thresholds_],
            'y': [round(float(v), 5) for v in iso.y_thresholds_],
            'rows': int(len(y)), 'positives': int(y.sum())}


def _load_booster(path: Path, family: str):
    if family == 'xgb':
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(str(path))
        b.set_param({'device': 'cpu'})
        return lambda X: b.inplace_predict(X)
    from catboost import CatBoostClassifier
    m = CatBoostClassifier()
    m.load_model(str(path))
    return lambda X: m.predict_proba(X)[:, 1]
