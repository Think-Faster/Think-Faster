"""Предиктор (M4): выгрузка `work/export` по манифесту main и счёт шести типов (П1).

Читает манифест в формате main (`models[tp][seed] = {file, family, params, trees, from_run}`,
сети — `equipment/tcn_s*.pt`) — без единой правки манифеста.

Оценка зерна кладётся на отсортированную шкалу валидации 2025, как `retro.load_mix_models()`:
`score = searchsorted(sorted_2025_scores, p, side='right') / len`; смесь по типу — среднее зёрен
по весам из манифеста (у main веса равные). Шкала каждого зерна — оценки его же прогона на
проверке (родитель seed в `from_run`), те же файлы `preds/<семейство>_<тип>_val.npy`, что читает
`operating.split`/`retro.load_mix_models`. Для сети отказа оборудования шкала — `tcn_s<seed>_<тип>
_val.npy` того же прогона. Истории с той же шкалой (bootstrap) отдаёт `bootstrap_history()`.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl

import svc as config


def load_manifest(export: Path | None = None) -> dict:
    p = (export or config.EXPORT) / 'manifest.json'
    assert p.exists(), f'нет выгрузки {p} — соберите её pipeline/export.py (INTEGRATION §3)'
    return json.loads(p.read_text(encoding='utf-8'))


def val_scale_file(from_run: str, family: str, seed: int, tp: str, work: Path) -> Path | None:
    """Файл оценок 2025 зерна — как его ищет `operating.split`/`retro.load_mix_models`."""
    d = work / 'runs' / from_run / 'preds'
    if family == 'tcn':
        names = [f'{family}_s{seed}_{tp}_val.npy', f'tcn_{tp}_val.npy']
    else:
        names = [f'{family}_{tp}_val.npy']
    for n in names:
        p = d / n
        if p.exists():
            return p
    return None


class Predictor:
    """Модели выгрузки + смесь зёрен на шкале проверки 2025.

    Модели грузятся лениво (xgb/cat/torch — опциональные зависимости), шкалы — по первому
    обращению и кэшируются на память процесса.
    """

    def __init__(self, export: Path | None = None, work: Path | None = None):
        self.export = Path(export) if export else config.EXPORT
        self.work = Path(work) if work else config.WORK
        self.manifest = load_manifest(self.export)
        self.meta = json.loads((self.work / 'features' / 'meta.json').read_text(encoding='utf-8'))
        self.features = self.manifest['features']
        self._boosters: dict[tuple, object] = {}
        self._nets: dict[int, object] = {}
        self._scales: dict[tuple, np.ndarray] = {}
        self.loaded = False

    # ----- модели --------------------------------------------------------
    def has_nets(self) -> bool:
        return any(b['family'] == 'tcn' for tp in self.manifest['models'].values()
                   for b in tp.values())

    def _ensure(self):
        if self.loaded:
            return
        if self.has_nets():
            import torch
            from seqmodel import Net
            for tp, seeds in self.manifest['models'].items():
                for seed, info in seeds.items():
                    if info['family'] != 'tcn' or int(seed) in self._nets:
                        continue
                    state = torch.load(self.export / info['file'], map_location='cpu', weights_only=True)
                    width, c_in = state['inp.weight'].shape[:2]
                    net = Net(c_in, state['head.0.weight'].shape[1] - 2 * width,
                              len(config.TYPES), width, 0.0)
                    net.load_state_dict(state)
                    self._nets[int(seed)] = net.eval()
        for tp, seeds in self.manifest['models'].items():
            for seed, info in seeds.items():
                if info['family'] == 'tcn' or (tp, int(seed)) in self._boosters:
                    continue
                self._boosters[(tp, int(seed))] = _load_booster(self.export / info['file'],
                                                                info['family'])
        self.loaded = True

    def matrix(self, frame: pl.DataFrame) -> np.ndarray:
        return frame.select(self.features).to_numpy().astype(np.float32)

    def scale(self, tp: str, seed: int, from_run: str, family: str) -> np.ndarray:
        key = (tp, seed, from_run)
        if key not in self._scales:
            p = val_scale_file(from_run, family, seed, tp, self.work)
            if p is None:
                raise FileNotFoundError(
                    f'нет шкалы 2025 для {tp}/зерна {seed} ({family}): ждём {val_scale_file(from_run, family, seed, tp, self.work)} '
                    f'в runs/{from_run}/preds — это тот же файл, что читает retro.load_mix_models')
            self._scales[key] = np.sort(np.load(p))
        return self._scales[key]

    def _netcols(self, pack: tuple) -> dict[int, np.ndarray]:
        """Оценки сетей на входе pack: seed → массив по строкам (сигмоида по всем типам)."""
        import torch
        x, s = (torch.from_numpy(a) for a in pack)
        cols = {}
        for seed, net in self._nets.items():
            with torch.no_grad():
                p = torch.sigmoid(net(x, s).float()).cpu().numpy()
            cols[seed] = p
        return cols

    def predict(self, frame: pl.DataFrame, seqdata: tuple | None = None) -> dict:
        """Тип → смесь оценок по порядку строк frame (0..1, шкала проверки 2025)."""
        self._ensure()
        X = self.matrix(frame)
        netcols = self._netcols(seqdata) if seqdata is not None else {}
        out = {}
        for tp, seeds in self.manifest['models'].items():
            parts = []
            for seed, info in seeds.items():
                if info['family'] == 'tcn':
                    p = netcols[int(seed)][:, config.TYPES.index(tp)]
                else:
                    p = self._boosters[(tp, int(seed))](X)
                base = self.scale(tp, int(seed), info['from_run'], info['family'])
                parts.append(np.searchsorted(base, p, side='right') / len(base))
            out[tp] = np.mean(parts, axis=0).astype(np.float32)
        return out

    # ----- история и шкала -------------------------------------------------
    def bootstrap_history(self, year: int = 2025) -> dict:
        """(h_hist, p_hist) смеси по строкам витрины года — начальная история для retro.rolling.

        Считается той же функцией, что и такт (predict), поэтому порог с первой же границы часа
        встаёт в ту же точку, что и у retro на тех же моделях. Массив парка сортируется по часу —
        как `retro.mix_history`, иначе retro.rolling не увидит окно.
        """
        years = (self.meta.get('years') or [year])
        if year not in years:
            years = [year]
        lf = pl.scan_parquet([self.work / 'features' / f'{y}.parquet' for y in years])
        df = lf.select(['object_id', 'h'] + self.features).collect()
        h = df['h'].to_numpy()
        if self.has_nets():
            # вход сетей по витрине не восстанавливается (нужны ряды журнала) — для истории сети
            # берём ранг-смесь по шкале без пересчёта входов, как retro.load_mix_models для S
            dfr = df.sort('h')
            scores = self.predict(dfr)
            hr = dfr['h'].to_numpy()
            return {tp: (hr, np.asarray(v, np.float32)) for tp, v in scores.items()}
        scores = self.predict(df)
        order = np.argsort(h, kind='stable')
        return {tp: (h[order], np.asarray(v[order], np.float32)) for tp, v in scores.items()}

    # ----- основания тревоги (ТЗ §5) -----------------------------------------
    def reasons(self, frame: pl.DataFrame, idx: int, tp: str, k: int = 5) -> list[dict]:
        """Топ признаков строки по важности из отчёта прогона-родителя (если file есть).

        Не SHAP и не вклад в конкретное решение; манифест main важности не несёт, поэтому берём
        importance из runs/<from_run>/report_*.json — того же источника, что thresholds() в retro.
        """
        seeds = self.manifest['models'].get(tp, {})
        imp: dict[str, float] = {}
        for seed, info in seeds.items():
            run = info.get('from_run', '')
            fam = info['family']
            for f in sorted((self.work / 'runs' / run).glob('report_*.json')):
                block = json.loads(f.read_text(encoding='utf-8')).get(tp, {})
                im = block.get('importance', {}).get(fam)
                if im:
                    for name, w in im.items():
                        imp[name] = imp.get(name, 0.0) + w / max(len(seeds), 1)
        feats = [f for f in self.features if f in imp]
        feats.sort(key=lambda f: -abs(imp[f]))
        return [{'feature': f, 'value': float(frame[idx][f])} for f in feats[:k]]


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