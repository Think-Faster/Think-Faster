"""Предиктор (M4): загрузка выгрузки по manifest и счёт шести типов.

Оценка типа — среднее рангов пяти зёрен (INTEGRATION §2.1): каждое зерно даёт вероятность,
мы превращаем её в место строки в распределении оценок парка этого часа, среднее по зёрнам —
0..1 и есть score. Порог применяется не здесь, а в threshold/rules.

Для отказа оборудования выгрузка может нести сети TCN (manifest.seq.present) — тогда по типу
`equipment` считается сеть, а бустинг по нему пропускается.
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


class Predictor:
    """Собранные модели выгрузки + ранг-среднее по зёрнам."""

    def __init__(self, export: Path | None = None):
        self.export = Path(export) if export else config.EXPORT
        self.manifest = load_manifest(self.export)
        self.mix = None        # грузится лениво (нужны xgboost/catboost/torch)
        self.net = None
        self.loaded = False

    def _ensure(self):
        if self.loaded:
            return
        import retro
        self.mix = retro.load_mix_models(self.export)          # тип → {сеeds: lambda}
        if self.manifest['seq'].get('present') and 'nets' in self.manifest.get('seq', {}):
            try:
                self.net = retro.load_nets(self.export)
            except (ImportError, KeyError) as e:
                print(f'  сети TCN недоступны ({e}) — по «отказу оборудования» идёт бустинг')
        self.loaded = True

    def features(self, frame: pl.DataFrame) -> np.ndarray:
        cols = self.manifest['features']
        return frame.select(cols).to_numpy().astype(np.float32)

    def predict(self, frame: pl.DataFrame, seq_in: dict | None = None) -> dict:
        """Возврат: тип → массив score по порядку строк frame."""
        self._ensure()
        X = self.features(frame)
        out = {}
        if self.net is not None and seq_in is not None:
            import retro
            out['equipment'] = retro.net_scores(self.net, seq_in['x'], seq_in['s'])[:, 3]
        for tp, block in self.mix.items():
            if tp in out:
                continue
            seed_prob = np.stack([fn(X) for fn in block['seeds'].values()])
            out[tp] = rank_mean(seed_prob)
        return out

    def reasons(self, frame: pl.DataFrame, idx: int, tp: str, k: int = 5) -> list[dict]:
        """Основания тревоги (ТЗ §5): топ признаков строки по важности модели (приближённо).

        Важность берём из манифеста (mean по зёрнам); значения строки — из frame. Это не SHAP и не
        вклад в конкретное решение, а «что в этой строке выделяется у типа» — для карточки.
        """
        block = self.manifest['models'].get(tp, {})
        imp = {}
        for seed, info in block.get('seeds', {}).items():
            imp_seed = info.get('importance')
            if imp_seed:
                for f, w in imp_seed.items():
                    imp[f] = imp.get(f, 0.0) + w / max(len(block['seeds']), 1)
        feats = self.manifest['features']
        feats = [f for f in feats if f in (imp or {})]
        feats.sort(key=lambda f: -abs(imp[f]))
        out = []
        for f in feats[:k]:
            out.append({'feature': f, 'value': float(frame[idx][f])})
        return out


def rank_mean(prob: np.ndarray) -> np.ndarray:
    """Среднее по зернам `prob` (зёрна × строки) мест строк в распределении оценок парка."""
    n = prob.shape[1]
    if n < 2:
        return np.full(n, 0.5, np.float32)
    ranked = np.argsort(np.argsort(prob, axis=1), axis=1).astype(np.float32) / (n - 1)
    return ranked.mean(axis=0)