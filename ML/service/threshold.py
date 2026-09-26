"""Скользящий порог (M5): квантиль оценок всего парка за 90 суток до часа.

Ровно `retro.rolling()` из main — тот же код, то же окно, тот же номер квантиля 1-share; у сервиса
нет своей копии и, главное, нет админ-допуска «порог вручную» (threshold_override), который П7
просил выпилить, чтобы прод не ухал от исследования. Менять долю тревоги можно только настройкой
operating.json (доля → квантиль), и то же самое видит диспетчер в /api/ml/estimate.

Историю оценок текущей версии моделей сервис ведёт сам (каждый такт дописывает час) и стартует
её из витрины 2025 bootstrap-смесью той же версии — поэтому порог с самого первого часа стоит ровно
туда, куда его поставил бы retro. На смене версии моделей история пересчитывается на новой смеси.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

import svc as config
import settings as s


def _feat_hour(t: datetime) -> int:
    """Час в часах от начала витрины — тот же `h`, что в features.parquet"""
    import features as ft
    return int((t - ft.T0).total_seconds() // 3600)


def _thr(h_hist: np.ndarray, p_hist: np.ndarray, h: int, share: float) -> float:
    """extend.rolling() из main для одного часа: квантиль парка за 90 суток строго до h.

    Окно и квантиль — те же, что в retro.rolling (тот же поиск по часам). Единственное отличие
    от retro: если окно ещё не пересеклось с историей (бутстрап впереди хвоста парка, а порог
    нужен уже сейчас), берём последние 90 суток известной истории — порог без заглушки-константы
    и строго без заглядывания в будущее.
    """
    lo, hi = np.searchsorted(h_hist, [h - 90 * 24 + 1, h + 1])
    if lo >= hi:
        lo, hi = np.searchsorted(h_hist, [h_hist[-1] - 90 * 24 + 1, h_hist[-1] + 1])
    return float(np.quantile(p_hist[lo:hi], 1 - share))


class Thresholds:
    """Пороги по типам на момент t; приводится к такту через .estimate()."""

    def __init__(self, history_path: Path | None = None):
        self.history_path = Path(history_path) if history_path else config.HISTORY
        self.versions: dict[str, int] = {}      # тип → версия операционки, с которой читаем порог
        self._hist: dict[str, np.ndarray] = {}  # тип → (h_hist, p_hist) смеси на шкале 2025
        self.thresholds: dict[str, float] = {}
        self.model_version: str = ''            # версия выгрузки, с которой собрана история

    # ----- история --------------------------------------------------------
    def bootstrap(self, scores: dict, st: s.OperatingSettings, h: int,
                  model_version: str = '') -> None:
        """Начальная история смеси (П1/Д3): из 2025-витрины, сортировка по h как mix_history.

        merge=True потом: добавим и оба накопленных окна перед сменой версии (rebootstrap).
        """
        self.model_version = model_version
        for tp, (h_hist, p_hist) in scores.items():
            self._hist[tp] = (h_hist, p_hist)
            self.versions[tp] = st.version
            self.thresholds[tp] = _thr(h_hist, p_hist, h, st.share(tp))

    def extend(self, stamp: int, scores: dict, st: s.OperatingSettings) -> None:
        """Дописать час, который только что посчитан тактом, и сразу пересчитать порог.

        Оценки часа — по строке на объект (как строки парка в mix_history): час повторяется столько
        раз, сколько объектов в кадре. Пересчёт уже добавленного часа заменяет его хвост, а час,
        который покрыт бутстрапом истории (витрина 2025), не дублируется.
        """
        for tp, v in scores.items():
            hh, pp = self._hist[tp]
            v = np.atleast_1d(np.asarray(v, np.float32))
            start = len(hh)
            while start and hh[start - 1] == stamp:
                start -= 1
            if start < len(hh):
                # пересчёт того же часа (уже лежит хвостом): заменяем его блок, не плодим строки
                hh = np.concatenate([hh[:start], np.full(len(v), stamp, hh.dtype)])
                pp = np.concatenate([pp[:start], v])
            elif len(hh) and np.searchsorted(hh, stamp) < len(hh):
                pass                            # час уже покрыт историей — не дублируем строки
            else:
                # окно retro.rolling — 90 суток: не хранить больше, чем понадобится
                cut = np.searchsorted(hh, stamp - 90 * 24)
                hh = np.concatenate([hh[cut:], np.full(len(v), stamp, hh.dtype)])
                pp = np.concatenate([pp[cut:], v])
            self._hist[tp] = (hh, pp)
            self.thresholds[tp] = _thr(hh, pp, stamp, st.share(tp))

    def estimate(self, t: datetime, st: s.OperatingSettings) -> None:
        """Порог на ручку /api/ml/estimate и на границу часа такта (без дописывания истории)."""
        for tp in config.TYPES:
            if tp in self._hist:
                self.thresholds[tp] = _thr(*self._hist[tp], _feat_hour(t), st.share(tp))

    def apply_settings(self, st: s.OperatingSettings, h: int) -> list[str]:
        """Новая версия настроек → пороги на месте; история не пересчитывается (шум низкий)."""
        changed = [tp for tp in config.TYPES if self.versions.get(tp) != st.version]
        for tp in changed:
            self.versions[tp] = st.version
            if tp in self._hist:
                self.thresholds[tp] = _thr(*self._hist[tp], h, st.share(tp))
        return changed

    def refresh(self, h: int, st: s.OperatingSettings) -> None:
        """Порог на час h без дописывания истории — час в игнорируемом периоде (settings.gaps)."""
        for tp in config.TYPES:
            if tp in self._hist:
                self.thresholds[tp] = _thr(*self._hist[tp], h, st.share(tp))

    def drop_hours(self, spans: list[tuple[int, int]]) -> int:
        """Убрать из истории часы [a, b) игнорируемых периодов (M5: брак не идёт в порог)."""
        n = 0
        for tp, (hh, pp) in self._hist.items():
            keep = np.ones(len(hh), bool)
            for a, b in spans:
                keep &= ~((hh >= a) & (hh < b))
            if not keep.all():
                n += int((~keep).sum())
                self._hist[tp] = (hh[keep], pp[keep])
        return n

    def replace_type(self, tp: str, hist: tuple, st: s.OperatingSettings, h: int,
                     model_version: str) -> None:
        """model.switch (§9.4): история и порог одного типа — по ретропрогону новой версии."""
        self._hist[tp] = hist
        self.versions[tp] = st.version
        self.thresholds[tp] = _thr(*hist, h, st.share(tp))
        self.model_version = model_version

    def rebootstrap(self, scores: dict, st: s.OperatingSettings, h: int,
                    model_version: str = '') -> None:
        self.bootstrap(scores, st, h, model_version)
        self.dump()

    # ----- наблюдение/гид --------------------------------------------------
    def history(self, tp: str) -> tuple[np.ndarray, np.ndarray]:
        return self._hist[tp]

    def dump(self) -> None:
        if not self._hist:
            return
        parts = []
        mv = self.model_version
        for tp, (hh, pp) in self._hist.items():
            parts.append(pl.DataFrame({'tp': tp, 'h': hh, 'p': pp,
                                       'model_version': [mv] * len(hh)}))
        df = pl.concat(parts)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.history_path)

    def load(self, st: s.OperatingSettings, h: int, model_version: str = '') -> bool:
        if not self.history_path.exists():
            return False
        df = pl.read_parquet(self.history_path)
        if 'model_version' not in df.columns:
            return False                        # журнал старого формата — без версии, не доверяем
        saved = str(df['model_version'][0])
        self.model_version = saved
        if model_version and saved != model_version:
            return False                        # история собрана другой выгрузкой — пересчитать
        for tp in config.TYPES:
            sub = df.filter(pl.col('tp') == tp)
            if sub.height:
                self._hist[tp] = (sub['h'].to_numpy().astype(np.int64),
                                  sub['p'].to_numpy().astype(np.float32))
                self.thresholds[tp] = _thr(*self._hist[tp], h, st.share(tp))
                self.versions[tp] = st.version
        return bool(self._hist)