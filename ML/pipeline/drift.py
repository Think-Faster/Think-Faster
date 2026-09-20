"""Шаг 19. Скорость устаревания признаков: на каком окне дообучать и почему.

`retrain.py` отвечает, какая стратегия дообучения выигрывает по метрикам. Этот скрипт отвечает,
почему именно она, и делает это до всякого обучения — по самим данным. Вопрос ставится по каждому
признаку: через сколько суток его распределение перестаёт быть тем, на котором модель училась.

Мера — PSI (Population Stability Index), отраслевой стандарт контроля дрейфа: распределение
эталонного окна режется на децили, и считается, насколько разошлись доли в новом окне.
Принятые границы: до 0,1 сдвига нет, 0,1–0,25 умеренный, больше 0,25 — распределение другое и
модель работает не с теми данными, на которых училась.

Эталон — последние 90 суток обучающего периода: именно их модель видела последними. Дальше идут
окна по 30 суток, и для каждого признака ищется, на каком лаге PSI переходит границу.

Само по себе это ещё не ответ: у нас 306 признаков, и дрейф редкого признака ничего не стоит.
Поэтому второй проход взвешивает дрейф важностью признака в обученной модели каждого типа —
получается, через сколько суток устаревает то, на что модель реально опирается. Это и есть
рекомендуемый период дообучения, посчитанный отдельно для каждого типа происшествий.

    python drift.py
    python drift.py --step 6 --ref-days 90 --window 30
"""
import argparse
import json

import numpy as np
import polars as pl

import config
import train

FEAT = config.WORK / 'features'
DAY = 24
GRADES = [(0.10, 'умеренный'), (0.25, 'сильный')]

# семьи признаков: по смыслу, а не по алфавиту — дрейф интересен группой
FAMILY = {
    'календарь и сезон': ('hour', 'dow', 'month', 'doy', 'holiday', 'long', 'may9', 'days'),
    'люди на объекте': ('guard', 'arm', 'disarm', 'armed', 'motion', 'door', 'hatch', 'visit',
                        'arrival', 'manual'),
    'срабатывания каналов': ('trig', 'smoke', 'gas', 'flood', 'heat', 'fault', 'disconnected',
                             'undefined', 'norm', 'off', 'switch', 'av', 'noise'),
    'инженерные системы': ('temp', 'pump', 'fan', 'phase', 'ups', 'uir', 'comp'),
    'соседи по коллектору': ('coll',),
    'давность и новизна': ('since', 'onset', 'events', 'channels'),
}


def family(name: str) -> str:
    head = name.split('_')[0]
    for fam, heads in FAMILY.items():
        if head in heads:
            return fam
    return 'прочее'


def edges(ref: np.ndarray, bins: int = 10) -> np.ndarray | None:
    """Границы корзин по эталону. None — признак почти константа, дрейфа у него не бывает."""
    e = np.unique(np.quantile(ref[np.isfinite(ref)], np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        u = np.unique(ref[np.isfinite(ref)])
        if len(u) < 2:
            return None
        e = np.concatenate([u, [u[-1] + 1]])          # редкие значения — каждое своей корзиной
    e[0], e[-1] = -np.inf, np.inf
    return e


def psi(ref: np.ndarray, cur: np.ndarray, e: np.ndarray) -> float:
    """Насколько разошлись доли по корзинам. Пустая корзина подпирается eps, иначе log уходит в ∞."""
    eps = 1e-4
    a = np.clip(np.histogram(ref, e)[0] / max(len(ref), 1), eps, None)
    b = np.clip(np.histogram(cur, e)[0] / max(len(cur), 1), eps, None)
    return float(np.sum((b - a) * np.log(b / a)))


def crossing(lags: list[int], vals: np.ndarray, level: float) -> str:
    """Первый лаг, на котором PSI перешёл границу. Прочерк — не перешёл за всё наблюдение."""
    hit = np.where(vals >= level)[0]
    return f'{lags[hit[0]]}' if len(hit) else '—'


def gains(feats: list[str]) -> dict[str, np.ndarray]:
    """Важность признаков в обученных моделях прогона — веса для взвешенного дрейфа."""
    import xgboost as xgb
    out = {}
    for tp in config.TYPES:
        p = config.WORK / 'runs' / 'main_h24_tuned' / 'models' / f'xgb_{tp}.json'
        if not p.exists():
            continue
        b = xgb.Booster()
        b.load_model(str(p))
        g = np.zeros(len(feats), np.float64)
        for k, v in b.get_score(importance_type='total_gain').items():
            g[int(k[1:]) if k[0] == 'f' else feats.index(k)] = v
        s = g.sum()
        out[tp] = g / s if s else g
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=12, help='прореживание часов витрины')
    ap.add_argument('--ref-days', type=int, default=90, help='эталон: последние сутки обучения')
    ap.add_argument('--window', type=int, default=30, help='ширина окна наблюдения, суток')
    ap.add_argument('--bins', type=int, default=10)
    args = ap.parse_args()
    feats = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))['features']

    df = train.load([2024, 2025, 2026], args.step, ['h'] + feats)
    h = df['h'].to_numpy()
    X = train.matrix(df, feats)
    del df
    end = int((np.datetime64(str(config.TRAIN_END.date())) - np.datetime64('2019-01-01'))
              / np.timedelta64(1, 'h'))
    ref = (h >= end - args.ref_days * DAY) & (h < end)
    lags, masks = [], []
    d = 0
    while True:
        m = (h >= end + d * DAY) & (h < end + (d + args.window) * DAY)
        if m.sum() < 1000:
            break
        lags.append(d + args.window)
        masks.append(m)
        d += args.window
    print(f'эталон {int(ref.sum())} строк (последние {args.ref_days} сут обучения), '
          f'окон наблюдения {len(lags)} по {args.window} сут, признаков {len(feats)}', flush=True)

    P = np.full((len(feats), len(lags)), np.nan)
    flat = []
    for i, name in enumerate(feats):
        col = X[:, i]
        r = col[ref]
        e = edges(r, args.bins)
        if e is None:
            flat.append(name)
            continue
        for j, m in enumerate(masks):
            P[i, j] = psi(r, col[m], e)

    rows = []
    for i, name in enumerate(feats):
        if np.isnan(P[i]).all():
            continue
        rows.append({'признак': name, 'семья': family(name),
                     'PSI 90': round(float(P[i, min(2, len(lags) - 1)]), 3),
                     'PSI 180': round(float(P[i, min(5, len(lags) - 1)]), 3),
                     'PSI 360': round(float(P[i, min(11, len(lags) - 1)]), 3),
                     'умеренный с суток': crossing(lags, P[i], 0.10),
                     'сильный с суток': crossing(lags, P[i], 0.25)})
    tab = pl.DataFrame(rows)
    tab.write_csv(config.WORK / 'drift_features.csv')

    out = [f'Дрейф признаков. Эталон — последние {args.ref_days} суток обучающего периода '
           f'(до {config.TRAIN_END.date()}), дальше окна по {args.window} суток. '
           f'Мера — PSI по {args.bins} корзинам эталона.\n',
           f'Признаков без дрейфа по построению (почти константа): {len(flat)}.\n']

    out.append('\n## Семьи признаков\n')
    out.append('| семья | признаков | PSI ч/з 90 сут | ч/з 180 | ч/з 360 | '
               'перешли 0,25 за год |')
    out.append('|---|---:|---:|---:|---:|---:|')
    for fam in list(FAMILY) + ['прочее']:
        g = tab.filter(pl.col('семья') == fam)
        if not len(g):
            continue
        strong = sum(1 for v in g['сильный с суток'] if v != '—')
        out.append(f"| {fam} | {len(g)} | {g['PSI 90'].median():.3f} | {g['PSI 180'].median():.3f} | "
                   f"{g['PSI 360'].median():.3f} | {strong} из {len(g)} |".replace('.', ','))

    out.append('\n## Быстрее всех устаревают\n')
    out.append('| признак | семья | PSI ч/з 90 сут | ч/з 180 | ч/з 360 | сильный сдвиг с суток |')
    out.append('|---|---|---:|---:|---:|---:|')
    for r in tab.sort('PSI 180', descending=True).head(15).iter_rows(named=True):
        out.append(f"| {r['признак']} | {r['семья']} | {r['PSI 90']} | {r['PSI 180']} | "
                   f"{r['PSI 360']} | {r['сильный с суток']} |".replace('.', ','))

    out.append('\n## Дрейф, взвешенный важностью: через сколько суток устаревает то, '
               'на что модель опирается\n')
    out.append('Календарные признаки из взвешивания исключены. У них PSI в разы выше всех '
               'остальных, но это не дрейф: эталон взят за осень, окна наблюдения приходятся на '
               'другие сезоны, и `doy_sin`, `month`, `days_to_holiday` обязаны разойтись — так они '
               'и устроены. Переобучение их не «освежит». Ниже — обе версии, наивная для сверки.\n')
    w = gains(feats)
    head = ' | '.join(str(l) for l in lags)
    out.append(f'| тип | взвешивание | {head} | дообучать не реже |')
    out.append('|---' * (len(lags) + 3) + '|')
    ok = np.where(~np.isnan(P).all(axis=1))[0]
    cal = np.array([family(feats[i]) == 'календарь и сезон' for i in ok])
    for tp, g in w.items():
        for tag, sel in (('без календаря', ~cal), ('наивное, с календарём (отброшено)',
                                                   np.ones(len(ok), bool))):
            idx = ok[sel]
            gg = g[idx] / max(g[idx].sum(), 1e-9)
            wp = np.nansum(P[idx] * gg[:, None], axis=0)
            rec = crossing(lags, wp, 0.10)
            cells = ' | '.join(f'{v:.3f}'.replace('.', ',') for v in wp)
            out.append(f'| {config.TYPE_NAMES[tp]} | {tag} | {cells} | '
                       f"{rec + ' сут' if rec != '—' else 'ниже порога за всё наблюдение'} |")

    out.append('\n## Дрейф самой цели: как меняется доля часов, за которыми идёт происшествие\n')
    out.append('Признаки — это «что на объекте». Цель — «что с ним случается». Дрейфовать может и '
               'то, и другое, и лечится это по-разному: сдвиг признаков — переобучением на свежем '
               'окне, сдвиг доли происшествий — ещё и пересмотром порога, потому что порог, снятый '
               'на старой доле, на новой даёт другое число тревог.\n')
    tg = train.load([2024, 2025, 2026], args.step, ['h'] + [f'next_{t}' for t in config.TYPES])
    th = tg['h'].to_numpy()
    out.append('| тип | эталон | ' + ' | '.join(f'+{l}' for l in lags) + ' |')
    out.append('|---' * (len(lags) + 2) + '|')
    tref = (th >= end - args.ref_days * DAY) & (th < end)
    for tp in config.TYPES:
        y = (tg[f'next_{tp}'].to_numpy() <= config.HORIZON)
        b = float(y[tref].mean())
        cells = []
        for j in range(len(lags)):
            m = (th >= end + (lags[j] - args.window) * DAY) & (th < end + lags[j] * DAY)
            cells.append(f'{float(y[m].mean()):.4f}'.replace('.', ',') if m.sum() else '—')
        out.append(f"| {config.TYPE_NAMES[tp]} | {b:.4f} | ".replace('.', ',')
                   + ' | '.join(cells) + ' |')

    (config.WORK / 'drift.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))
    print('\n→ drift.md, drift_features.csv', flush=True)


if __name__ == '__main__':
    main()
