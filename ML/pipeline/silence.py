"""Тишина в журнале: тревожится ли прогноз там, где от объекта перестали приходить строки.

Разбор пропусков (раздел 44) смотрел, из чего рождается эпизод. Здесь обратный вопрос, важный для
эксплуатации: система живёт памятью об объекте, и когда объект замолкает — оборвалась связь,
выключился коллектор, потерян кусок журнала, — признаки не обнуляются, а «замораживаются» на
последнем известном уровне. Если при этом тревога продолжает гореть, диспетчер получает сигнал о
том, чего система уже не видит.

Меряется две вещи:

1. **По часам тишины.** Час относится к корзине по тому, сколько прошло с последней строки журнала
   по этому объекту (`n_events` в часовых рядах). В каждой корзине — сколько часов, какая доля из
   них под тревогой, сколько эпизодов начинается и сколько поймано. Смотреть надо на пару чисел:
   доля тревог в корзине против доли эпизодов в ней.
2. **После провала журнала.** Известные дыры (`config.GAPS`) — это потеря данных по всему парку.
   Считается поток тревог в первые сутки и первые трое суток после возобновления против среднего
   по периоду.

Порог — доля часов из настроек, общий за период (как в `daily.py`).

    python silence.py "$MIXT"
"""
from datetime import timedelta

import numpy as np

import config
import features as ft
from smooth import load

EDGES = [1, 7, 25, 73]
NAMES = ['есть строки', '1–6 ч', '7–24 ч', '25–72 ч', 'больше 72 ч']
SHARES = config.shares()
YEARS = {'val': 2025, 'test': 2026}


def since_quiet() -> tuple[dict, np.ndarray]:
    """Часы с последней строки журнала по объекту: {object_id: индекс строки}, матрица (объект, час)."""
    z = np.load(config.WORK / 'seq.npz')
    act = np.asarray(z['base'][:, :, ft.IDX['n_events']], np.float32) > 0
    t = np.arange(act.shape[1])
    last = np.maximum.accumulate(np.where(act, t, -10 ** 6), axis=1)
    return {int(o): i for i, o in enumerate(z['objects'])}, (t - last).astype(np.int32)


def bucket(since: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES, since, side='right')


def table(on: str, rows: dict, quiet: np.ndarray, idx: dict) -> None:
    print(f'\n**{on}**: часы по времени с последней строки журнала\n')
    print('| тип | корзина | часов | под тревогой | доля тревог типа | часов с эпизодом впереди | '
          'из них под тревогой | Precision |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for tp, d in rows.items():
        alarm = d['p'] >= np.quantile(d['p'], 1 - SHARES[tp])
        b = bucket(quiet[[idx[int(o)] for o in d['o']], d['h']])
        for k, name in enumerate(NAMES):
            m = b == k
            if not m.any():
                continue
            a = alarm & m
            print(f'| {config.TYPE_NAMES[tp]} | {name} | {int(m.sum())} | {int(a.sum())} | '
                  f'{a.sum() / max(alarm.sum(), 1):.3f} | {int((d["y"] & m).sum())} | '
                  f'{int((a & d["y"]).sum())} | {(a & d["y"]).sum() / max(a.sum(), 1):.3f} |', flush=True)


def gaps(on: str, rows: dict) -> None:
    """Поток тревог сразу после того, как витрина снова появляется за провалом журнала.

    Окно отсчитывается не от конца дыры в журнале, а от первого часа, который снова есть в витрине:
    вокруг потери данных строки выбрасываются шире самой дыры (раздел 39).
    """
    year = YEARS[on]
    ends = [b for a, b in config.GAPS if a.year == year]
    if not ends:
        return
    h = next(iter(rows.values()))['h']
    starts = []
    for b in ends:
        after = h[h >= int((b - ft.T0).total_seconds() // 3600)]
        if len(after):
            starts.append(int(after.min()))
    if not starts:
        return
    back = [(ft.T0 + timedelta(hours=s)).isoformat(' ') for s in starts]
    print(f'\n**{on}**: первые часы после провала журнала (витрина возобновляется '
          f'{", ".join(back)})\n')
    print('| тип | за период: доля часов / Precision | первые сутки | первые трое суток |')
    print('|---|---:|---:|---:|')
    for tp, d in rows.items():
        alarm = d['p'] >= np.quantile(d['p'], 1 - SHARES[tp])
        out = [f'{alarm.mean():.3f} / {(alarm & d["y"]).sum() / max(alarm.sum(), 1):.3f}']
        for w in (24, 72):
            m = np.zeros(len(d['h']), bool)
            for s in starts:
                m |= (d['h'] >= s) & (d['h'] < s + w)
            a = alarm & m
            out.append(f'{a.sum() / max(m.sum(), 1):.3f} / {(a & d["y"]).sum() / max(a.sum(), 1):.3f}')
        print(f'| {config.TYPE_NAMES[tp]} | ' + ' | '.join(out) + ' |', flush=True)


def main() -> None:
    idx, quiet = since_quiet()
    share = (quiet > 0).mean()
    print(f'## Тишина в журнале\n\nЧасов без строки по объекту — {share:.1%} всех объекто-часов '
          f'парка (всего {quiet.size} объекто-часов).')
    for on in ('val', 'test'):
        rows = {tp: load(tp, on) for tp in config.TYPES}
        table(on, rows, quiet, idx)
        gaps(on, rows)


if __name__ == '__main__':
    main()
