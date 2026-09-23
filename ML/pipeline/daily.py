"""Рабочая точка под суточный режим: тот же бюджет, но мера — один прогноз в 07:00.

Доли часов подбирались по ложным **часам** за все сутки (разделы 38, 42, 46), а диспетчер получает
сводку раз в смену. В ретропрогоне видно, что это разные меры: длинная тревога в часах стоит много,
а в снимке 07:00 считается один раз (раздел 49). Здесь проверяется, не оставляет ли часовая
раздача поимок на столе в суточной мере.

Мера. Берутся часы прогноза, попадающие на 07:00 (`h % 24 == 6`, час индексируется от `features.T0`,
момент прогноза — конец часа). Объекто-сутки с эпизодом в ближайшие 24 ч — это эпизоды; тревога в
этот час — сигнал диспетчеру. Отсюда тревог в сутки на парк, точность, поймано и полнота — ровно то,
что печатает `retro.py`.

Раздача. По сетке долей для каждого типа снимается кривая «тревог в сутки → поймано», приводится к
вогнутой оболочке и раздаётся жадно под общий бюджет тревог в сутки — так же, как `curve.split`
делит ложные часы. Без ограничений редкий тип выключается совсем, поэтому доля типа не опускается
ниже половины нынешней (`FLOOR`). Выбор — на проверке 2025, оценка — на тесте 2026 по перенесённым
долям, как в разделе 32.

Порог здесь берётся по всему периоду (как в `smooth.py` и `budget.py`), а не скользящим окном, как
в эксплуатации и в `retro.py`: сравниваются рабочие точки между собой, и обе считаются одинаково.

Третий вопрос — сама форма порога. В эксплуатации это доля часов за период, но сводка раз в смену
допускает наряд: «каждые сутки берём k объектов с наибольшей оценкой». Наряд не зависит от того,
куда уехал общий уровень оценок, но и не умеет промолчать в спокойные сутки. Сравнение — при
равной нагрузке, у типов, где тревог не меньше одной в сутки.

    python daily.py "$MIXT"
"""
import sys

import numpy as np

import config
from smooth import load
from reject import simulate

HOUR = 6                      # h % 24 == 6 -> момент прогноза 07:00
FLOOR = 0.5                   # доля типа не ниже половины нынешней
SHARES = np.unique(np.round(np.geomspace(0.002, 0.2, 45), 6))
OP = config.operating()['types']


def alarms(d: dict, share: float, k) -> np.ndarray:
    """Часовая тревога при доле `share` с правилом отклонения типа (k = None — правило выключено)."""
    if k is None:
        return d['p'] >= np.quantile(d['p'], 1 - share)
    by = {}
    for a, b in d['eps']:
        by.setdefault(a, []).append(b)
    return simulate(d, {a: np.array(sorted(b), np.int64) for a, b in by.items()}, share, 1, 10 ** 6, k)


def daily(d: dict, alarm: np.ndarray) -> dict:
    """Метрика одного прогноза в сутки: тревог в сутки на парк, точность, эпизоды, поймано."""
    sel = d['h'] % 24 == HOUR
    a, y = alarm[sel], d['y'][sel]
    days = len(np.unique(d['h'][sel]))
    return dict(days=days, per_day=a.sum() / days, hit=int((a & y).sum()), alarm=int(a.sum()),
                eps=int(y.sum()), prec=float((a & y).sum() / max(a.sum(), 1)),
                recall=float((a & y).sum() / max(y.sum(), 1)))


def curve(d: dict, k) -> list[tuple[float, float, int]]:
    """(доля, тревог в сутки, поймано) по сетке долей."""
    out = []
    for s in SHARES:
        m = daily(d, alarms(d, float(s), k))
        out.append((float(s), m['per_day'], m['hit']))
    return out


def hull(pts: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Вогнутая оболочка кривой: только точки, где отдача на тревогу ещё не выросла."""
    pts = sorted(pts, key=lambda x: (x[1], x[2]))
    out: list[tuple[float, float, int]] = []
    for p in pts:
        while out and p[2] <= out[-1][2]:
            out.pop()
        while len(out) >= 2 and ((p[2] - out[-2][2]) * (out[-1][1] - out[-2][1])
                                 >= (out[-1][2] - out[-2][2]) * (p[1] - out[-2][1])):
            out.pop()
        out.append(p)
    return out


def greedy(curves: dict, budget: float, floors: dict) -> dict:
    """Жадная раздача бюджета тревог в сутки: шаг отдаём типу с лучшей отдачей на тревогу."""
    pick = {tp: min([p for p in c if p[0] >= floors[tp]], key=lambda p: p[1]) for tp, c in curves.items()}
    spent = sum(p[1] for p in pick.values())
    idx = {tp: [p[0] for p in curves[tp]].index(pick[tp][0]) for tp in curves}
    while True:
        best, gain = None, 0.0
        for tp, c in curves.items():
            i = idx[tp] + 1
            if i >= len(c):
                continue
            dc, dk = c[i][1] - pick[tp][1], c[i][2] - pick[tp][2]
            if dc <= 0 or spent + dc > budget:
                continue
            if dk / dc > gain:
                best, gain = tp, dk / dc
        if best is None:
            return {tp: p[0] for tp, p in pick.items()}
        idx[best] += 1
        spent += curves[best][idx[best]][1] - pick[best][1]
        pick[best] = curves[best][idx[best]]


def watch(d: dict, k: int) -> np.ndarray:
    """Суточный наряд: в каждые сутки тревожатся k объектов парка с наибольшей оценкой."""
    alarm = np.zeros(len(d['p']), bool)
    sel = np.flatnonzero(d['h'] % 24 == HOUR)
    order = np.lexsort((-d['p'][sel], d['h'][sel]))
    sel = sel[order]
    h = d['h'][sel]
    start = np.flatnonzero(np.r_[True, h[1:] != h[:-1]])
    rank = np.arange(len(sel)) - start[np.cumsum(np.r_[True, h[1:] != h[:-1]]) - 1]
    alarm[sel[rank < k]] = True
    return alarm


def table(name: str, on: str, data: dict, shares: dict) -> tuple[float, int]:
    rows, tot = [], np.zeros(4)
    for tp in config.TYPES:
        m = daily(data[tp], alarms(data[tp], shares[tp], OP[tp]['reject_k']))
        rows.append((tp, shares[tp], m))
        tot += [m['per_day'], m['alarm'], m['hit'], m['eps']]
    print(f'\n**{name}**, {on}, суток {rows[0][2]["days"]}\n')
    print('| тип | доля часов | тревог в сутки | Precision | эпизодов | поймано | Recall |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for tp, s, m in rows:
        print(f'| {config.TYPE_NAMES[tp]} | {s:.3f} | {m["per_day"]:.2f} | {m["prec"]:.3f} | '
              f'{m["eps"]} | {m["hit"]} | {m["recall"]:.3f} |')
    print(f'| **всего** | | **{tot[0]:.2f}** | **{tot[2] / max(tot[1], 1):.3f}** | {int(tot[3])} | '
          f'**{int(tot[2])}** | {tot[2] / max(tot[3], 1):.3f} |', flush=True)
    return float(tot[0]), int(tot[2])


def main() -> None:
    data = {on: {tp: load(tp, on) for tp in config.TYPES} for on in ('val', 'test')}
    now = {tp: OP[tp]['share'] for tp in config.TYPES}
    print('## Суточная мера: нынешняя рабочая точка')
    budget, _ = table('доли настроек (раздел 46)', 'val', data['val'], now)
    table('доли настроек (раздел 46)', 'test', data['test'], now)

    print('\n## Раздача того же бюджета тревог в сутки под суточную меру')
    curves = {}
    for tp in config.TYPES:
        curves[tp] = hull(curve(data['val'][tp], OP[tp]['reject_k']))
        print(f'{tp}: точек {len(curves[tp])}', file=sys.stderr, flush=True)
    floors = {tp: min(s for s, _, _ in curves[tp] if s >= now[tp] * FLOOR) for tp in config.TYPES}
    pick = greedy(curves, budget, floors)
    print(f'\nБюджет — {budget:.2f} тревоги в сутки (нынешняя точка на проверке), '
          f'доля типа не ниже {FLOOR:.0%} нынешней.\n')
    print('| тип | доля сейчас | доля под сутки |')
    print('|---|---:|---:|')
    for tp in config.TYPES:
        print(f'| {config.TYPE_NAMES[tp]} | {now[tp]:.3f} | {pick[tp]:.3f} |')
    table('раздача под суточную меру', 'val', data['val'], pick)
    table('раздача под суточную меру', 'test', data['test'], pick)

    print('\n## Отброшенные варианты раздачи (проверка / тест, поймано)\n')
    print('| вариант | тревог в сутки | поймано на проверке | поймано на тесте |')
    print('|---|---:|---:|---:|')
    var = {'одна доля на все типы': None, 'без нижней границы доли': {tp: 0.0 for tp in config.TYPES}}
    for name, fl in var.items():
        if fl is None:
            lo, hi = 0.002, 0.2
            for _ in range(30):
                mid = (lo + hi) / 2
                cost = sum(daily(data['val'][tp], alarms(data['val'][tp], mid, OP[tp]['reject_k']))['per_day']
                           for tp in config.TYPES)
                lo, hi = (mid, hi) if cost < budget else (lo, mid)
            sh = {tp: lo for tp in config.TYPES}
        else:
            sh = greedy(curves, budget, {tp: min(s2 for s2, _, _ in curves[tp]) for tp in config.TYPES})
        got = []
        for on in ('val', 'test'):
            m = [daily(data[on][tp], alarms(data[on][tp], sh[tp], OP[tp]['reject_k'])) for tp in config.TYPES]
            got.append((sum(x['per_day'] for x in m), sum(x['hit'] for x in m)))
        print(f'| {name} | {got[0][0]:.2f} | {got[0][1]} | {got[1][1]} |')
        print(f'   доли: ' + ', '.join(f'{tp} {sh[tp]:.3f}' for tp in config.TYPES), file=sys.stderr, flush=True)

    print('\n## Суточный наряд против порога за период\n')
    print('Наряд: k объектов в сутки. Порог: доля часов, подобранная под ту же нагрузку '
          '(интерполяция по кривой). Типы, где тревог меньше одной в сутки, наряду не подходят.\n')
    print('| тип | k | наряд: проверка / тест | порог при той же нагрузке: проверка / тест |')
    print('|---|---:|---:|---:|')
    for tp in config.TYPES:
        base = daily(data['val'][tp], alarms(data['val'][tp], now[tp], OP[tp]['reject_k']))['per_day']
        k = int(round(base))
        if k < 1:
            continue
        w = [daily(data[on][tp], watch(data[on][tp], k)) for on in ('val', 'test')]
        thr = []
        for on, m in zip(('val', 'test'), w):
            c = sorted(curve(data[on][tp], OP[tp]['reject_k']), key=lambda x: x[1])
            thr.append(np.interp(m['per_day'], [x[1] for x in c], [x[2] for x in c]))
        print(f'| {config.TYPE_NAMES[tp]} | {k} | {w[0]["hit"]} / {w[1]["hit"]} | '
              f'{thr[0]:.0f} / {thr[1]:.0f} |', flush=True)


if __name__ == '__main__':
    main()
