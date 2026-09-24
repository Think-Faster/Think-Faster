"""Затухающее снижение в суточном режиме: меняется ли вывод раздела 47, если мерить снимком 07:00.

Раздел 47 оставил затухание кандидатом из-за одного возражения: полнота при равных ложных часах
растёт на 1,2–1,9 пункта, но ложных сигналов становится на 40–60% больше — погашенная на сутки
тревога возвращается новым сигналом, и диспетчер отклоняет её ещё раз. Цена правила — лишние
уведомления.

Но уведомление стоит дорого только в часовом режиме. В суточном (раздел 49: один прогноз в 07:00,
объекто-сутки, а не часы) вернувшаяся тревога — это та же строка наряда на следующее утро, и
двойной счёт исчезает сам собой. Поэтому вопрос задаётся заново: при равной нагрузке на диспетчера
(тревог в сутки) даёт ли затухание больше пойманных объекто-суток, чем простое снижение доли.

Мера — из `daily.py`: часы `h % 24 == 6`, объекто-сутки с эпизодом в ближайшие 24 ч, порог по всему
периоду. Сравнение — при равных тревогах в сутки: кривая «без правила» снимается по сетке долей,
и поимки варианта сравниваются с поимками на этой кривой в той же точке по нагрузке. Правила и их
параметры — те же, что в разделе 47; выбор на проверке 2025, тест 2026 — контроль.

    python decayday.py "$MIXT"
"""
import numpy as np

import config
from daily import alarms, daily
from decay import simulate_decay
from reject import simulate
from smooth import load

K0 = (0.0, 0.2, 0.5)
TAU = (24, 72, 168)
F = np.geomspace(0.4, 3.0, 17)          # сетка долей для кривой без правила
FR = (1.0, 1.4, 2.0)                    # уровни доли, на которых считается правило
OP = config.operating()['types']


def ons_of(d: dict) -> dict:
    by = {}
    for a, b in d['eps']:
        by.setdefault(a, []).append(b)
    return {a: np.array(sorted(b), np.int64) for a, b in by.items()}


def base_curve(d: dict, share: float) -> np.ndarray:
    """(тревог в сутки, поймано объекто-суток) по сетке долей, правило отклонения выключено."""
    pts = []
    for f in F:
        m = daily(d, alarms(d, float(share * f), None))
        pts.append((m['per_day'], m['hit']))
    return np.array(sorted(pts))


def at(c: np.ndarray, x: float) -> float:
    return float(np.interp(x, c[:, 0], c[:, 1]))


def rules(d: dict, ons: dict, share: float) -> dict:
    """Тревоги всех правил на одной доле: затухание по сетке и ступенька раздела 44."""
    out = {'step_k0.2': simulate(d, ons, share, 1, 10 ** 6, 0.2)}
    for k0 in K0:
        for tau in TAU:
            out[f'decay_k{k0}_t{tau}'] = simulate_decay(d, ons, share, k0, tau)
    return out


def main() -> None:
    res, cur = {}, {}
    for tp in config.TYPES:
        for on in ('val', 'test'):
            d = load(tp, on)
            ons = ons_of(d)
            s = OP[tp]['share']
            c = base_curve(d, s)
            m0 = daily(d, alarms(d, s, OP[tp]['reject_k']))
            cur[f'{tp}|{on}'] = (m0['per_day'], m0['hit'], m0['prec'], m0['eps'])
            for f in FR:
                for name, a in rules(d, ons, s * f).items():
                    m = daily(d, a)
                    res[f'{tp}|{on}|{name}|{f}'] = (m['per_day'], m['hit'], at(c, m['per_day']), m['prec'])
            res[f'{tp}|{on}|curve'] = c.tolist()
            print(f'{tp} {on} готово', flush=True)

    print()
    print('## Затухание в суточном режиме')
    print()
    print('Нынешняя рабочая точка (снимок 07:00, порог за период):')
    print()
    print('| тип | период | тревог в сутки | поймано объекто-суток | из эпизодов | Precision |')
    print('|---|---|---:|---:|---:|---:|')
    for tp in config.TYPES:
        for on in ('val', 'test'):
            p, h, pr, e = cur[f'{tp}|{on}']
            print(f'| {config.TYPE_NAMES[tp]} | {on} | {p:.2f} | {h} | {e} | {pr:.3f} |')

    names = ['step_k0.2'] + [f'decay_k{k0}_t{t}' for k0 in K0 for t in TAU]
    for on in ('val', 'test'):
        print()
        print(f'### {on}: поймано сверх простой смены доли при равной нагрузке, сумма по шести типам')
        print()
        print('| правило | ' + ' | '.join(f'доля x{f:g}' for f in FR) + ' |')
        print('|---|' + '---:|' * len(FR))
        for name in names:
            cells = []
            for f in FR:
                g = sum(res[f'{tp}|{on}|{name}|{f}'][1] - res[f'{tp}|{on}|{name}|{f}'][2]
                        for tp in config.TYPES)
                cells.append(f'{g:+.0f}')
            print(f'| {name} | ' + ' | '.join(cells) + ' |')

    print()
    print('### Выбор по проверке 2025 и тот же выбор на тесте 2026')
    print()
    print('| тип | правило | доля | проверка: тревог/сут, поймано, сверх кривой | тест: то же |')
    print('|---|---|---|---|---|')
    tot = np.zeros(4)
    for tp in config.TYPES:
        best = max(((n, f) for n in names for f in FR),
                   key=lambda x: res[f'{tp}|val|{x[0]}|{x[1]}'][1] - res[f'{tp}|val|{x[0]}|{x[1]}'][2])
        cells = []
        for on in ('val', 'test'):
            p, h, r, pr = res[f'{tp}|{on}|{best[0]}|{best[1]}']
            cells.append(f'{p:.2f} / {h} / {h - r:+.0f}')
            if on == 'test':
                tot += [p, h, r, 0]
        print(f'| {config.TYPE_NAMES[tp]} | {best[0]} | x{best[1]:g} | ' + ' | '.join(cells) + ' |')
    print()
    print(f'Тест, все типы: {tot[0]:.2f} тревог в сутки, поймано {tot[1]:.0f}, '
          f'на кривой без правила при той же нагрузке {tot[2]:.0f}.')


if __name__ == '__main__':
    main()
