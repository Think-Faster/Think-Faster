"""Цена опоздания данных: во сколько поимок обходится час несвежести признаков.

Все числа исследования получены так, будто в момент прогноза известен весь журнал до конца
последнего часа. В проде так не бывает: события идут через воронку и Kafka, часть приходит с
задержкой, потребитель может отстать, а сборка витрины занимает время. Вопрос, на который здесь
отвечаем: сколько стоит каждый час несвежести — и какое опоздание ещё можно терпеть.

Модель опоздания — пессимистичная и простая: в момент решения на часе `t` доступны признаки,
собранные на часе `t − Δ` (все события последних Δ часов ещё не дошли). Это верхняя оценка вреда:
реальная задержка задевает не весь поток, а часть каналов. Тревога при этом считается на час `t` и
проверяется по эпизодам ближайших 24 ч от `t`, то есть упреждение честно уменьшается на Δ.

Порог, правило отклонения и доли — из настроек; при опоздании порог берётся по тем же опоздавшим
оценкам (в эксплуатации скользящий квантиль считается по тому же потоку).

Меры две: часовая (как в разделах 34, 46) и суточная — снимок 07:00 (раздел 49). Для суточной
задержка особенно наглядна: при Δ = 3 сводка в 07:00 опирается на данные до 04:00.

    python lag.py "$MIXT"
"""
import numpy as np

import config
from daily import alarms, daily
from smooth import load, measure, shifted

LAGS = (0, 1, 2, 3, 6, 12)
OP = config.operating()['types']


def lagged(d: dict, j: int) -> dict:
    """Копия данных, где оценка часа t — это оценка, посчитанная на часе t − j того же объекта."""
    if j == 0:
        return d
    q = shifted(d['p'], d['o'], j)
    return {**d, 'p': np.where(np.isnan(q), -1.0, q)}


def main() -> None:
    hourly, day = {}, {}
    for tp in config.TYPES:
        s, k = OP[tp]['share'], OP[tp]['reject_k']
        for on in ('val', 'test'):
            d = load(tp, on)
            for j in LAGS:
                a = alarms(lagged(d, j), s, k)
                hourly[tp, on, j] = measure(d, a)
                day[tp, on, j] = daily(d, a)
            print(f'{tp} {on} готово', flush=True)

    print()
    print('## Цена опоздания данных')
    print()
    print(f'Доли и правило отклонения — из настроек. Опоздание Δ: в момент решения известны признаки '
          f'часа t − Δ. В клетке часовой меры: поймано / свежих / ложных сигналов.')
    print()
    print('### Часовая мера')
    print()
    print('| тип | период | ' + ' | '.join(f'Δ {j} ч' if j else 'без опоздания' for j in LAGS) + ' |')
    print('|---|---|' + '---|' * len(LAGS))
    for tp in config.TYPES:
        for on in ('val', 'test'):
            cells = [f'{hourly[tp, on, j]["caught"]} / {hourly[tp, on, j]["fresh"]} / '
                     f'{hourly[tp, on, j]["false_sig"]}' for j in LAGS]
            print(f'| {config.TYPE_NAMES[tp]} | {on} | ' + ' | '.join(cells) + ' |')
    for on in ('val', 'test'):
        cells = []
        for j in LAGS:
            c = sum(hourly[tp, on, j]['caught'] for tp in config.TYPES)
            f = sum(hourly[tp, on, j]['fresh'] for tp in config.TYPES)
            fs = sum(hourly[tp, on, j]['false_sig'] for tp in config.TYPES)
            cells.append(f'**{c} / {f} / {fs}**')
        print(f'| **все типы** | {on} | ' + ' | '.join(cells) + ' |')

    print()
    print('### Медиана упреждения, часов до начала эпизода')
    print()
    print('| тип | период | ' + ' | '.join(f'Δ {j} ч' if j else 'без опоздания' for j in LAGS) + ' |')
    print('|---|---|' + '---:|' * len(LAGS))
    for tp in config.TYPES:
        for on in ('val', 'test'):
            cells = [f'{hourly[tp, on, j]["lead"]:.0f}' for j in LAGS]
            print(f'| {config.TYPE_NAMES[tp]} | {on} | ' + ' | '.join(cells) + ' |')

    print()
    print('### Суточная мера: снимок 07:00')
    print()
    print('В клетке: тревог в сутки на парк / поймано объекто-суток.')
    print()
    print('| тип | период | ' + ' | '.join(f'Δ {j} ч' if j else 'без опоздания' for j in LAGS) + ' |')
    print('|---|---|' + '---|' * len(LAGS))
    for tp in config.TYPES:
        for on in ('val', 'test'):
            cells = [f'{day[tp, on, j]["per_day"]:.2f} / {day[tp, on, j]["hit"]}' for j in LAGS]
            print(f'| {config.TYPE_NAMES[tp]} | {on} | ' + ' | '.join(cells) + ' |')
    for on in ('val', 'test'):
        cells = []
        for j in LAGS:
            p = sum(day[tp, on, j]['per_day'] for tp in config.TYPES)
            h = sum(day[tp, on, j]['hit'] for tp in config.TYPES)
            cells.append(f'**{p:.2f} / {h}**')
        print(f'| **все типы** | {on} | ' + ' | '.join(cells) + ' |')

    print()
    print('### Потеря поимок к варианту без опоздания, %')
    print()
    print('| период | мера | ' + ' | '.join(f'Δ {j} ч' for j in LAGS[1:]) + ' |')
    print('|---|---|' + '---:|' * (len(LAGS) - 1))
    for on in ('val', 'test'):
        for name, src, key in (('часовая, поймано', hourly, 'caught'),
                               ('часовая, свежих', hourly, 'fresh'),
                               ('суточная, поймано', day, 'hit')):
            base = sum(src[tp, on, 0][key] for tp in config.TYPES)
            cells = []
            for j in LAGS[1:]:
                cur = sum(src[tp, on, j][key] for tp in config.TYPES)
                cells.append(f'{100 * (cur - base) / max(base, 1):+.1f}')
            print(f'| {on} | {name} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
