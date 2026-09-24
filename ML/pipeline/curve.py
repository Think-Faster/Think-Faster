"""Чем оплачивается полнота: доля часов под тревогой и часы ложной тревоги (раздел 31).

Зачем понадобилось. Во всех таблицах сравнения ложные считаются **сигналами** — подряд идущие
часы тревоги на одном объекте это один сигнал, а не десять. Мера правильная: диспетчер ходит на
сигнал, а не на час. Но у неё есть слепое пятно. Если опускать порог, блоки тревоги начинают
склеиваться между собой, и число сигналов, дойдя до максимума, **падает**, хотя тревога висит всё
дольше. У отказа оборудования на проверке 2025: 569 ложных при 10% часов под тревогой, 696 при
20%, 244 при 50% — и в последней точке средний сигнал длится 651 час, почти месяц.

Где проходит граница, заранее не известно. Снятые кривые по шести типам на двух периодах дают
разброс от 3% часов (пожар, тест) до 30% (датчик), причём у одного и того же типа граница между
проверкой и тестом уезжает втрое. Значит, запомнить одно безопасное число нельзя: его надо
смотреть на тех же данных, на которых идёт сравнение. Столбец «счёт растёт» помечает строки, в
которых число сигналов ещё увеличивается с ростом доли часов; ниже первой пометки `—` читать
счёт сигналов как меру качества нельзя.

Выход из слепого пятна — **часы ложной тревоги**. Тревога это множество часов выше порога;
опуская порог, часы можно только добавить, поэтому часы ложной тревоги монотонны по порогу
**по построению**, а не по счастливому совпадению. Их можно сравнивать в любой точке. Счёт
сигналов остаётся эксплуатационной ценой (диспетчер ходит на сигнал, а не на час) и остаётся в
таблице, но как цена, а не как мера для выбора.

    python curve.py --on val --type equipment
    python curve.py --on test --type fire
"""
import argparse

import numpy as np

import config
import metrics
import operating

# Точки перебора: сгущаются к малым долям часов, потому что рабочая область именно там.
SHARES = [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]


def curve(run: str, on: str, tp: str, model: str, H: int) -> None:
    year = operating.years(run, on)
    obj, h, nxt, p = operating.load_mix(run, on, year, tp, model, '')
    y = (nxt <= H).astype(np.int8)

    print(f'{config.TYPE_NAMES[tp]}, {on}, доля положительных часов {y.mean():.2%}')
    print('| доля часов под тревогой | полнота эпизодов | ложных сигналов | счёт растёт | '
          'средняя длина сигнала, ч | часов ложной тревоги |')
    print('|---:|---:|---:|:--:|---:|---:|')
    rows, prev = [], -1
    for q in SHARES:
        t = float(np.quantile(p, q))
        a = p >= t
        if not a.any():
            continue
        sig, true = metrics.signals(obj, h, y, a)
        m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
        hours, false_sig = int(a.sum()), sig - true
        # Часы ложной тревоги — точный счёт, а не оценка через среднюю длину: это часы тревоги,
        # в горизонте которых происшествия нет. Величина монотонна по порогу по построению и
        # потому сравнима в любой точке кривой, в отличие от числа блоков.
        false_h = int((a & (y == 0)).sum())
        rows.append((a.mean(), m['caught'], m['episodes'], false_sig,
                     hours / max(sig, 1), false_h))
    # Идём от малых долей к большим: счёт сигналов осмыслен, пока он ещё растёт.
    grows = []
    for share, _, _, false_sig, _, _ in reversed(rows):
        grows.append(false_sig >= prev)  # равный счёт — ещё не падение
        prev = max(prev, false_sig)
    grows.reverse()
    for (share, caught, eps, false_sig, dur, false_h), ok in zip(rows, grows):
        print(f'| {share:.1%} | {caught / eps:.1%} ({caught} из {eps}) | {false_sig} | '
              f'{"да" if ok else "—"} | {dur:.0f} | {false_h} |')


def envelope(run: str, on: str, tp: str, model: str, H: int, steps: int) -> tuple[
        list[tuple[int, int]], list[tuple[int, int]], list[tuple[float, int, int]]]:
    """Пары (часы ложной тревоги, поймано эпизодов): вогнутая оболочка кривой типа.

    Часы монотонны по порогу, поэтому сама кривая строится перебором квантилей. Дальше берутся
    две оболочки подряд, и вторая важнее первой. Первая выбрасывает точки, которые платят больше
    за меньшее. Вторая — **вогнутая** — оставляет только те, у которых отдача на час падает с
    ростом цены; без неё жадная раздача в `split` не была бы оптимальной, а лишь похожей на
    оптимальную. Точки под оболочкой не теряются: до любой из них можно дойти смесью двух
    соседних точек оболочки, а на нашем масштабе часов округление до целого часа не заметно.

    Третьим возвращается сырая сетка `(доля часов под тревогой, часы, эпизоды)` — она нужна
    `split` для базы «одно правило на всех», где доля у всех типов одна и та же.
    """
    year = operating.years(run, on)
    obj, h, nxt, p = operating.load_mix(run, on, year, tp, model, '')
    y = (nxt <= H).astype(np.int8)
    pts = [(0, 0)]
    grid: list[tuple[float, int, int]] = []
    for q in np.linspace(0.999, 0.5, steps):
        t = float(np.quantile(p, q))
        a = p >= t
        if not a.any():
            continue
        m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
        false_h = int((a & (y == 0)).sum())
        pts.append((false_h, m['caught']))
        grid.append((round(1 - float(q), 6), false_h, m['caught']))
    pts.sort()
    rise: list[tuple[int, int]] = []
    best = -1
    for cost, caught in pts:
        if caught <= best:
            continue
        # На одной и той же цене оставляем только лучшую точку, иначе в раздаче появится
        # бесплатный шаг и жадный перебор встанет на нём.
        if rise and rise[-1][0] == cost:
            rise.pop()
        rise.append((cost, caught))
        best = caught
    out: list[tuple[int, int]] = []
    for pt in rise:
        # Держим отдачу невозрастающей: если новая точка делает предыдущую невыгодной
        # (до неё дешевле дойти напрямую), предыдущая снимается.
        while len(out) >= 2:
            (c1, k1), (c2, k2) = out[-2], out[-1]
            if (k2 - k1) * (pt[0] - c1) <= (pt[1] - k1) * (c2 - c1):
                out.pop()
            else:
                break
        out.append(pt)
    return out, rise, grid


def split(run: str, on: str, model: str, H: int, steps: int, budgets: list[int],
          floors: dict[str, float] | None = None, target: str = '', gap: int = 6) -> None:
    """Как делить общий бюджет ложной тревоги между шестью типами.

    В отчёте бюджет всегда задавался «тревог в сутки» и раздавался типам по одному правилу на
    всех. Вопрос, который при этом не задавался: не выгоднее ли отдать часы тому типу, где они
    дешевле покупают эпизоды. Считается в часах, а не в сигналах, потому что делить можно только
    монотонную величину (раздел 31).

    Жадная раздача здесь точна, а не приблизительна: цель — сумма пойманных эпизодов, ограничение
    одно, а каждая кривая приведена к вогнутой оболочке (см. `envelope`), так что шаг с лучшей
    отдачей на час и есть оптимальный.

    `floors` — обязательная полнота, которую каждый тип получает первым делом, до раздачи по
    отдаче: общее значение под пустым ключом и, при надобности, своё для отдельных типов. Без неё сумма эпизодов максимизируется буквально: редкий и дорогой тип получает ноль
    часов и выключается совсем. По ТЗ так нельзя — типы не взаимозаменяемы, проникновение нельзя
    выключить ради двух сотен отказов оборудования, — поэтому раздача без минимума читается как
    верхняя граница выигрыша, а не как предлагаемая настройка.

    Баз сравнения две, и слабую нельзя брать за единственную. «Поровну часов» никто никогда не
    предлагал — это удобная, но соломенная база. Настоящая — «одно правило на всех»: одинаковая
    доля часов под тревогой у каждого типа, ровно так бюджет и делился во всех прежних разделах.
    Выигрыш считается к лучшей из двух.

    Обязательный минимум выплачивается до проверки бюджета, поэтому на малых бюджетах он может
    в них не поместиться. Такие строки помечаются, и выигрыш по ним не считается: сравнивать
    раздачу, вышедшую за бюджет, с базами внутри бюджета нельзя.

    `target` — период, на котором раздача **оценивается**, если он другой. Выбранной считается
    доля часов под тревогой у каждого типа: её и переносим, как в разделе 32. Без переноса
    раздача подобрана на том же периоде, где мерится, и выигрыш читается как верхняя граница.

    Под раздачей на последнем бюджете печатается сводка для диспетчера: сигналы после склейки
    дребезга `gap` (раздел 27), ложные сигналы в сутки на весь парк и упреждение. Часы — мера
    выбора, сигналы — то, на что ходит диспетчер; обе нужны итоговой конфигурации (раздел 38).

    Разная полнота по типам нужна потому, что ТЗ их не уравнивает: критическим там названо
    «событие, угрожающее жизни человека», и уведомления в реальном времени требуются именно о
    таких. Пожар и загазованность нельзя держать на том же минимуме, что отказ датчика.
    """
    env, full, grid = {}, {}, {}
    for tp in config.TYPES:
        try:
            env[tp], full[tp], grid[tp] = envelope(run, on, tp, model, H, steps)
        except FileNotFoundError:
            continue
    # Где мерить: по умолчанию там же, где выбирали; с `target` — на другом периоде по тем же долям.
    score_grid = grid
    if target:
        score_grid = {tp: envelope(run, target, tp, model, H, steps)[2] for tp in env}
    at = {tp: {sh: (c, k) for sh, c, k in g} for tp, g in score_grid.items()}
    share_of = {tp: {(c, k): sh for sh, c, k in reversed(g)} for tp, g in grid.items()}

    def score(pick: dict[str, float | None]) -> tuple[int, int]:
        """Выбранные доли по типам -> (часы ложной тревоги, поймано) на периоде оценки."""
        cost = caught = 0
        for tp, sh in pick.items():
            c, k = at[tp].get(sh, (0, 0)) if sh is not None else (0, 0)
            cost, caught = cost + c, caught + k
        return cost, caught

    # Одно правило на всех: суммируем типы на общей доле часов под тревогой. Доли, до которых
    # дошли не все типы, не берём — иначе в сумме окажется разное число типов.
    by_share: dict[float, tuple[int, int]] = {}
    if grid:
        common = set.intersection(*[{sh for sh, _, _ in g} for g in grid.values()])
        for sh in sorted(common):
            pick = [next((c, k) for s2, c, k in g if s2 == sh) for g in grid.values()]
            by_share[sh] = (sum(c for c, _ in pick), sum(k for _, k in pick))

    floors = floors or {}
    base = floors.get('', 0.0)
    need_of = {tp: floors.get(tp, base) for tp in env}
    text = f'{base:.0%}'
    if len(set(need_of.values())) > 1:
        text += ''.join(f', {config.TYPE_NAMES[tp]} {v:.0%}'
                        for tp, v in need_of.items() if v != base)
    where = f'выбор на {on}, оценка на {target}' if target else on
    print(f'{where}, {run}: раздача бюджета ложной тревоги между типами, '
          f'обязательная полнота {text}')
    if target:
        print('В клетках — «поймано / часов ложной тревоги» на периоде оценки: бюджет '
              'соблюдался при выборе, а не при оценке.')
        print('| бюджет часов | одно правило на всех | поровну часов | по отдаче | '
              'выигрыш к лучшей базе |')
        print('|---:|---:|---:|---:|---:|')
    else:
        print('| бюджет часов | одно правило на всех | поровну часов | по отдаче | '
              'выигрыш к лучшей базе | израсходовано часов |')
        print('|---:|---:|---:|---:|---:|---:|')
    for b in budgets:
        # База считается по полной кривой, а не по вогнутой оболочке: оболочка выбрасывает
        # точки, и сравнение поехало бы в пользу раздачи по отдаче.
        pick_even: dict[str, float | None] = {}
        for tp, e in full.items():
            ok = [(k, c) for c, k in e if c <= b / len(full)]
            k, c = max(ok) if ok else (0, 0)
            pick_even[tp] = share_of[tp].get((c, k))
        fit = [(k, sh) for sh, (c, k) in by_share.items() if c <= b]
        sh_uni = max(fit)[1] if fit else None
        pick_uni = {tp: sh_uni for tp in env}
        # Жадная раздача: каждый следующий шаг отдаём типу с лучшей отдачей на час.
        idx = {tp: 0 for tp in env}
        spent = 0
        # Сначала обязательный минимум каждому типу — дешевейшая точка, дающая нужную полноту.
        for tp, e in env.items():
            need = need_of[tp] * e[-1][1]
            for i, (cost, caught) in enumerate(e):
                if caught >= need:
                    idx[tp], spent = i, spent + cost
                    break
        while True:
            best, gain = None, 0.0
            for tp, e in env.items():
                i = idx[tp]
                if i + 1 >= len(e):
                    continue
                dc = e[i + 1][0] - e[i][0]
                dk = e[i + 1][1] - e[i][1]
                if dc <= 0 or spent + dc > b:
                    continue
                if dk / dc > gain:
                    best, gain = tp, dk / dc
            if best is None:
                break
            spent += env[best][idx[best] + 1][0] - env[best][idx[best]][0]
            idx[best] += 1
        pick_greedy = {tp: share_of[tp].get(tuple(env[tp][idx[tp]])) for tp in env}
        (cu, ku), (ce, ke), (cg, kg) = score(pick_uni), score(pick_even), score(pick_greedy)
        gain = f'{kg - max(ku, ke):+d}' if spent <= b else '— (минимум не помещается)'
        if target:
            print(f'| {b} | {ku} / {cu} | {ke} / {ce} | {kg} / {cg} | {gain} |')
        else:
            print(f'| {b} | {ku} | {ke} | {kg} | {gain} | {spent} |')
    print()
    print(f'Раздача по отдаче на последнем бюджете ({budgets[-1]} часов)'
          + (f', измерено на {target}' if target else '') + ':')
    print('| тип | доля часов под тревогой | часов ложной тревоги | поймано |')
    print('|---|---:|---:|---:|')
    for tp in env:
        sh = pick_greedy[tp]
        c, k = at[tp].get(sh, (0, 0)) if sh is not None else (0, 0)
        print(f'| {config.TYPE_NAMES[tp]} | {0 if sh is None else sh:.1%} | {c} | {k} |')

    def durations(obj, h, alarm):
        """Сколько часов длится каждый сигнал после склейки: от первого часа тревоги до последнего."""
        if not alarm.any():
            return np.zeros(0)
        o, hh = obj[alarm], h[alarm]
        order = np.lexsort((hh, o))
        o, hh = o[order], hh[order]
        start = np.r_[True, (o[1:] != o[:-1]) | (hh[1:] - hh[:-1] > gap + 1)]
        first = hh[start]
        last = np.maximum.reduceat(hh, np.flatnonzero(start))
        return (last - first + 1).astype(np.float64)

    def watchlist(tp, obj, h, share):
        """Правило без модели: тревога всё время горит у объектов с худшей историей на проверке.

        Объекты ранжируются по доле часов «инцидент в горизонте» на val, тревога включается
        целиком на верхних, пока не набрана та же доля часов. Если модель на большом бюджете ловит
        не больше этого правила, она работает как список проблемных объектов, а не как прогноз.
        """
        ov, _, nv, _ = operating.load_mix(run, 'val', operating.years(run, 'val'), tp, model, '')
        size = max(int(ov.max()), int(obj.max())) + 1
        cnt = np.bincount(ov, minlength=size)
        pos = np.bincount(ov, weights=(nv <= H), minlength=size)
        rate = (pos + 0.5) / (cnt + 50)          # объект без истории тянется к среднему
        # равенство внутри объекта — сплошным блоком по времени. Случайный разрыв рассыпал бы
        # тревогу по часам объекта, а россыпь в треть часов накрывает почти каждое окно 24 ч:
        # правило «поймало» бы 39 проникновений из 41 на одном объекте, ничего не зная о времени
        score = rate[obj] + 1e-12 * (h - h.min())
        return score, float(np.quantile(score, 1 - share))

    where = target or on
    year = operating.years(run, where)
    print()
    print(f'Для диспетчера, та же раздача на {where}, склейка дребезга {gap} ч:')
    print('| тип | эпизодов | поймано заранее | остаётся каналу «по факту» | сигналов | '
          'ложных сигналов | ложных в сутки | медиана упреждения, ч | сигнал длится, ч: медиана / 90% | '
          'пойманы тревогой, горевшей ≥ 7 сут | правило «худшие объекты»: поймано |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    tot = [0] * 5
    all_dur = []
    stand = wl = 0
    days = 0.0
    for tp in env:
        obj, h, nxt, p = operating.load_mix(run, where, year, tp, model, '')
        days = max(days, (float(h.max()) - float(h.min()) + 1) / 24)
        sh = pick_greedy[tp]
        t = float(np.quantile(p, 1 - sh)) if sh else float('inf')
        m = metrics.evaluate(obj, h, nxt, p, t, H, metrics.RUN_CAP)
        y = (nxt <= H).astype(np.int8)
        sig, sig_true = metrics.signals(obj, h, y, p >= t, gap)
        row = [m['episodes'], m['caught'], m['episodes'] - m['caught'], sig, sig - sig_true]
        tot = [a + b for a, b in zip(tot, row)]
        lead = m['lead_median_h']
        d = durations(obj, h, p >= t)
        all_dur.append(d)
        dur = f'{np.median(d):.0f} / {np.quantile(d, 0.9):.0f}' if len(d) else '—'
        # alarm_run_capped — доля пойманных, у которых тревога к началу эпизода горела RUN_CAP часов
        st = round(m['caught'] * m['alarm_run_capped']) if m['caught'] else 0
        ws, wt = watchlist(tp, obj, h, sh) if sh else (p, float('inf'))
        wc = metrics.evaluate(obj, h, nxt, ws, wt, H, metrics.RUN_CAP)['caught']
        stand, wl = stand + st, wl + wc
        print(f'| {config.TYPE_NAMES[tp]} | ' + ' | '.join(map(str, row))
              + f' | {row[4] / days:.1f} | {"—" if lead != lead else f"{lead:.0f}"} | {dur} | {st} | {wc} |')
    d = np.concatenate(all_dur)
    dur = f'{np.median(d):.0f} / {np.quantile(d, 0.9):.0f}' if len(d) else '—'
    print(f'| **итого** | ' + ' | '.join(map(str, tot)) + f' | {tot[4] / days:.1f} | | {dur} | {stand} | {wl} |')

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', default='main_h24_tuned_r100e20')
    ap.add_argument('--on', default='val', choices=['val', 'test'])
    ap.add_argument('--type', dest='tp', default='', help='тип происшествия; пусто — все шесть')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--mode', default='curve', choices=['curve', 'split'])
    ap.add_argument('--steps', type=int, default=120, help='точек на кривую для --mode split')
    ap.add_argument('--budgets', default='5000,10000,20000,40000,80000',
                    help='общие бюджеты часов ложной тревоги через запятую')
    ap.add_argument('--floor', default='0',
                    help='обязательная полнота до раздачи по отдаче: общее число '
                         'либо «0.4,fire=0.6,gas=0.6»')
    ap.add_argument('--target', default='', choices=['', 'val', 'test'],
                    help='период оценки раздачи, выбранной на --on; пусто — тот же')
    ap.add_argument('--gap', type=int, default=6, help='склейка дребезга для сводки сигналов')
    args = ap.parse_args()

    if args.mode == 'split':
        floors: dict[str, float] = {}
        for part in args.floor.split(','):
            key, _, val = part.rpartition('=')
            floors[key.strip()] = float(val)
        split(args.run, args.on, args.model, args.horizon, args.steps,
              [int(x) for x in args.budgets.split(',')], floors, args.target, args.gap)
        return

    types = [args.tp] if args.tp else list(config.TYPES)
    for tp in types:
        curve(args.run, args.on, tp, args.model, args.horizon)
        print()


if __name__ == '__main__':
    main()
