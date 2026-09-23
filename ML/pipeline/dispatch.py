"""Кому это разгребать: сколько карточек рабочая точка кладёт на стол и как они распределены.

Разделы 34–49 считали качество: поймано, ложные часы, тревог в сутки на парк. Интерфейсу этого
мало. Диспетчер видит не «0.36 тревоги в сутки на тип», а список карточек в окне: сколько их
одновременно, на сколько объектов они садятся, ложится ли нагрузка ровно на коллекторы (в ТЗ —
сегментный режим на 4–10 сегментов) и как часто на одном объекте одновременно висят два типа
(«Нахлёст»).

Считается на рабочей точке: доли и правило отклонения — из настроек, мера часовая (раздел 46).
Карточка = объект × тип × час тревоги. Объединение по объекту показано отдельно: это ответ на
вопрос, сколько карточек уберёт схлопывание нахлёста.

Коллектор берётся из справочника: объект уровня 3, его `родитель` — коллектор.

    python dispatch.py "$MIXT"
"""
import csv
import sys
from collections import Counter

import numpy as np

import config
from daily import HOUR, alarms
from smooth import load

OP = config.operating()['types']
SEG = 10          # столько карточек на экране — граница, после которой нужен сегментный режим (ТЗ)


def collectors() -> dict:
    """object_id -> collector_id по справочнику диспетчера."""
    path = config.DICT / 'справочник_объектов_диспетчер.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    return {int(r['ид_объект']): int(r['родитель']) for r in rows
            if r['иерархия_уровень'] == '3' and r['родитель']}


def q(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p)) if len(x) else 0.0


def per_hour(cards: dict, hours: np.ndarray) -> np.ndarray:
    """Сколько карточек в каждом часе периода (включая пустые часы)."""
    cnt = Counter(h for _, h, _ in cards['list'])
    return np.array([cnt.get(int(h), 0) for h in hours], dtype=np.int64)


def runs(d: dict, a: np.ndarray) -> list:
    """Длины непрерывных тревог одного объекта: данные отсортированы по (объект, час)."""
    i = np.flatnonzero(a)
    if not len(i):
        return []
    brk = np.r_[True, (d['o'][i[1:]] != d['o'][i[:-1]]) | (d['h'][i[1:]] != d['h'][i[:-1]] + 1)]
    start = np.flatnonzero(brk)
    return np.diff(np.r_[start, len(i)]).tolist()


def gather(on: str) -> dict:
    """Все карточки периода: (тип, час, объект), попавшие в эпизод и длины непрерывных тревог."""
    out, hits, hours, runlen = [], [], None, []
    for tp in config.TYPES:
        d = load(tp, on)
        a = alarms(d, OP[tp]['share'], OP[tp]['reject_k'])
        out += [(tp, int(d['h'][i]), int(d['o'][i])) for i in np.flatnonzero(a)]
        hits += [(tp, int(d['h'][i]), int(d['o'][i])) for i in np.flatnonzero(a & d['y'])]
        runlen += runs(d, a)
        if hours is None:
            hours = np.unique(d['h'])
        print(f'{on} {tp}: карточек {int(a.sum())}', file=sys.stderr, flush=True)
    return dict(list=out, hours=hours, hits=hits, runs=np.array(runlen, dtype=np.int64))


def table_hours(name: str, cnt: np.ndarray) -> None:
    print(f'| {name} | {cnt.mean():.2f} | {(cnt == 0).mean():.0%} | {q(cnt, 0.5):.0f} | '
          f'{q(cnt, 0.9):.0f} | {q(cnt, 0.99):.0f} | {cnt.max()} | {(cnt > SEG).mean():.2%} |')


def main() -> None:
    coll = collectors()
    data = {on: gather(on) for on in ('val', 'test')}
    print('\n## Нагрузка на диспетчера: сколько карточек и куда они садятся\n')
    print('Карточка — объект × тип × час тревоги, доли и правило отклонения из настроек '
          f'(раздел 46). «Больше {SEG}» — доля часов, где карточек на парк больше {SEG}: '
          'столько сегментов даёт сегментный режим ТЗ.\n')

    print('### Одновременных карточек в часе\n')
    print(f'| период | среднее | пустых часов | медиана | 90% | 99% | максимум | больше {SEG} |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    cnt = {}
    for on in ('val', 'test'):
        cnt[on] = per_hour(data[on], data[on]['hours'])
        table_hours(on, cnt[on])

    print('\n### Нахлёст: два и более типа на одном объекте в один час\n')
    print('| период | карточек | объекто-часов под тревогой | карточек на объекто-час | '
          'доля объекто-часов с нахлёстом | карточек уберёт схлопывание |')
    print('|---|---:|---:|---:|---:|---:|')
    for on in ('val', 'test'):
        pairs = Counter((h, o) for _, h, o in data[on]['list'])
        n, m = len(data[on]['list']), len(pairs)
        over = sum(1 for v in pairs.values() if v > 1)
        print(f'| {on} | {n} | {m} | {n / max(m, 1):.2f} | {over / max(m, 1):.1%} | '
              f'{n - m} ({(n - m) / max(n, 1):.0%}) |')

    print('\n### На сколько объектов садится нагрузка\n')
    print('| период | объектов с тревогами | топ-5 объектов, % карточек | топ-10, % | '
          'объектов на половину карточек | они же дают поимок, % |')
    print('|---|---:|---:|---:|---:|---:|')
    for on in ('val', 'test'):
        by_obj = Counter(o for _, _, o in data[on]['list'])
        hits = Counter(o for _, _, o in data[on]['hits'])
        tot, th = sum(by_obj.values()), sum(hits.values())
        rank = [o for o, _ in by_obj.most_common()]
        acc, half = 0, 0
        for i, o in enumerate(rank, 1):
            acc += by_obj[o]
            if acc >= tot / 2:
                half = i
                break
        top_half = set(rank[:half])
        print(f'| {on} | {len(by_obj)} | {sum(by_obj[o] for o in rank[:5]) / tot:.0%} | '
              f'{sum(by_obj[o] for o in rank[:10]) / tot:.0%} | {half} | '
              f'{sum(hits[o] for o in top_half) / max(th, 1):.0%} |')

    print('\n### Коллекторы: ровно ли ложится нагрузка\n')
    print('Карточек в сутки на коллектор — по часовой мере за весь период.\n')
    print('| период | коллекторов | объектов | тихий, карточек в сутки | медиана | '
          'шумный | шумный / тихий |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for on in ('val', 'test'):
        days = len(data[on]['hours']) / 24
        by_coll = Counter(coll.get(o, -1) for _, _, o in data[on]['list'])
        obj_by_coll = Counter(coll.get(o, -1) for o in set(coll))
        known = {c: v for c, v in by_coll.items() if c != -1}
        for c in obj_by_coll:
            known.setdefault(c, 0)
        v = np.array(sorted(known.values()), dtype=float) / days
        print(f'| {on} | {len(known)} | {sum(obj_by_coll[c] for c in known)} | {v[0]:.2f} | '
              f'{np.median(v):.2f} | {v[-1]:.2f} | {v[-1] / max(v[0], 1e-9):.1f}x |')

    print('\n### По часам суток: когда пик\n')
    print('Час — момент прогноза (конец часа). Карточек в среднем на час.\n')
    print('| период | ' + ' | '.join(f'{(h + 1) % 24:02d}' for h in range(24)) + ' |')
    print('|---|' + '---:|' * 24)
    for on in ('val', 'test'):
        days = len(data[on]['hours']) / 24
        by_h = Counter(h % 24 for _, h, _ in data[on]['list'])
        print(f'| {on} | ' + ' | '.join(f'{by_h.get(h, 0) / days:.1f}' for h in range(24)) + ' |')

    print('\n### Сколько живёт одна тревога\n')
    print('Непрерывная серия часов тревоги по одному объекту и типу — это одна карточка, '
          'которая висит в списке. От этого зависит, обновляется список каждый час или стоит.\n')
    print('| период | серий | медиана, ч | 90%, ч | максимум, ч | доля карточек в сериях дольше суток |')
    print('|---|---:|---:|---:|---:|---:|')
    for on in ('val', 'test'):
        r = data[on]['runs']
        print(f'| {on} | {len(r)} | {q(r, 0.5):.0f} | {q(r, 0.9):.0f} | {r.max()} | '
              f'{r[r > 24].sum() / r.sum():.0%} |')

    print('\n### Снимок 07:00: что видит смена\n')
    print('| период | карточек в снимке, среднее | максимум | объектов в снимке, среднее |')
    print('|---|---:|---:|---:|')
    for on in ('val', 'test'):
        snap = Counter()
        objs = {}
        for _, h, o in data[on]['list']:
            if h % 24 == HOUR:
                snap[h] += 1
                objs.setdefault(h, set()).add(o)
        ndays = len({int(h) for h in data[on]['hours'] if h % 24 == HOUR})
        peak = max(snap.values()) if snap else 0
        print(f'| {on} | {sum(snap.values()) / ndays:.2f} | {peak} | '
              f'{sum(len(s) for s in objs.values()) / ndays:.2f} |')


if __name__ == '__main__':
    main()
