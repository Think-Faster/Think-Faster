"""Липкая тревога сети отказа оборудования (раздел 60): откуда она и чем лечится.

Тест 2026 режется на отрезки прогона вперёд, как в `rollcmp.py`; семейства пишутся так же
(`tcn:all:0,1,2`, `tcn:all#2025-01-01:0,1,2`, `tcn24`, `mix`, `cat:all`). Порог — доля часов
раздела 38. Свежая поимка — эпизод, пойманный тревогой, которая стоит меньше `metrics.RUN_CAP` часов.

    python sticky.py folds   tcn:all#2025-01-01:0,1,2 tcn:frozen:0,1,2 tcn:all:0,1,2
    python sticky.py runs    tcn:all#2025-01-01:0,1,2 tcn:frozen:0,1,2 tcn:all:0,1,2
    python sticky.py chronic tcn:all#2025-01-01:0,1,2 tcn:frozen:0,1,2
    python sticky.py detrend equipment tcn:all:0,1,2 tcn:all#2025-01-01:0,1,2 mix

folds   — поймано / свежих по каждому отрезку: держится ли разница на всём тесте или на одном месяце;
runs    — форма тревог: сколько серий, их длина, доля часов в сериях дольше 7 суток и сколько тревога
          уже стоит к началу эпизода;
chronic — доля часов тревоги на 10 объектах с наибольшим числом эпизодов 2025 года и поимки там
          и на остальных;
detrend — поправка p' = p − a·(среднее p за прошлые 168 ч того же объекта). a выбирается на
          январе–марте, проверяется на апреле–июне. В каждой половине опора — a = 0 при доле
          раздела 38, у каждого a берётся лучшая по свежим точка кривой, не дороже опоры ни по ложным
          часам, ни по ложным сигналам (склейка 6 ч, раздел 27).
"""
import sys

import numpy as np
import metrics
import operating
import rollcmp as R

H = R.H
AS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
b5 = lambda run, fam: '+'.join(f'{run}{"" if s == 0 else f"_s{s}"}{fam}' for s in range(5))
R.MIX = (f"fire~{b5('main_h24_tunedh24', '/cat')};gas~{b5('main_h24', '/cat')};"
         f"flood~{b5('main_h24_tuned', '/cat')};equipment~{b5('main_h24_tunedh24', '/cat')};"
         f"sensor~{b5('main_h24_tuned', '')};intrusion~{b5('main_h24', '/cat')}")


def catches(d, alarm, W=None):
    """Какие эпизоды пойманы и какие из них свежие (тревога стоит меньше RUN_CAP часов)."""
    W = d['W'] if W is None else W
    o, h = d['o'], d['h']
    Wv = np.where(W >= 0, W, 0)
    hit = alarm[Wv] & (W >= 0)
    caught = hit.any(1)
    first = hit.argmax(1)
    idx = np.flatnonzero(alarm)
    stood = np.full(len(alarm), -1, np.int64)
    if len(idx):
        oo, hh = o[idx], h[idx]
        start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
        stood[idx] = hh - hh[start][np.cumsum(start) - 1]
    stand = caught & (stood[Wv[np.arange(len(W)), first]] + (H - first) >= metrics.RUN_CAP)
    return caught, caught & ~stand


def alarm_at(p, share):
    return p >= np.quantile(p, 1 - share)


def folds(specs, tp='equipment'):
    d = R.test_rows(tp)
    efold = np.searchsorted(R.CUT_H, d['eps'][:, 1], side='right') - 1
    nf = len(R.CUTS)
    print('эпизодов по отрезкам: ' + ' / '.join(str((efold == f).sum()) for f in range(nf)))
    print()
    print('| сеть | ' + ' | '.join(f'с {c[5:]}' for c in R.CUTS) + ' | всего |')
    print('|---|' + '---:|' * (nf + 1))
    for spec in specs:
        caught, fresh = catches(d, alarm_at(R.score(spec, tp, d), R.S38[tp]))
        cells = [f'{caught[efold == f].sum()} / {fresh[efold == f].sum()}' for f in range(nf)]
        print(f'| {spec} | ' + ' | '.join(cells) + f' | {caught.sum()} / {fresh.sum()} |', flush=True)


def runs(specs, tp='equipment'):
    d = R.test_rows(tp)
    o, h, key = d['o'], d['h'], d['key']
    ek = d['eps'][:, 0] * 10**7 + d['eps'][:, 1] - 1          # час перед началом эпизода
    pe = np.minimum(np.searchsorted(key, ek), len(key) - 1)
    pe = pe[key[pe] == ek]
    print('| сеть | серий тревоги | средняя длина, ч | медиана, ч | доля часов в сериях > 7 сут '
          '| к началу эпизода тревога стоит, медиана ч |')
    print('|---|---:|---:|---:|---:|---:|')
    for spec in specs:
        a = alarm_at(R.score(spec, tp, d), R.S38[tp])
        idx = np.flatnonzero(a)
        oo, hh = o[idx], h[idx]
        start = np.r_[True, (oo[1:] != oo[:-1]) | (hh[1:] - hh[:-1] > 1)]
        rid = np.cumsum(start) - 1
        L = np.bincount(rid)
        stood = np.full(len(a), -1)
        stood[idx] = hh - hh[start][rid]
        print(f'| {spec} | {len(L)} | {L.mean():.1f} | {np.median(L):.0f} | {L[L > 168].sum() / L.sum():.1%} '
              f'| {np.median(stood[pe][a[pe]]):.0f} |', flush=True)


def chronic(specs, tp='equipment'):
    d = R.test_rows(tp)
    vo, vh, vn, _ = operating.load_mix(R.SEEDS5, 'val', [2025], tp, 'xgb', '')
    e25 = np.array(sorted(metrics.onsets(vo, vh, vn, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    ids, cnt = np.unique(e25[:, 0], return_counts=True)
    top = ids[np.argsort(-cnt)[:10]]
    rows = np.isin(d['o'], top)
    ep = np.isin(d['eps'][:, 0], top)
    print(f'эпизодов 2025: {len(e25)}, у верхних 10 объектов {np.sort(cnt)[::-1][:10].sum()}; '
          f'эпизодов теста: {len(ep)}, у тех же объектов {ep.sum()}')
    print()
    print('| сеть | часов тревоги на них | ложных часов на них | поймано: они / остальные | свежих: они / остальные |')
    print('|---|---:|---:|---:|---:|')
    for spec in specs:
        a = alarm_at(R.score(spec, tp, d), R.S38[tp])
        caught, fresh = catches(d, a)
        false = a & ~d['y']
        print(f'| {spec} | {a[rows].sum() / a.sum():.1%} | {false[rows].sum() / false.sum():.1%} '
              f'| {caught[ep].sum()} / {caught[~ep].sum()} | {fresh[ep].sum()} / {fresh[~ep].sum()} |', flush=True)


def detrend(tp, specs):
    d = R.test_rows(tp)
    key = d['key']
    lo = np.searchsorted(key, key - 168)
    i = np.arange(len(key))
    halves = []
    for k in (0, 1):
        m = (d['fold'] >= 3) == bool(k)
        sub = {x: d[x][m] for x in ('o', 'h', 'n', 'y', 'key', 'fold')}
        eps = np.array(sorted(metrics.onsets(sub['o'], sub['h'], sub['n'], metrics.RUN_CAP)),
                       dtype=np.int64).reshape(-1, 2)
        wk = eps[:, :1] * 10**7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
        pos = np.minimum(np.searchsorted(sub['key'], wk), len(sub['key']) - 1)
        W = np.where(sub['key'][pos] == wk, pos, -1)
        sub['W'] = W[(W >= 0).any(1)]
        halves.append((m, sub))

    def point(q, share, sub):
        alarm = alarm_at(q, share)
        caught, fresh = catches(sub, alarm)
        tot, true = metrics.signals(sub['o'], sub['h'], sub['y'], alarm, gap=6)
        return int((alarm & ~sub['y']).sum()), int(caught.sum()), int(fresh.sum()), tot - true

    fmt = lambda x: '—' if x is None else f'{x[1]} / {x[2]} / {x[0]} / {x[3]}'
    print(f'{tp}: поймано / свежих / ложных ч / ложных сигналов — лучшая точка, не дороже опоры')
    print()
    print('| сеть | половина | ' + ' | '.join(f'a = {a}' for a in AS) + ' | a по январю–марту → апрель–июнь |')
    print('|---|---|' + '---|' * len(AS) + '---|')
    for spec in specs:
        p = R.score(spec, tp, d)
        cs = np.r_[0, np.cumsum(p)]
        past = np.where(i > lo, (cs[i] - cs[lo]) / np.maximum(i - lo, 1), 0)   # прошлые 168 ч, без текущего
        res = []
        for m, sub in halves:
            ref = point(p[m], R.S38[tp], sub)
            row = []
            for a in AS:
                q = R.pct_fold((p - a * past)[m], sub['fold'])
                pts = [point(q, s, sub) for s in R.S38[tp] * np.geomspace(0.2, 1.5, 22)]
                pts += [ref] if a == 0 else []
                ok = [x for x in pts if x[0] <= ref[0] and x[3] <= ref[3]]
                row.append(max(ok, key=lambda x: (x[2], x[1])) if ok else None)
            res.append(row)
        k = int(np.argmax([-1 if x is None else x[2] for x in res[0]]))
        for hk, name in ((0, 'янв–мар'), (1, 'апр–июн')):
            tail = f'a = {AS[k]}: {fmt(res[1][k])}' if hk else ''
            print(f'| {spec} | {name} | ' + ' | '.join(fmt(x) for x in res[hk]) + f' | {tail} |', flush=True)


if __name__ == '__main__':
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == 'detrend':
        detrend(args[0], args[1:])
    else:
        {'folds': folds, 'runs': runs, 'chronic': chronic}[mode](args)
