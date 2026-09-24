"""Сбой канала: что будет с прогнозом, если группа датчиков замолчит.

В журнале датчики иногда выходят из строя, а на стенде сбой канала — один из трёх сценариев
эмулятора. Модель при этом остаётся прежней: переобучать её на «мир без газовых датчиков» никто не
станет, она просто перестаёт получать часть входа. Здесь считается цена такого молчания на рабочих
моделях смеси: группа признаков заменяется на «данных нет», остальное не трогается.

Как задаётся молчание. Счётчики событий обнуляются — событий по мёртвому каналу не приходит.
Числовые сводки (максимум, среднее, тренд газа и температуры) становятся NaN — показаний нет, а
ноль был бы враньём (ноль градусов — это значение, а не молчание). Бустинги оба умеют NaN.

Что ещё меряется этим же ходом: группы «календарь», «состав объекта», «история эпизодов»,
«коллектор» — это не сбой канала, а проверка, на чём модель держится. Если выключение группы не
двигает числа, её можно не собирать в проде (M3 в `INTEGRATION.md`), и наоборот.

Мера — часовая при доле из настроек (разделы 34, 46) и суточный снимок 07:00 (раздел 49).
Базовая строка считается тем же кодом на нетронутых признаках: она должна совпасть с таблицами
раздела 46, это проверка сцепки.

    python ablate.py "$MIXT"
"""
import json
import re
import sys

import numpy as np
import polars as pl

import config
import metrics
import operating as op
import retro
from daily import alarms, daily
from smooth import measure

MIX = sys.argv[1]
H = 24
# У отказа оборудования в эксплуатации стоит сеть (раздел 48), а она читает не строку витрины, а
# часовые ряды: выключить в ней группу признаков нечем. Этот тип меряется на бустинговой смеси —
# той, что стояла до раздела 48; выводы о вкладе групп переносятся на сеть только качественно.
EQ_CAT = '+'.join(['main_h24_tunedh24/cat'] + [f'main_h24_tunedh24_s{i}/cat' for i in range(1, 5)])
OP = config.operating()['types']
FEAT = config.WORK / 'features'
META = json.loads((FEAT / 'meta.json').read_text(encoding='utf-8'))
FEATURES = META['features']

# Группы: имя -> префиксы имён признаков (без окна). Первые восемь — семейства датчиков,
# остальные — не каналы, а способ проверить, на чём держится модель.
GROUPS = {
    'дым': ('smoke', 'smoke_clear', 'smoke_fault'),
    'тепловые и ручные': ('heat', 'manual', 'uir_call', 'uir_lever'),
    'температура': ('temp_hi', 'temp_lo', 'temp_fault', 'temp_max', 'temp_min', 'temp_mean',
                    'temp_trend', 'since_temp'),
    'газ': ('gas', 'gas_fault', 'gas_max', 'gas_mean', 'gas_trend'),
    'насосы': ('pump_on', 'pump_off', 'pump_flooded', 'pump_all', 'pump_fault'),
    'вентиляция': ('fan_on', 'fan_off', 'fan_fault'),
    'датчик воды': ('flood_sensor',),
    'питание и УПС': ('ups_batt', 'ups_fault', 'ups_mains', 'phase_on', 'phase_off', 'phase_fault',
                      'switch', 'off', 'disconnected'),
    'охрана': ('arm', 'armed', 'disarm', 'door', 'motion', 'hatch', 'guard_faulty', 'guard_object',
               'since_guard'),
    'выезды бригады': ('visit', 'arrival', 'since_visit'),
    'история эпизодов объекта': tuple(f'{a}_{b}' for a in ('onset', 'trig') for b in config.TYPES)
                                + tuple(f'since_{b}' for b in config.TYPES)
                                + ('noise_fire', 'noise_flood', 'since_event'),
    'коллектор': tuple(f'coll_{a}_{b}' for a in ('onset', 'trig') for b in config.TYPES)
                 + ('coll_noise_fire', 'coll_noise_flood'),
    'календарь': ('hour', 'dow', 'month', 'doy_sin', 'doy_cos', 'holiday', 'long_holiday', 'may9',
                  'days_to_holiday'),
    'состав объекта': tuple(f'comp_{i}' for i in range(19)) + ('channels',),
    'прочие счётчики': ('events', 'av', 'fault_other', 'undefined', 'norm'),
}
NUMERIC = ('_max', '_min', '_mean', '_trend')   # сводки: молчание — это NaN, а не ноль


def is_numeric(name: str) -> bool:
    """Сводка показаний (а не счётчик событий): у неё молчание канала — пропуск, не ноль."""
    return re.sub(r'_\d+h$', '', name).endswith(NUMERIC)


def columns(prefixes: tuple) -> list:
    out = []
    for i, name in enumerate(FEATURES):
        for p in prefixes:
            if name == p or name.startswith(p + '_') and name[len(p) + 1:].rstrip('h').isdigit():
                out.append(i)
                break
    return out


def build(o, h, n, p) -> dict:
    """То же, что `smooth.load`, но с оценкой, посчитанной здесь."""
    order = np.lexsort((h, o))
    o, h, n, p = o[order], h[order], n[order], p[order]
    p = p.argsort().argsort() / len(p)
    key = o.astype(np.int64) * 10 ** 7 + h
    eps = np.array(sorted(metrics.onsets(o, h, n, metrics.RUN_CAP)), dtype=np.int64).reshape(-1, 2)
    wk = eps[:, :1] * 10 ** 7 + eps[:, 1:] - np.arange(H, 0, -1)[None, :]
    pos = np.minimum(np.searchsorted(key, wk), len(key) - 1)
    W = np.where(key[pos] == wk, pos, -1)
    return dict(o=o, h=h, y=n <= H, p=p, W=W[(W >= 0).any(1)], eps=eps)


def spec_without_nets(mix: str) -> str:
    table = dict(part.split('~', 1) for part in mix.split(';') if part)
    if 'tcn' in table.get('equipment', ''):
        table['equipment'] = EQ_CAT
    return ';'.join(f'{k}~{v}' for k, v in table.items())


def main() -> None:
    mix = spec_without_nets(MIX)
    if mix != MIX:
        print('отказ оборудования меряется на бустинговой смеси: сеть не читает строку витрины',
              file=sys.stderr, flush=True)
    models = retro.load_mix_models(mix)
    cols = {g: columns(pref) for g, pref in GROUPS.items()}
    for g, c in cols.items():
        assert c, f'группа {g} не нашла ни одного признака'
    print('Признаков в группах: ' + ', '.join(f'{g} {len(c)}' for g, c in cols.items()),
          file=sys.stderr, flush=True)

    res = {}
    for on in ('val', 'test'):
        yr = op.years(mix, on)
        keep = FEATURES + ['object_id', 'h'] + [f'next_{t}' for t in config.TYPES]
        df = pl.concat([pl.scan_parquet(FEAT / f'{y}.parquet').select(keep).collect() for y in yr])
        X = np.array(df.select(FEATURES).to_numpy(), dtype=np.float32)   # своя копия: столбцы правятся на месте
        o, h = df['object_id'].to_numpy(), df['h'].to_numpy()
        nxt = {t: df[f'next_{t}'].to_numpy() for t in config.TYPES}
        del df
        print(f'{on}: строк {len(X)}', file=sys.stderr, flush=True)
        for tp in config.TYPES:
            n = nxt[tp]
            s, k = OP[tp]['share'], OP[tp]['reject_k']
            for g in ['без сбоя'] + list(GROUPS):
                if g != 'без сбоя':
                    idx = cols[g]
                    saved = X[:, idx].copy()
                    fill = np.where([is_numeric(FEATURES[i]) for i in idx],
                                    np.nan, 0.0).astype(np.float32)
                    X[:, idx] = fill
                p = models[tp](X, {})
                X.setflags(write=True)    # CatBoost закрывает массив на запись после предсказания
                if g != 'без сбоя':
                    X[:, idx] = saved
                d = build(o, h, n, p)
                a = alarms(d, s, k)
                res[tp, on, g] = (measure(d, a), daily(d, a))
                print(f'{on} {tp} {g}: поймано {res[tp, on, g][0]["caught"]}',
                      file=sys.stderr, flush=True)
        del X, nxt

    print()
    print('## Сбой канала и вклад групп признаков')
    print()
    print('Доли и правило отклонения — из настроек. В клетке: поймано эпизодов (часовая мера), '
          'в скобках — изменение к строке «без сбоя».')
    print()
    for on in ('val', 'test'):
        print(f'### {on}')
        print()
        print('| группа | ' + ' | '.join(config.TYPE_NAMES[tp] for tp in config.TYPES) + ' | все типы |')
        print('|---|' + '---:|' * (len(config.TYPES) + 1))
        for g in ['без сбоя'] + list(GROUPS):
            cells, tot, base_tot = [], 0, 0
            for tp in config.TYPES:
                c = res[tp, on, g][0]['caught']
                b = res[tp, on, 'без сбоя'][0]['caught']
                tot += c
                base_tot += b
                cells.append(f'{c}' if g == 'без сбоя' else f'{c} ({c - b:+d})')
            cells.append(f'**{tot}**' if g == 'без сбоя' else f'**{tot} ({tot - base_tot:+d})**')
            print(f'| {g} | ' + ' | '.join(cells) + ' |')
        print()

    print('### Ложные сигналы при том же пороге')
    print()
    print('| группа | проверка | тест |')
    print('|---|---:|---:|')
    for g in ['без сбоя'] + list(GROUPS):
        cells = []
        for on in ('val', 'test'):
            f = sum(res[tp, on, g][0]['false_sig'] for tp in config.TYPES)
            b = sum(res[tp, on, 'без сбоя'][0]['false_sig'] for tp in config.TYPES)
            cells.append(f'{f}' if g == 'без сбоя' else f'{f} ({f - b:+d})')
        print(f'| {g} | ' + ' | '.join(cells) + ' |')

    print()
    print('### Суточный снимок 07:00: поймано объекто-суток')
    print()
    print('| группа | проверка | тест |')
    print('|---|---:|---:|')
    for g in ['без сбоя'] + list(GROUPS):
        cells = []
        for on in ('val', 'test'):
            x = sum(res[tp, on, g][1]['hit'] for tp in config.TYPES)
            b = sum(res[tp, on, 'без сбоя'][1]['hit'] for tp in config.TYPES)
            cells.append(f'{x}' if g == 'без сбоя' else f'{x} ({x - b:+d})')
        print(f'| {g} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
