"""Карточки диспетчера (INTEGRATION §9.8): сколько карточек открыто одновременно.

Карточка — объект × тип × час тревоги. Отсюда размер списка прогнозов в интерфейсе, нужность
пагинации и то, что тревогу надо склеивать по непрерывности часов (object_id + type + подряд),
иначе история происшествий заполнится копиями одной тревоги.

    python dispatch.py --alarms work/out/alarms.parquet   # колонки object_id, h, type, alarm

    python dispatch.py --alarms ... > ../work/dispatch.md
"""
import argparse

import numpy as np
import polars as pl

import config


def split_series(hours: pl.DataFrame) -> pl.DataFrame:
    """Разбивает часы тревоги пары (object_id, type) на непрерывные серии.

    Возвращает те же строки с номером серии: непрерывные часы одной пары — одна серия.
    """
    h = hours.with_columns((pl.col('h') - pl.col('h').shift(1)).over(['object_id', 'type'])
                           .fill_null(1).alias('gap'))
    return h.with_columns((pl.col('gap') > 1).over(['object_id', 'type']).cum_sum().alias('series'))


def card_metrics(alarms: pl.DataFrame) -> dict:
    """Показатели §9.8 по таблице тревог (object_id, h, type, alarm=True)."""
    a = alarms.filter(pl.col('alarm'))
    open_cards = (a.group_by('h').agg(pl.len().alias('open'))
                  .with_columns(hour=True).sort('h'))
    nopen = open_cards['open'].to_numpy()
    series = split_series(a)
    life = series.group_by(['object_id', 'type', 'series']).agg(n=pl.len())
    med = float(np.median(life['n'].to_numpy())) if life.height else 0.0
    long_s = life['n'].to_numpy()
    share_long = float(long_s[long_s > 24].sum() / max(long_s.sum(), 1))
    by_multi = (series.group_by(['object_id', 'h']).agg(types=pl.col('type').n_unique())
                .filter(pl.col('types') > 1).height)
    total_oh = int(series.group_by(['object_id', 'h']).len().height)
    return {
        'open_avg': float(np.mean(nopen)),
        'open_p99': float(np.percentile(nopen, 99)),
        'open_max': float(np.max(nopen)) if nopen.size else 0.0,
        'hours_over_10': float(np.mean(nopen > 10)),
        'overlap_share': by_multi / max(total_oh, 1),
        'life_median_h': med,
        'card_hours_in_long_series': share_long,
        'longest_series_h': int(long_s.max()) if long_s.size else 0,
    }


def md(res: dict, df: pl.DataFrame | None = None) -> str:
    lines = ['| показатель | значение |']
    lines.append('|---|---|')
    for k, v in res.items():
        lines.append(f'| {k} | {v:.2f}' if isinstance(v, float) else f'| {k} | {v} |')
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--alarms', required=True, help='parquet с колонками object_id, h, type, alarm')
    args = ap.parse_args()
    alarms = pl.read_parquet(args.alarms)
    needed = {'object_id', 'h', 'type', 'alarm'}
    missing = needed - set(alarms.columns)
    if missing:
        raise SystemExit(f'в {args.alarms} нет колонок: {", ".join(sorted(missing))}')
    print('## Карточки диспетчера\n')
    print(md(card_metrics(alarms)))