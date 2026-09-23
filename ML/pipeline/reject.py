"""Правило отклонения прогноза диспетчером (INTEGRATION §2.2, §9.1).

Диспетчер может отклонить прогноз: по конкретному объекту и типу. Отклонение не пишется константой:
для большинства типов оно только попадает в историю (и в метки для переобучения §9.2), а для газа и
подтопления (настройка `reject_types`) включает правило: оценка падает в `reject_k` раз — и держится,
пока не придёт настоящий эпизод либо диспетчер не снимет отклонение вручную (REOPEN).

`apply_reject` и `rejection_table` — чистые функции без связи с хранилищем: их переносит сервис
(service/rules.py) и вызывает на каждой границе часа.

    python reject.py --run main_h24_tuned --type gas        # число на прогнозах прогона
"""
import argparse
import json

import numpy as np
import polars as pl

import config


def apply_reject(score: float, decision_ts: int, episode_ts: int | None, reopen_ts: int | None,
                 now: int, k: float) -> tuple[float, bool]:
    """Скорректированная оценка и флаг «отклонение активно».

    Время в часах (целые индексы). Отклонение активно, если оно было принято до `now`, после него
    не было настоящего эпизода и диспетчер его не снял. Пока отклонение активно, оценка типа
    умножается на (1 - k) — тревога может уйти, только если модель была действительно уверена.
    """
    active = (decision_ts is not None and decision_ts <= now
              and (reopen_ts is None or reopen_ts <= decision_ts)
              and (episode_ts is None or episode_ts < decision_ts or episode_ts > now))
    return (score * (1.0 - k) if active else score), active


def rejection_table(objects: list[int], hours: np.ndarray, tp: str, decisions: pl.DataFrame,
                    episodes: pl.DataFrame, k: float) -> pl.DataFrame:
    """Активность отклонения по объекто-часам одной таблицей (для оценки и тестов).

    decisions: колонки object_id, h (час решения), action ('REJECT'/'REOPEN'/...).
    episodes:  object_id, h0 (час начала эпизода — «настоящего»).
    Возвращает по объекто-часу: активно ли отклонение и во сколько умножить оценку.
    """
    n = len(objects) * len(hours)
    o, hh = np.repeat(objects, len(hours)), np.tile(hours, len(objects))
    df = pl.DataFrame({'object_id': o, 'h': hh}).sort(['object_id', 'h'])
    df = df.join(decisions.filter(pl.col('type') == tp).select('object_id', 'h', 'action'),
                 on=['object_id', 'h'], how='left')
    ep = episodes.filter(pl.col('type') == tp).select('object_id', 'h0')
    df = df.join(ep, on='object_id', how='left').with_columns(
        active=pl.Series(np.zeros(n, bool)),
        weight=pl.Series(np.full(n, 1.0)))
    # активность считается по ходу времени: свёртка по каждому объекту
    rows = []
    for oid in np.unique(o):
        r = df.filter(pl.col('object_id') == oid).sort('h')
        last = None
        out = []
        for row in r.rows():
            h, action = row[1], row[2]
            if action == 'REJECT':
                last = h
            elif action == 'REOPEN':
                last = None
            active = last is not None and h >= last
            out.append(active)
        rows.append(pl.DataFrame({'object_id': oid, 'h': r['h'].to_list(), 'active': out}))
    act = pl.concat(rows)
    return act.with_columns(weight=pl.when(pl.col('active')).then(1.0 - k).otherwise(1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='прогон для оценки (только числа в контексте прогнозов)')
    ap.add_argument('--type', default='gas', help='тип: для остальных правило выключено (§9.1)')
    args = ap.parse_args()
    if not args.run:
        raise SystemExit('нужен --run с сохранёнными прогнозами')
    st = json.loads((config.ML / 'settings' / 'operating.json').read_text(encoding='utf-8'))
    if args.type not in st['reject_types']:
        print(f'тип {args.type} вне списка {st["reject_types"]} — правило выключено, '
              'см. operating.json')
        return
    print(f'правило: оценка ×(1 - {st["reject_k"]}) для {args.type}, '
          'пока не придёт настоящий эпизод или REOPEN')