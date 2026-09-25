"""Обратный поток решений диспетчера (M11): потребитель tf.dispatch.decisions (§9.1).

Схема §9.1 — {schema, prediction_id, object_id, type, action, reason_code, user_id,
decided_at, task_id}; действие MUTE несёт срок `mute_hours` (без него — REJECT_N_HOURS).
Парсер толерантен к `ts` вместо `decided_at` — стендовый parquet-журнал писал по-старому.

Семантика действий (§9.1, rules.py): TAKE ничего не меняет (для статистики и обучения),
REJECT включает правило отклонения (газ/подтопление, k=0,2; у остальных — только запись в
историю), MUTE — временное молчание по паре, REOPEN снимает отклонение/молчание вручную;
CONFIRMED — подтверждённое происшествие: снимает отклонение и молчание, как настоящий эпизод.
Час решения — от начала витрины, тот же `h`, что в features.parquet (§9.2: пары
«прогноз — решение — происшествие» сравниваются внутри одной версии).
"""
from datetime import datetime, timedelta, timezone

import svc as config

VALID_ACTIONS = ('TAKE', 'REJECT', 'MUTE', 'REOPEN', 'CONFIRMED')


def _offsets() -> timedelta:
    tz = config.TZ if getattr(config, 'TZ', None) else '+03:00'
    return timedelta(hours=int(tz[1:3]), minutes=int(tz[4:6] or 0))


def to_hour(ts) -> int:
    """Час решения в часах от начала витрины (features.T0) — тот же индекс, что у оценок."""
    t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    if t.tzinfo is not None:
        t = t.astimezone(timezone(_offsets())).replace(tzinfo=None)
    import features as ft
    return int((t - ft.T0).total_seconds() // 3600)


def apply(rules, rows: list[dict], audit_path=None) -> int:
    """Применить решения диспетчера к состоянию правил; возврат — число применённых.

    Строки без object_id/type/action/времени или с неизвестным действием пропускаются:
    неизвестное действие логируем в правила как есть, если оно в VALID_ACTIONS. Шум топика —
    это проблема шины, а не аварийная остановка такта.

    audit_path (необязателен, чтобы тесты оставались без побочных файлов) — parquet §9.2: пары
    «прогноз — решение — происшествие» складываются в таблицу, которую читает обучение (M9).
    Хранится сторона решения: час h (тот же индекс, что у оценок), пара и мета решения. Оценку
    прогноза и сбывшееся происшествие обучение подтягивает по (object_id, type, h) из истории —
    там же, где считает пары для отчёта «отклонили-и-случилось» (INTEGRATION §9.2).
    """
    n = 0
    entries = []
    for d in rows:
        try:
            oid, tp, action = int(d['object_id']), str(d['type']), str(d['action'])
        except (KeyError, TypeError, ValueError):
            continue
        ts = d.get('decided_at') or d.get('ts')
        if action not in VALID_ACTIONS or ts is None:
            continue
        mute = d.get('mute_hours')
        rules.on_decision(oid, tp, action, to_hour(ts),
                          mute_hours=int(mute) if mute is not None else None)
        n += 1
        entries.append({'h': to_hour(ts), 'object_id': oid, 'type': tp, 'action': action,
                        'reason_code': d.get('reason_code'), 'user_id': d.get('user_id'),
                        'decided_at': ts, 'task_id': d.get('task_id'),
                        'mute_hours': int(mute) if mute is not None else None})
    if audit_path and entries:
        _append_audit(audit_path, entries)
    return n


def _append_audit(path, rows: list[dict]) -> None:
    import polars as pl
    df = (pl.DataFrame(rows)
          .with_columns([pl.col('h').cast(pl.Int64),
                         pl.col('object_id').cast(pl.Int64),
                         pl.col('mute_hours').cast(pl.Int64)]))
    if path.exists():
        df = pl.concat([pl.read_parquet(path), df])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)