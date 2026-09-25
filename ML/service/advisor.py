"""Рекомендации в такте (M13, INTEGRATION §2.3 и §10.1): к каждой тревоге прогноза — меры по словарю.

Вывод правил целиком живёт в main (`pipeline/recommend.py`): словарь мер `settings/recommendations.csv`,
стадии рецидива `settings/recurrence.csv`, один выезд `visit`, составная `compose`. Здесь — только
тонкий контекст тревоги прогноза на момент такта. Он строки эпизода не требует: у тревоги модели ещё
нет ни канала, ни паттерна сработки, поэтому контекст собирается лоскутно, а не `context()` из
recommend.py (тот строит временные таблицы по эпизодам всего парка — на каждый такт слишком тяжело):

- счёт чистых эпизодов объекта и типа за 30 суток и последний с подтверждением-выездом — из разметки
  `inc`, которую на такт уже построил M3 (labels.build внутри `retro.snapshot`) — на стадию R0–R3 и
  поправку «повтор после выезда»;
- обстановка за сутки (`recommend.ambient`: мигалка насоса, холод);
- свежий режим охраны и подача питания за 10 минут — поправки «персонал на объекте», «перезапуск
  после питания», «плановая проверка газа»;
- согласие типов (на объекте и по коллектору) — из результатов того же такта, передаются из main.

Рекомендация того же формата, что у `recommend.run()` в прогнозной ветке: `recommend.recommend()` с
`stype = None`. В сообщение она ложится как `recommendation` типа (§2.3); когда на объекте тревожат
несколько типов, main зовёт `recommend.compose` и кладёт `object_recommendation`. Режим «факт» (M8)
для объявлений по факту здесь не строится — у него свой путь (`factalert`), рекомендация приходит
той же функцией со строкой эпизода.
"""
import recommend

_once = None


def load():
    """Словарь и рецидив один раз на процесс: правка settings/*.csv меняет версию — см. §2.3."""
    global _once
    if _once is None:
        rules, version = recommend.load_rules()
        _once = (rules, recommend.load_recurrence(), version)
    return _once


def _history(con, obj: int, tp: str, at):
    """(чистых эпизодов за 30 суток, (t0, подтверждён) последнего) — для стадии и поправок."""
    row = con.sql(f"""SELECT count(*) AS k, arg_max(confirmed, t0) AS c, max(t0) AS t
                      FROM inc WHERE object_id = {obj} AND type = '{tp}' AND noise IS NULL
                        AND t0 < TIMESTAMP '{at.isoformat()}'
                        AND t0 >= TIMESTAMP '{at.isoformat()}' - INTERVAL 30 DAY""").fetchone()
    return int(row[0] or 0), (row[2], bool(row[1])) if row[2] is not None else None


def _armed(con, obj: int, at):
    row = con.sql(f"""SELECT armed FROM guard WHERE object_id = {obj} AND ts <= TIMESTAMP '{at.isoformat()}'
                      ORDER BY ts DESC LIMIT 1""").fetchone()
    return bool(row[0]) if row and row[0] is not None else None


def _restart(con, obj: int, at) -> bool:
    row = con.sql(f"""SELECT count(*) FROM ev_all WHERE object_id = {obj}
                      AND ts BETWEEN TIMESTAMP '{at.isoformat()}' - INTERVAL 10 MINUTE AND TIMESTAMP '{at.isoformat()}'
                      AND state IN ('Есть питание', 'Питание от сети')""").fetchone()
    return int(row[0]) > 0


def build(con, obj: int, tp: str, at, *, reasons=(), since_h=None, co_types=(), line_n: int = 0) -> dict:
    """Рекомендация к одной тревоге прогноза (режим recommend() сам определит как «прогноз»)."""
    rules, recur, ver = load()
    k, last = _history(con, obj, tp, at)
    r = {
        'object_id': obj, 'type': tp, 't0': at, 'stype': None, 'what': None, 'channel_id': None,
        'k30': k, 'prev_t0': last[0] if last else None, 'prev_confirmed': last[1] if last else False,
        'prev_channel': None, 'line_n': int(line_n), 'armed': _armed(con, obj, at),
        'restart': _restart(con, obj, at), 'co_types': list(co_types),
        'reason_triggers': recommend.reason_triggers(tp, list(reasons), rules),
        'since_hours': since_h,
    }
    r.update(recommend.ambient(con, obj, at))
    return recommend.recommend(r, rules, ver, recur)