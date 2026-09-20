"""Шаг 17. Канал «по факту»: что происшествие уже идёт, а прогноз его не дал.

У прогноза всегда будет доля пропусков: на тесте 2026 по пожару поймано 436 эпизодов из 603.
Оставшиеся 167 не должны исчезать. Для них нужен второй, отдельный канал — не «может случиться»,
а «уже случилось»: он не предсказывает, а опознаёт происшествие по первым же событиям журнала.
Требование к нему обратное прогнозу: полнота около 100% и ноль ложных, ценой того, что упреждения
нет — есть только задержка объявления.

Такой канал возможен, потому что эпизод в разметке и есть пачка триггеров: первый триггер — это и
есть момент происшествия. Вопрос только в том, сколько ждать перед объявлением. Ждать приходится
из-за шума: массовое срабатывание извещателей (Н8) и «Затоплен» сразу за событием питания (Н7)
выглядят как авария, но ею не являются. Оба признака видны из самого журнала в пределах десяти
минут, то есть фильтр шума работает в реальном времени, а не задним числом.

Сравниваются правила объявления:

| правило | когда объявляет |
|---|---|
| `first` | сразу на первом триггере типа |
| `nch2` | когда сработал второй разный канал объекта |
| `n2` | когда пришёл второй триггер |
| `clean` | на первом триггере, не помеченном шумом (Н7, Н8) |
| `clean2` | на втором таком триггере |
| `cleanch2` | когда шумом не помечен второй разный канал |
| `silent` | через 10 минут после первого триггера, если сам первый триггер не помечен шумом |

По каждому: сколько эпизодов объявлено, сколько из объявленных шумные (ложные), сколько настоящих
пропущено и с какой задержкой приходит объявление. Отдельно — сколько эпизодов, пропущенных
прогнозом, канал «по факту» закрывает.

    python factalert.py
    python factalert.py --run main_h24_tuned --year 2026
"""
import argparse

import duckdb
import numpy as np

import config
import labels
import metrics
import operating as op
import persist

RULES = ['first', 'n2', 'nch2', 'clean', 'clean2', 'cleanch2', 'silent']
WAIT = 10   # минут задержки: Н8 считается по ±10 минутам вокруг триггера, раньше не вычислим


def episodes(con, tp: str, years: tuple[int, ...], wait: int = WAIT) -> 'duckdb.DuckDBPyRelation':
    """Эпизоды типа с моментом срабатывания каждого правила объявления.

    Шум считается по тем же признакам, что и в разметке (labels.py), но по данным, доступным в
    момент события: массовость дыма — по числу разных каналов объекта за ±10 минут, подтопление —
    по событию питания в коллекторе за минуту до. Задним числом ничего не берётся.
    """
    yrs = ', '.join(str(y) for y in years)
    return con.sql(f"""
        WITH e AS (SELECT object_id, collector_id, type, t0, t1, noise, confirmed
                   FROM inc WHERE type = '{tp}' AND year(t0) IN ({yrs})
                     AND object_id IN (SELECT object_id FROM obj3)),
             g AS (SELECT t.object_id, t.channel_id, t.ts, t.type,
                          n.noise IS NULL AS clean          -- триггер не помечен Н7/Н8
                   FROM trig t LEFT JOIN trig_noise n
                     ON n.object_id = t.object_id AND n.channel_id = t.channel_id AND n.ts = t.ts
                   WHERE t.type = '{tp}'),
             t AS (SELECT e.object_id, e.t0, e.noise, e.confirmed, g.ts, g.clean,
                          row_number() OVER (PARTITION BY e.object_id, e.t0 ORDER BY g.ts) AS k,
                          row_number() OVER (PARTITION BY e.object_id, e.t0, g.clean
                                             ORDER BY g.ts) AS kc,
                          dense_rank() OVER (PARTITION BY e.object_id, e.t0
                                             ORDER BY g.channel_id) AS ch,
                          dense_rank() OVER (PARTITION BY e.object_id, e.t0, g.clean
                                             ORDER BY g.channel_id) AS chc
                   FROM e JOIN g ON g.object_id = e.object_id AND g.ts BETWEEN e.t0 AND e.t1)
        SELECT object_id, t0, any_value(noise) IS NULL AS real_, any_value(confirmed) AS conf,
               min(CASE WHEN k = 1 THEN ts END) AS t_first,
               min(CASE WHEN k = 2 THEN ts END) AS t_n2,
               min(CASE WHEN ch = 2 THEN ts END) AS t_nch2,
               min(CASE WHEN clean AND kc = 1 THEN ts END) AS t_clean,
               min(CASE WHEN clean AND kc = 2 THEN ts END) AS t_clean2,
               min(CASE WHEN clean AND chc = 2 THEN ts END) AS t_cleanch2,
               -- Н7/Н8 проверяются на первом триггере эпизода и только на нём. Объявление
               -- задержано на {wait} минут не ради «тишины», а потому что Н8 (массовый дым)
               -- считается по ±10 минутам вокруг триггера: раньше этого срока признак физически
               -- не вычислим. Ноль ложных у этого правила — следствие того, как построена
               -- разметка, а не измерение; что он значит и чего не значит — в разделе 18
               CASE WHEN max(CASE WHEN NOT clean AND ts <= t0 THEN 1 ELSE 0 END) = 0
                    THEN t0 + INTERVAL {wait} MINUTE END AS t_silent
        FROM t GROUP BY object_id, t0""")


def present(a) -> np.ndarray:
    """Булев массив «значение есть».

    duckdb отдаёт маскированные массивы, и оба очевидных способа их прочитать врут. У самого
    маскированного массива `.mean()` и `.sum()` молча пропускают пропуски, так что «пропущено
    настоящих» всегда выходит нулём. А `np.asarray` снимает маску, не заполняя её: под пропуском
    лежит не NaT, а нулевая дата, и пропуск считается объявлением. Смотреть надо на саму маску.
    """
    ok = ~np.ma.getmaskarray(a)
    data = np.ma.getdata(a)
    return ok & ~np.isnat(data) if data.dtype.kind == 'M' else ok


def delay(t0: np.ndarray, t) -> tuple[float, float]:
    """Задержка объявления в минутах: медиана и 90-й процентиль."""
    have = present(t)
    d = np.array((np.ma.getdata(t)[have] - np.ma.getdata(t0)[have]) / np.timedelta64(1, 'm'),
                 dtype=np.float64)
    return (float(np.median(d)), float(np.percentile(d, 90))) if len(d) else (float('nan'),) * 2


def tradeoff(con, args) -> None:
    """Два канала вместе: чем выше порог прогноза, тем меньше ложных — и тем больше происшествий
    уходит из «с упреждением» в «по факту». Полнота при этом не теряется, теряется только время.

    Это и есть настоящая ручка диспетчера. Пока канал «по факту» не подключён, поднимать порог
    страшно: каждое потерянное происшествие теряется совсем. Когда подключён — потерянное всё равно
    объявляется, просто без форы, и цена подъёма порога измеряется в минутах, а не в пропусках.
    """
    H = args.horizon
    print(f'Порог прогноза меняется, канал «по факту» добирает остальное. Правило канала выбрано '
          f'по типу: сначала ноль ложных объявлений, при равенстве — большая полнота. '
          f'Тест {args.year}, прогон {args.run}'
          + (f', склейка дребезга {args.gap} ч' if args.gap else '')
          + (f', сглаживание оценки {args.smooth} ч' if args.smooth > 1 else '') + '.\n')
    print('| тип | правило «по факту» | порог | сигналов прогноза | из них ложных | '
          'с упреждением | медиана форы, ч | по факту | не объявлено | ложных «по факту» |')
    print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        d = episodes(con, tp, (args.year,)).fetchnumpy()
        if not len(d['t0']):
            continue
        real = np.ma.getdata(d['real_']).astype(bool)
        # правило выбирается по типу: `first` полнее, но у пожара и подтопления шумит, а фильтр
        # шума у оборудования наоборот вычёркивает 423 настоящих эпизода ни за что
        opts = {r: present(d[f't_{r}']) for r in ('first', 'clean', 'silent')}
        rule = min(opts, key=lambda r: (int((opts[r] & ~real).sum()),
                                        -int(opts[r][real].sum())))
        said = opts[rule]
        cover = float(said[real].sum()) / max(int(real.sum()), 1)
        bad = int((said & ~real).sum())
        ov, hv, nv, pv = op.split(args.run, 'val', 2025, tp, args.model)
        obj, h, ns, ps = op.split(args.run, 'test', args.year, tp, args.model)
        # сглаживание применяется и к проверке, и к тесту: порог снимается уже на сглаженной
        # оценке, иначе он был бы снят не с той величины, к которой его прикладывают
        pv = persist.smooth_rows(ov, hv, pv, args.smooth)
        ps = persist.smooth_rows(obj, h, ps, args.smooth)
        y = (ns <= H).astype(np.int8)
        thr0 = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        above = ps[ps >= thr0]
        name = config.TYPE_NAMES[tp]
        for tag, t in [('рабочий', thr0)] + [
                (f'выше, q{int(q * 100)}', float(np.quantile(above, q)) if len(above) else 1.1)
                for q in (0.25, 0.5, 0.75, 0.9)]:
            m = metrics.evaluate(obj, h, ns, ps, t, H, metrics.RUN_CAP)
            sig, true = metrics.signals(obj, h, y, ps >= t, args.gap)
            lost = m['episodes'] - m['caught']
            byfact = int(round(lost * cover))
            print(f'| {name} | {rule} | {tag} | {sig} | {sig - true} | {m["caught"]} | '
                  f'{m["lead_median_h"]:.0f} | {byfact} | {lost - byfact} | {bad} |')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='main_h24_tuned')
    ap.add_argument('--model', default='xgb')
    ap.add_argument('--types', default=','.join(config.TYPES))
    ap.add_argument('--year', type=int, default=2026)
    ap.add_argument('--horizon', type=int, default=config.HORIZON)
    ap.add_argument('--mode', default='rules', choices=['rules', 'tradeoff'])
    ap.add_argument('--smooth', type=int, default=0,
                    help='окно сглаживания оценки в часах (раздел 29); 0 — сырая оценка')
    ap.add_argument('--gap', type=int, default=0,
                    help='склейка дребезга: повтор на том же объекте в пределах gap '
                         'часов — продолжение прежней тревоги, а не новая (раздел 27)')
    args = ap.parse_args()
    H = args.horizon
    con = duckdb.connect(str(config.WORK / 'tf.duckdb'), read_only=True)
    if args.mode == 'tradeoff':
        tradeoff(con, args)
        con.close()
        return

    print(f'Канал «по факту» на {args.year}. Объявление — по первым событиям журнала, упреждения '
          f'нет, считается задержка. Ложное — объявление по эпизоду, который разметка признала '
          f'шумом.\n')
    print('| тип | правило | объявлено | из них ложных | пропущено настоящих | задержка, мин | '
          'она же 90% | за объявлением был выезд |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    miss = {}
    for tp in args.types.split(','):
        d = episodes(con, tp, (args.year,)).fetchnumpy()
        if not len(d['t0']):
            continue
        real = np.ma.getdata(d['real_']).astype(bool)
        conf = np.ma.getdata(d['conf']).astype(bool)
        name = config.TYPE_NAMES[tp]
        for rule in RULES:
            t = d[f't_{rule}']
            said = present(t)
            said_real, said_bad = said & real, said & ~real
            med, p90 = delay(d['t0'][said_real], t[said_real])
            visit = float(conf[said].mean()) if said.any() else float('nan')
            print((f'| {name} | {rule} | {int(said.sum())} | {int(said_bad.sum())} | '
                   f'{int((real & ~said).sum())} | {med:.1f} | {p90:.1f} | {visit:.3f} |'
                   ).replace('.', ','))
        miss[tp] = (present(d['t_silent'])[real],)

    if args.year != config.VAL_END.year:
        # на проверочном годе прогноза теста нет: таблица правил выше считается по журналу и
        # от прогона не зависит — её и просили, когда задают другой год
        con.close()
        return

    print('\nЧто канал «по факту» добирает за прогнозом (правило first, порог прогноза по лучшему '
          f'F1 на проверке, прогон {args.run}):\n')
    print('| тип | эпизодов | поймал прогноз | пропустил прогноз | из них объявит «по факту» |')
    print('|---|---:|---:|---:|---:|')
    for tp in args.types.split(','):
        if tp not in miss:
            continue
        _, _, nv, pv = op.split(args.run, 'val', 2025, tp, args.model)
        obj, h, ns, ps = op.split(args.run, 'test', args.year, tp, args.model)
        thr = metrics.best_threshold((nv <= H).astype(np.int8), pv)
        m = metrics.evaluate(obj, h, ns, ps, thr, H, metrics.RUN_CAP)
        (tf,) = miss[tp]
        lost = m['episodes'] - m['caught']
        covered = int(tf.sum() / max(len(tf), 1) * lost)   # доля эпизодов с чистым триггером
        print(f"| {config.TYPE_NAMES[tp]} | {m['episodes']} | {m['caught']} | {lost} | {covered} |")
    con.close()


if __name__ == '__main__':
    main()
