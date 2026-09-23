"""Такт сервиса (INTEGRATION §7): час за часом — признаки, шесть типов, порог, правила, сообщение.

Порядок такта ровно из §7:

1. потребитель дописал события часа в горячий журнал (M1/M2);
2. сборка признаков парка и входа сети (M3) — около 3 с;
3. шесть типов (M4), история оценок и скользящий порог (M5);
4. правила: склейка, отклонение, mute, факт-канал (M6/M7/M8);
5. сообщение на объект в tf.forecast.results; DLQ — вне часового потока;
6. наблюдение и гидренизация журнала (M9/M10).

В `--replay`/`--loop`-режимах источник — ReplayReader (файлы) либо KafkaReader.
"""
import argparse
import json
import logging
import time
from datetime import datetime, timedelta

import numpy as np

import svc as config
import fact as factmod
import observer as observ
import outbox
import predict as predmod
import rules as rulesmod
import snapshot as snapmod
import storage
import threshold as thrmod
from settings import OperatingSettings

log = logging.getLogger('tf-model')


def hour_end_iso(h: int) -> str:
    import features as ft
    t = ft.T0 + timedelta(hours=h + 1)
    return t.isoformat(sep='T', timespec='minutes') + config.TZ


def tick(store, predictor, history, rules, settings, sink, observer, now: datetime,
         meta: dict | None = None) -> dict:
    t0 = time.time()
    meta = meta or predmod.load_manifest()
    import features as ft
    hi = thrmod.hour_index(now)

    # 2+3: витрина (labels.build внутри retro.snapshot) и вход сети
    cal = ft.calendar()
    frame = snapmod.snapshot(store, now, meta, cal)
    seq_in = snapmod.seq(store, now, meta)
    log.info('признаки %d объектов за %.1f с', frame.height, time.time() - t0)

    # 4: шесть типов
    scores = predictor.predict(frame, seq_in)
    objects = frame['object_id'].to_numpy()

    # 5: история и порог
    history.update(hi, scores)
    thresholds = {}
    for tp in config.TYPES:
        thr = history.threshold_override.get(tp)
        thresholds[tp] = thr if thr is not None else history.threshold(tp)

    # 6a: факт-канал — разметку уже построил snapshot, второй раз не строим
    facts = factmod.detect(store, now, need_build=False)
    rules.on_fact(facts, hi)

    # 6b: правила (склейка/отклонение/mute)
    applied = rules.apply(scores, objects, hi, thresholds)

    # 7:сообщения
    model_version = meta.get('exported', '')
    for i, oid in enumerate(objects):
        msg = outbox.build_message(int(oid), hour_end_iso(hi - 1), model_version, config.TYPES)
        for tp in config.TYPES:
            alarm, since = applied[tp][0][i], applied[tp][1][i]
            reasons = predictor.reasons(frame, i, tp, k=3) if alarm else None
            fill(msg, tp, scores[tp][i], thresholds[tp], alarm, since, reasons)
        sink.send(msg)

    # 8: наблюдение и обслуживание
    fres = snapmod.freshness(store, now)
    stale = any(fres.get(tp, 0) >= config.STALE_HOURS.get(tp, config.STALE_HOURS['default'])
                for tp in config.TYPES)
    summary = observer.observe(alarms={tp: applied[tp][0] for tp in config.TYPES},
                               history={tp: history.window(tp) for tp in config.TYPES},
                               freshness=fres, stale=stale, dlq=0, now=now)
    observer.dump()
    if now.hour == 0:
        history.save()
    sweep = store.retention_sweep(now)
    return {'seconds': round(time.time() - t0, 1), 'objects': frame.height,
            'alarms': int(sum(int(x[0].sum()) for x in applied.values())),
            'facts': len(facts), 'stale': stale, 'sweep': sweep}


def fill(msg, tp, score, thr, alarm, since, reasons):
    outbox.fill_type(msg, tp, score=score, threshold=thr, alarm=alarm, since_hours=since,
                     reasons=reasons)


def pull_decisions(rules, kafka: bool = False) -> int:
    """Решения диспетчера из parquet-журнала (стенд) или Kafka (прод, M11)."""
    import features as ft
    rows = []
    if kafka:
        from ingest import KafkaReader
        r = KafkaReader([config.TOPIC_DECISIONS])
        for i in range(3):
            try:
                rows.append(next(r.poll()))
            except StopIteration:
                break
        r.close()
    else:
        p = config.DECISIONS_LOG
        if p.exists():
            import polars as pl
            rows = pl.read_parquet(p).iter_rows(named=True)
    n = 0
    for d in rows:
        rules.on_decision(d['object_id'], d['type'], d['action'],
                          thrmod.hour_index(d['ts'] if isinstance(d.get('ts'), datetime)
                                            else datetime.fromisoformat(d['ts'])))
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tick-now', default='', help='разовый расчёт на момент времени (стенд)')
    ap.add_argument('--loop', action='store_true', help='часовой цикл с читателем Kafka')
    ap.add_argument('--replay', default='', help='проиграть журнал из csv-файлов, час за часом')
    ap.add_argument('--days', type=int, default=14, help='сколько суток проиграть')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    settings = OperatingSettings.load()
    store = storage.HotStore()
    if store.con.sql('SELECT count(*) FROM ch').fetchone()[0] == 0:
        store.import_reference()
    predictor = predmod.Predictor()
    history = thrmod.ScoreHistory(settings)
    rules = rulesmod.RuleState(settings)
    sink = outbox.FileSink()

    if args.loop:
        from ingest import KafkaReader
        reader = KafkaReader()
        log.info('цикл: потребление %s, такт каждый час', config.TOPIC_READINGS)
        last = None
        while True:
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            if last != now:
                pull_decisions(rules, kafka=True)
                tick(store, predictor, history, rules, settings, sink,
                     observ.Observer(), now)
                last = now
            time.sleep(30)
    elif args.replay:
        from glob import glob
        from ingest import ReplayReader, pull
        files = [f for f in glob(str(config.JOURNAL / 'ext-journal-*.csv')) if '2021' not in f]
        pull(store, ReplayReader(files))
        log.info('проиграно журнала в горячий журнал')
        t = datetime(2026, 1, 1, 7)
        end = t + timedelta(days=args.days)
        obs = observ.Observer()
        while t < end:
            tick(store, predictor, history, rules, settings, sink, obs, t)
            t += timedelta(hours=1)
    else:
        now = datetime.fromisoformat(args.tick_now) if args.tick_now else \
            datetime.now().replace(minute=0, second=0, microsecond=0)
        res = tick(store, predictor, history, rules, settings, sink,
                   observ.Observer(), now)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()