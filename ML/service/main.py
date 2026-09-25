"""Такт сервиса (INTEGRATION §7): час за часом — признаки, шесть типов, порог, правила, сообщение.

Порядок такта ровно из §7 — но по интерфейсам §10.1:

1. потребитель дописал события часа в горячий журнал (M1/M2); в `--loop` приём живёт отдельным
   потоком с явным коммитом смещений (П3);
2. сборка признаков парка и входа сети (M3) — `retro.snapshot()` из main: сам строит вид `ev` и
   разметку labels.build, сервис свои копии не держит (П2);
3. шесть типов (M4): смесь зёрен из манифеста, шкала 2025 — `searchsorted/len` (П1);
4. история оценок и скользящий порог (M5) — `retro.rolling()` main, никого override (П7);
   на смену operating.json порог пересчитывается с места (share → квантиль);
5. правила: склейка дребезга (П6), отклонение `quantile(1-share·k)` (П5), mute, факт-канал;
6. сообщение на объект в tf.forecast.results; DLQ — вне часового потока;
7. наблюдение, гидренизация журнала, статус переобучения.

Режимы: `--loop` (прод: Kafka-поток + API + такт), `--replay N` (стенд: файлы журнала, N суток),
`--tick-now` (разовый расчёт).
"""
import argparse
import json
import logging
import time
from datetime import datetime, timedelta

import numpy as np

import svc as config
import advisor as ad
import assurance as confmod
import decisions
import fact as factmod
import observer as observ
import outbox
import planmute
import predict as predmod
import recommend as recmod
import rules as rulesmod
import snapshot as snapmod
import storage
import threshold as thrmod
from settings import OperatingSettings

log = logging.getLogger('tf-model')


def hour_iso(h: int) -> str:
    import features as ft
    return (ft.T0 + timedelta(hours=h + 1)).isoformat(sep='T', timespec='seconds') + config.TZ


def hour_index(ts: datetime) -> int:
    import features as ft
    return int((ts - ft.T0).total_seconds() // 3600)


def bootstrap(history: thrmod.Thresholds, predictor: predmod.Predictor,
              settings: OperatingSettings, h: int) -> None:
    """История порога (П1): если в parquet есть прошлый прогон той же версии — поднимаем его,
    иначе (нет файла либо история собрана другой выгрузкой) пересчитываем на текущей."""
    if not history.load(settings, h, model_version=predictor.version):
        history.bootstrap(predictor.bootstrap_history(year=2025), settings, h,
                          model_version=predictor.version)
        history.dump()


def tick(store, predictor, history, rules, settings, sink, observer, now: datetime,
         meta: dict | None = None) -> dict:
    t0 = time.time()
    meta = meta or predictor.meta
    import features as ft
    h = hour_index(now) - 1                      # строка витрины h = hi − 1 (как retro.snapshot)

    # новое в operating.json → пороги (share-квантиль) пересчитываются на месте
    history.apply_settings(settings, h)

    # 2+3: витрина и вход сети; разметка (labels.build) — внутри snapshot, факты берём без сбopки
    cal = ft.calendar()
    frame, seq_in = snapmod.snapshot(store, now, meta, cal, seq=predictor.has_nets())
    log.info('признаки %d объектов за %.1f с', frame.height, time.time() - t0)

    # 4: шесть типов (смесь зёрен на шкале 2025)
    scores = predictor.predict(frame, seq_in)
    objects = frame['object_id'].to_numpy()

    # 5: история и порог (retro.rolling: квантиль 90 суток строго до часа)
    history.extend(h, scores, settings)
    thresholds = history.thresholds

    # 6a: факт-канал — разметку numerator построил snapshot, второй раз не строим
    facts = factmod.detect(store, now, need_build=False)
    rules.on_fact(facts, h)

    # 6b: правила (склейка 6 ч / отклонение / mute)
    rules.history = {tp: history.history(tp) for tp in config.TYPES}
    applied = rules.apply(scores, objects, h, thresholds)

    # 7: сообщения
    model_version = predictor.version
    coll = dict(store.con.sql('SELECT object_id, collector_id FROM obj3').fetchall())
    al_by = {tp: {int(oid) for i, oid in enumerate(objects) if applied[tp][0][i]} for tp in config.TYPES}
    collectors_by_tp = {tp: {coll[obj] for obj in s if obj in coll} for tp, s in al_by.items()}

    # M7: прогноз типа в окне графика работ — статус MUTED (INTEGRATION §1.6), в тревогу не идёт
    mutes = {}
    if planmute.windows(store.con, now.year):
        for tp in config.TYPES:
            m = planmute.mask(store.con, tp, now)
            if m:
                mutes[tp] = m

    # §7: свежесть данных по типам — флаг «данные несвежие N часов» едет в каждое сообщение
    fres = snapmod.freshness(store, now)

    for i, oid in enumerate(objects):
        msg = outbox.build_message(int(oid), hour_iso(h), model_version, config.TYPES)
        msg['data_freshness_hours'] = {tp: round(float(fres.get(tp, 0.0)), 1) for tp in config.TYPES}
        comp = []
        for tp in config.TYPES:
            alarm, since = applied[tp][0][i], applied[tp][1][i]
            w_id = mutes.get(tp, {}).get(int(oid))
            if alarm and w_id:
                fill(msg, tp, scores[tp][i], float(thresholds[tp]), False,
                     muted={'work_id': w_id, 'reason': planmute.REASON})
                continue
            if not alarm:
                fill(msg, tp, scores[tp][i], float(thresholds[tp]), False, 0, None)
                continue
            reasons = predictor.reasons(frame, i, tp, k=3)
            evidence = snapmod.evidence(store, int(oid), tp, now)
            channels = len({e['sensor_id'] for e in evidence})
            neighbors = sum(1 for o in config.TYPES if o != tp and bool(applied[o][0][i]))
            conf = confmod.assurance(int(since), neighbors, channels)
            line_n = 0
            if tp == 'equipment' and oid in coll:
                same_col = collectors_by_tp[tp] & {coll[oid]}
                line_n = len({o for o in al_by[tp] if coll.get(o) in same_col}) - 1
            rec = advisor.build(store.con, int(oid), tp, now,
                                reasons=[r['feature'] for r in reasons],
                                since_h=int(since), co_types=[t for t in config.TYPES if t != tp
                                                              and applied[t][0][i]], line_n=line_n)
            fill(msg, tp, scores[tp][i], float(thresholds[tp]), True, int(since), reasons,
                 conf, evidence, recommendation=rec)
            comp.append(rec)
        if len(comp) >= 2:
            outbox.object_recommendation(msg, recmod.compose(comp))
        sink.send(msg)

    # 8: наблюдение и обслуживание
    stale = any(fres.get(tp, 0) >= config.STALE_HOURS.get(tp, config.STALE_HOURS['default'])
                for tp in config.TYPES)
    summary = observer.observe(alarms={tp: applied[tp][0] for tp in config.TYPES},
                               history={tp: history.history(tp)[1] for tp in config.TYPES},
                               muted={tp: sum(1 for o in mutes.get(tp, {}) if o in al_by[tp])
                                      for tp in config.TYPES},
                               freshness=fres, stale=stale, dlq=0, now=now)
    observer.dump()
    if now.hour == 0:
        history.dump()
    sweep = store.retention_sweep(now)
    return {'seconds': round(time.time() - t0, 1), 'objects': frame.height,
            'alarms': int(sum(int(al[0].sum()) for al in applied.values())),
            'muted': int(sum(len(v) for v in mutes.values())),
            'facts': len(facts), 'stale': stale, 'sweep': sweep}


def fill(msg, tp, score, thr, alarm, since, reasons, confidence=None, evidence=None,
         recommendation=None, muted=None):
    outbox.fill_type(msg, tp, score=score, threshold=thr, alarm=alarm, since_hours=since,
                     reasons=reasons, confidence=confidence, evidence=evidence,
                     recommendation=recommendation, muted=muted)


def pull_decisions(rules, kafka: bool = False) -> int:
    """Решения диспетчера (M11): Kafka tf.dispatch.decisions (прод, П4) либо parquet (стенд)."""
    if kafka:
        from ingest import KafkaReader
        with KafkaReader([config.TOPIC_DECISIONS], group=config.KAFKA_GROUP_DECISIONS) as r:
            decoded = []
            for i in range(64):
                msg = r.poll_readings(timeout_ms=200, commit=False)
                if msg is None:
                    break
                try:
                    d = json.loads(msg.value().decode())
                except (ValueError, UnicodeDecodeError):
                    continue
                if 'object_id' in d and 'action' in d:
                    decoded.append(d)
            r.commit()
            return decisions.apply(rules, decoded)
    p = config.DECISIONS_LOG
    if not p.exists():
        return 0
    import polars as pl
    return decisions.apply(rules, pl.read_parquet(p).rows(named=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tick-now', default='', help='разовый расчёт на момент времени (стенд)')
    ap.add_argument('--loop', action='store_true', help='прод: Kafka + API + часовой такт')
    ap.add_argument('--replay', default='', help='стенд: проиграть журнал из csv, час за часом')
    ap.add_argument('--days', type=int, default=14, help='сколько суток проиграть')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    settings = OperatingSettings.load()
    store = storage.HotStore()
    if store.con.sql('SELECT count(*) FROM obj').fetchone()[0] == 0:
        store.import_reference()
    predictor = predmod.Predictor()
    if store.con.sql('SELECT count(*) FROM ev_all').fetchone()[0] == 0:
        store.bulk_import()
    history = thrmod.Thresholds()
    rules = rulesmod.RuleState(settings, history={tp: history.history(tp) for tp in config.TYPES})
    sink = outbox.FileSink()

    if args.loop:
        import threading
        import api as apimod
        from ingest import KafkaReader, pull
        state = apimod.State(store, predictor, history, rules, settings)
        apimod.start_api(state)             # П8: uvicorn из процесса сервиса, JWT на ручках
        reader = KafkaReader()
        thread = threading.Thread(target=pull, args=(store, reader, state.apply_settings), daemon=True)
        thread.start()
        log.info('цикл: поток приёма %s, такт каждый час, API :%d', config.TOPIC_READINGS,
                 config.API_PORT)
        pulls = 0
        while True:
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            if pulls == 0 or now.hour == 0:
                bootstrap(history, predictor, settings, hour_index(now) - 1)
                pulls += 1
            if now.hour == 0:
                from retrainer import Retrainer
                st = Retrainer().status()
                if st.get('status') == 'done' and st.get('version') != predictor.version:
                    from retrainer import apply_version
                    predictor = predmod.Predictor()
                    apply_version(history, predictor, hour_index(now) - 1)
            pull_decisions(rules, kafka=True)
            settings = OperatingSettings.load()          # могли сменить в админ-панели (П7)
            rules.settings = settings                    # доля/k для thr_mute — со свежего файла (П5)
            tick(store, predictor, history, rules, settings, sink, observ.Observer(), now)
            time.sleep(30)
    elif args.replay:
        from glob import glob
        from ingest import ReplayReader, pull
        files = [f for f in glob(str(config.JOURNAL / 'ext-journal-*.csv')) if '2021' not in f]
        pull(store, ReplayReader(files))
        log.info('проиграно журнала в горячий журнал: %d строк', store.con.sql(
            'SELECT count(*) FROM ev_all').fetchone()[0])
        t = datetime(2026, 1, 1, 7)
        end = t + timedelta(days=args.days)
        bootstrap(history, predictor, settings, hour_index(t) - 1)
        obs = observ.Observer()
        while t < end:
            tick(store, predictor, history, rules, settings, sink, obs, t)
            t += timedelta(hours=1)
    else:
        now = datetime.fromisoformat(args.tick_now) if args.tick_now else \
            datetime.now().replace(minute=0, second=0, microsecond=0)
        bootstrap(history, predictor, settings, hour_index(now) - 1)
        res = tick(store, predictor, history, rules, settings, sink, observ.Observer(), now)
        print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()