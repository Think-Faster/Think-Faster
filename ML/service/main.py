"""Запуск сервиса модели (INTEGRATION §7, §13.1). Сам такт, команды и чтение — в `core.Service`.

Режимы:
- `--loop` — контур: четыре потока одного процесса — такт по `TF_MODEL_CLOCK` (clock.py), приём
  `tf.ingest.*` в горячий журнал (ingest.consume), команды `tf.model.commands` (commands.py) и
  HTTP-ручки (api.py). Остановка по SIGTERM/SIGINT: текущий такт или команда дописываются до конца.
- `--tick-now ГГГГ-ММ-ДДTЧЧ:ММ` — один такт на границу часа (стенд, проверка поставки).
- `--replay ГГГГ-ММ-ДДTЧЧ:ММ --days N` — такт за тактом N суток подряд без ожидания (стенд).
- `--import` — залить журнал из csv в пустой горячий журнал (стенд без Kafka).

Выход прогнозов — Kafka `tf.forecast.results`; `TF_MODEL_SINK=file` (или `TF_ENV=dev` без Kafka) —
файл results.ndjson на томе. `TF_KAFKA_BOOTSTRAP=off` и `TF_RABBIT_URL=off` выключают приём и
команды на стенде.
"""
import argparse
import json
import logging
import os
import signal
import threading
from datetime import timedelta

import svc as config
import outbox
from clock import Clock
from core import Service, to_msk

log = logging.getLogger('tf-model')


def make_sink():
    kind = os.environ.get('TF_MODEL_SINK') or ('kafka' if config.ENV != 'dev' else 'file')
    inner = outbox.KafkaSink() if kind == 'kafka' else outbox.FileSink()
    return outbox.LastSink(inner)


def start_ingest(service: Service, stop: threading.Event):
    if config.KAFKA_BOOTSTRAP == 'off':
        log.warning('приём Kafka выключен (TF_KAFKA_BOOTSTRAP=off): горячий журнал не пополняется')
        return None
    from ingest import KafkaReader, consume
    dead = service.sink.inner.dead if isinstance(getattr(service.sink, 'inner', None), outbox.KafkaSink) else None
    stats = service.ingest_stats = {}

    def work():
        while not stop.is_set():
            try:
                with KafkaReader() as reader:
                    consume(service.ingest_store(), reader, stop, dead=dead, stats=stats)
            except Exception:
                log.exception('приём Kafka упал — снова через 30 с')
                stop.wait(30)

    t = threading.Thread(target=work, name='ingest', daemon=True)
    t.start()
    return t


def loop() -> None:
    import api as apimod
    import commands
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    clock = Clock()
    service = Service(sink=make_sink(), clock=clock)
    first = clock.last + timedelta(hours=1) if clock.last else \
        clock.start if clock.mode == 'replay' else config.now_msk().replace(minute=0, second=0, microsecond=0)
    service.boot(first)
    apimod.start_api(service, apimod.make_verifier())
    start_ingest(service, stop)
    if config.RABBIT_URL != 'off':
        commands.start(service, stop)
    else:
        log.warning('команды выключены (TF_RABBIT_URL=off)')
    log.info('сервис: часы %s, API :%d', clock.label, config.API_PORT)
    service.run(stop)
    log.info('остановлен')


def main() -> None:
    ap = argparse.ArgumentParser(description='Сервис модели Think Faster')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--loop', action='store_true', help='контур: такт, приём, команды, API')
    mode.add_argument('--tick-now', metavar='ВРЕМЯ', help='один такт на границу часа')
    mode.add_argument('--replay', metavar='НАЧАЛО', help='такты подряд от границы часа')
    mode.add_argument('--import', dest='imp', action='store_true', help='журнал из csv в пустой горячий журнал')
    ap.add_argument('--days', type=int, default=7, help='сколько суток для --replay')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(threadName)s %(message)s')

    if args.loop:
        return loop()
    if args.imp:
        import storage
        store = storage.HotStore()
        if store.con.sql('SELECT count(*) FROM obj').fetchone()[0] == 0:
            store.import_reference()
        n = store.con.sql('SELECT count(*) FROM ev_all').fetchone()[0]
        print(f'в журнале уже {n} строк' if n else f'залито {store.bulk_import()} строк')
        return
    t = to_msk(args.tick_now or args.replay).replace(minute=0, second=0, microsecond=0)
    service = Service(sink=make_sink())
    service.boot(t)
    end = t + timedelta(days=args.days) if args.replay else t + timedelta(hours=1)
    while t < end:
        res = service.tick(t)
        print(json.dumps(res, ensure_ascii=False, default=str))
        t += timedelta(hours=1)


if __name__ == '__main__':
    main()
