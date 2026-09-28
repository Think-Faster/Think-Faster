"""Ядро сервиса tf-model: такт, команды и чтение для ручек в одном объекте (INTEGRATION §13.1).

Потоки процесса (13.1): приём Kafka дописывает горячий журнал своим курсором; такт раз в час по
часам `clock.py`; потребитель команд RabbitMQ (`commands.py`); HTTP (`api.py`). Такт и команды
меняют одно состояние — рабочие настройки, график работ, игнорируемые периоды, решения диспетчера,
версии моделей, — поэтому идут под одним замком: команда посреди такта ждёт его конца и действует
со следующего часа. Ручки читают готовое (последнее сообщение по объекту, журнал прогнозов своим
курсором, пороги) и замка не ждут.

Команда подтверждается только после записи состояния на том (13.3). Всё, что меняет решения людей
или настройки, и всё, что модель сказала, но смена не увидела (молчание по графику, отклонение),
пишется в аудит (13.4) и в журнал прогнозов — историю главного диспетчера (§9.2).
"""
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import svc as config
import fact as factmod
import gaps as gapsmod
import outbox
import works as worksmod
from rules import RuleState
from settings import OperatingSettings, _validate, bounds as settings_bounds

log = logging.getLogger('tf-model')

SEEN_KEEP = 50_000                  # сколько command_id помнить для идемпотентности
RETRAIN_ENABLED = os.environ.get('TF_MODEL_RETRAIN', 'off') == 'on'   # §9.5: в контуре без журнала — нет


class CommandError(ValueError):
    """Сломанная команда (13.3): reject без повтора, сообщение уходит в tf.dlq."""


# ----- часы -------------------------------------------------------------------------------------
def hour_index(ts: datetime) -> int:
    import features as ft
    return int((ts - ft.T0).total_seconds() // 3600)


def hour_iso(h: int) -> str:
    """Конец часа строки h в том виде, что в сообщении §2.3."""
    import features as ft
    return (ft.T0 + timedelta(hours=h + 1)).isoformat(sep='T', timespec='minutes') + config.TZ


def _iso(ts: datetime, spec: str = 'minutes') -> str:
    """Момент журнала (московское время без зоны) в ISO с зоной для сообщений."""
    return ts.isoformat(timespec=spec) + config.TZ


def to_msk(s) -> datetime:
    """Время из команды: с зоной — переводится в MSK, без зоны — уже местное время журнала."""
    t = s if isinstance(s, datetime) else datetime.fromisoformat(str(s).replace('Z', '+00:00'))
    return t.astimezone(config.MSK).replace(tzinfo=None) if t.tzinfo else t


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def _load_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def make_audit():
    import tfkit
    # на стенде без Redis (dev) события копятся в файле на томе, а не ждут таймаута на каждом
    url = os.environ.get('TF_REDIS_URL', '' if config.ENV == 'dev' else config.REDIS_URL)
    return tfkit.Audit('ml', redis_url=url, spool=config.AUDIT_SPOOL)   # имя сервиса — как в §6.4


class _IngestStore:
    """Горячий журнал для потока приёма: своя копия соединения DuckDB (одно соединение из двух
    потоков небезопасно), справочник каналов — общий, прогретый до старта потока."""

    def __init__(self, store):
        self.store, self.con = store, store.con.cursor()

    def clean_event(self, channel_id, ts, value):
        return self.store.clean_event(channel_id, ts, value)

    def append(self, rows: list[dict]) -> int:
        return self.store.append(rows, con=self.con)

    def apply_reference(self, d: dict) -> str:
        return self.store.apply_reference(d, con=self.con)


class Service:
    def __init__(self, *, settings_dir: Path | None = None, seed: Path | None = None, audit=None,
                 sink=None, clock=None):
        self.lock = threading.RLock()
        self.dir = Path(settings_dir or config.SETTINGS_DIR)
        self.seed = Path(seed or config.SEED_SETTINGS)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.dir / 'operating.json'
        if not self.settings_path.exists():
            shutil.copyfile(self.seed / 'operating.json', self.settings_path)
        self.settings = OperatingSettings.load(self.settings_path)
        self.works = worksmod.Works(self.dir, seed=self.seed / worksmod.NAME)
        self.works.point_labels()
        self.gaps = gapsmod.Gaps(self.dir)
        self.gaps.apply()
        self._point_recommend()
        self.state = _load_json(config.SERVICE_STATE, {})
        self.state.setdefault('versions', dict(config.DEFAULT_VERSIONS))
        self.state.setdefault('retrain_needed', False)
        self.state.setdefault('works_open', {})
        self.rules = RuleState(self.settings)
        self.rules.load(config.RULES_STATE)
        self.seen = dict.fromkeys(_load_json(config.COMMANDS_SEEN, []))
        self.audit = audit if audit is not None else make_audit()
        self.sink = sink
        self.clock = clock
        self.store = self.predictor = self.history = self.fclog = self.observer = None
        self.last_now: datetime | None = None     # граница последнего посчитанного часа
        self.last: dict = {}                      # сводка последнего такта
        self.freshness: dict = {}
        self.ingest_stats: dict | None = None  # счётчики приёма Kafka (main.start_ingest)
        self._cal = None
        self._collectors: dict[int, int] | None = None
        self._retrain_thread: threading.Thread | None = None

    def _point_recommend(self) -> None:
        """Словарь мер и повторы (M13) — из поставки рядом с моделями, а не из репозитория."""
        import recommend
        for name, attr in (('recommendations.csv', 'RULES'), ('recurrence.csv', 'RECUR')):
            if (self.seed / name).exists():
                setattr(recommend, attr, self.seed / name)

    # ----- старт ----------------------------------------------------------------------------------
    def boot(self, now: datetime, store=None, predictor=None) -> None:
        """Горячий журнал, модели нужных версий, история порога, журнал прогнозов."""
        import storage
        import threshold as thrmod
        import observer as observ
        from fclog import ForecastLog
        self.store = store or storage.HotStore()
        if self.store.con.sql('SELECT count(*) FROM obj').fetchone()[0] == 0:
            self.store.import_reference()
        self.store._channels()
        self.predictor = predictor or self._predictor()
        self.history = thrmod.Thresholds()
        self.fclog = ForecastLog()
        self.observer = observ.Observer()
        if self.sink is None:
            self.sink = outbox.LastSink(outbox.FileSink())
        self.last_now = self.clock.last if self.clock is not None and self.clock.last else None
        self._bootstrap(hour_index(now) - 1)

    def _predictor(self):
        import predict as predmod
        p = predmod.Predictor()
        for tp, n in list(self.state['versions'].items()):
            try:
                p.use_version(tp, n)
            except (FileNotFoundError, ValueError) as e:
                log.warning('версия %s v%s не поднялась (%s) — тип считает основная выгрузка', tp, n, e)
                self.state['versions'].pop(tp)
        return p

    def _bootstrap(self, h: int) -> None:
        """История порога (П1): сохранённая той же версии либо смесь по витрине 2025."""
        if not self.history.load(self.settings, h, model_version=self.predictor.version):
            self.history.bootstrap(self.predictor.bootstrap_history(year=2025), self.settings, h,
                                   model_version=self.predictor.version)
        self.history.drop_hours(self._gap_hours())
        self.history.dump()

    def _gap_hours(self) -> list[tuple[int, int]]:
        return [(hour_index(a), hour_index(b)) for a, b in self.gaps.intervals()]

    def collectors(self) -> dict[int, int]:
        if self._collectors is None:
            self._collectors = {int(o): (None if c is None else int(c)) for o, c in self.store.con.sql(
                'SELECT object_id, parent_id FROM obj WHERE level = 3').fetchall()}
        return self._collectors

    def ingest_store(self) -> _IngestStore:
        return _IngestStore(self.store)

    # ----- такт -----------------------------------------------------------------------------------
    def tick(self, now: datetime) -> dict:
        """Час, закончившийся в `now`: признаки, шесть типов, порог, правила, M7, факты, M13, выход."""
        import features as ft
        import snapshot as snapmod
        t0 = time.time()
        with self.lock:
            h = hour_index(now) - 1                  # строка витрины h = hi − 1 (как retro.snapshot)
            ts = now - timedelta(hours=1)            # её час: так окна графика считает maintenance.in_works
            st = self.settings
            self.history.apply_settings(st, h)
            self.rules.settings = st
            need = hour_index(now) + 1
            if self._cal is None or len(self._cal['hour']) <= need:   # живой такт — после DATA_END
                self._cal = ft.calendar(max(ft.NH, need + 366 * 24))
            frame, seq_in = snapmod.snapshot(self.store, now, self.predictor.meta, self._cal,
                                             seq=self.predictor.has_nets(), holes=self.gaps.intervals())
            t_feat = time.time() - t0
            scores = self.predictor.predict(frame, seq_in)
            objects = frame['object_id'].to_numpy()
            if self.gaps.covers(ts):
                self.history.refresh(h, st)          # брак данных — час не идёт в историю порога
            else:
                self.history.extend(h, scores, st)
            thr = {tp: float(self.history.thresholds[tp]) for tp in config.TYPES}
            coll = self.collectors()

            # факты часа: чистые снимают отклонение и молчание, Н10 — с пометкой или в историю
            eps = factmod.episodes(self.store, now)
            clean = {(e['object_id'], e['type']) for e in eps if e['noise'] is None and e['new']}
            for key, was, ref in self.rules.on_fact(clean, h):
                back = h - int(ref.get('hour', h))
                self._audit_forecast('forecast.recurred', key, h, {
                    'was': was, 'command_id': ref.get('command_id'), 'hours_after_decision': back,
                    'within_horizon': back <= config.HORIZON, 'source': 'fact'})

            # правила (склейка, отклонение, молчание по решению) и M7 — окна графика работ
            self.rules.history = {tp: self.history.history(tp) for tp in config.TYPES}
            applied = self.rules.apply(scores, objects, h, thr)
            muted = self.works.mute(ts, objects, coll)
            statuses, opened = [], {}
            for tp in config.TYPES:
                alarm, _ = applied[tp]
                raw = np.asarray(scores[tp]) >= thr[tp]
                for i in np.flatnonzero(raw | alarm):
                    o = int(objects[i])
                    w = muted.get((o, tp))
                    if w is not None:
                        alarm[i] = False
                        if raw[i]:
                            statuses.append((o, tp, 'MUTED', 'works', w.work_id))
                            k = f'{o}:{tp}'
                            opened[k] = w.work_id
                            if self.state['works_open'].get(k) != w.work_id:
                                self._audit_forecast('forecast.muted', (o, tp), h, {
                                    'reason': 'works', 'work_id': w.work_id, 'work_kind': w.work_kind,
                                    'shifted': w.shifted, 'score': round(float(scores[tp][i]), 4),
                                    'threshold': round(thr[tp], 4)})
                    elif alarm[i]:
                        statuses.append((o, tp, 'ALARM', None, None))
                    else:
                        d = self.rules.status(o, tp, h)
                        if d is not None:
                            statuses.append((o, tp, d[0], 'decision', d[1].get('command_id')))
            # окно графика открыто, пока пара в нём; вышла из окна — следующее молчание снова в аудит
            self.state['works_open'] = {k: v for k, v in {**self.state['works_open'], **opened}.items()
                                        if (int(k.split(':')[0]), k.split(':')[1]) in muted}

            model_version = self.predictor.version
            clock = self.clock.label if self.clock is not None else 'live'
            fres = snapmod.freshness(self.store, now)
            n_alarm = self._emit_forecasts(now, h, frame, objects, scores, thr, applied, model_version, clock,
                                           fres)
            try:
                accs = factmod.accidents(self.store, now)
            except Exception:
                log.exception('аварии и слепота (§13.11) не посчитались')
                accs = []
            n_fact, fact_status = self._emit_facts(now, h, eps, coll, model_version, clock, accs)
            have = {(o, tp) for o, tp, *_ in statuses}
            statuses += [s for s in fact_status if (s[0], s[1]) not in have]

            self.fclog.write(ts, objects, scores, thr, statuses, model_version, st.version)
            if hasattr(self.sink, 'flush'):
                self.sink.flush()
            if hasattr(self.audit, 'flush'):
                self.audit.flush()                  # 13.4: события, скопившиеся в файле, пока Redis лежал
            self.rules.dump(config.RULES_STATE)
            self._save_state()

            # наблюдение и обслуживание
            stale = any(fres.get(tp, 0) >= config.STALE_HOURS.get(tp, config.STALE_HOURS['default'])
                        for tp in config.TYPES)
            self.observer.observe(alarms={tp: applied[tp][0] for tp in config.TYPES},
                                  history={tp: self.history.history(tp) for tp in config.TYPES},
                                  freshness=fres, stale=stale, dlq=getattr(self.sink, 'failed', 0), now=now)
            self.observer.dump()
            if now.hour == 0:
                self.history.dump()
                self.fclog.sweep(now)
            sweep = self.store.retention_sweep(now)
            self.freshness, self.last_now = fres, now
            self.last = {'hour_end': hour_iso(h), 'seconds': round(time.time() - t0, 1),
                         'features_seconds': round(t_feat, 1), 'objects': int(len(objects)),
                         'alarms': n_alarm, 'muted': sum(s[2] == 'MUTED' for s in statuses),
                         'rejected': sum(s[2] == 'REJECTED' for s in statuses), 'facts': n_fact,
                         'stale': stale, 'sweep': int(sweep or 0)}
            log.info('такт %s: %s', now.isoformat(), self.last)
            return self.last

    def _silence(self) -> dict[int, set]:
        """Объект → семейства датчиков, у которых воронка отметила молчащие каналы (раздел 55)."""
        import snapshot as snapmod
        out: dict[int, set] = {}
        for o, s in self.store.silent_channels().select(['object_id', 'stype']).iter_rows():
            for fam in snapmod.families_of(s):
                out.setdefault(int(o), set()).add(fam)
        return out

    def _evidence(self, now, objs: list[int]) -> dict[tuple, list]:
        """(объект, тип) → свидетели §9.7: последнее событие каналов семейств типа, свежие первыми."""
        import snapshot as snapmod
        ev = self.store.last_events(objs, now, config.EVIDENCE_HOURS).sort('ts', descending=True)
        out: dict[tuple, list] = {}
        for o, ch, ts, s, v in ev.select(['object_id', 'channel_id', 'ts', 'stype', 'value']).iter_rows():
            fams = snapmod.families_of(s)
            for tp, need in snapmod.FAMILIES.items():
                lst = out.setdefault((int(o), tp), [])
                if fams & set(need) and len(lst) < config.EVIDENCE_MAX:
                    lst.append({'sensor_id': int(ch), 'ts': ts.isoformat(timespec='seconds') + config.TZ,
                                'value': v})
        return out

    def _emit_forecasts(self, now, h, frame, objects, scores, thr, applied, model_version, clock,
                        fres: dict | None = None) -> int:
        import recommend
        import snapshot as snapmod
        per_obj: dict[int, list] = {}
        for tp in config.TYPES:
            for i in np.flatnonzero(applied[tp][0]):
                per_obj.setdefault(int(i), []).append(tp)
        reasons = {(i, tp): self.predictor.reasons(frame, i, tp, k=3) for i, tps in per_obj.items() for tp in tps}
        conf = {}
        for tp in config.TYPES:
            idx = [i for i, tps in per_obj.items() if tp in tps]
            try:
                c = self.predictor.confidence(tp, np.asarray(scores[tp])[idx]) if idx else None
            except Exception:                # уверенность — подсказка: без неё тревога всё равно уходит
                log.exception('калибровка %s не посчиталась', tp)
                c = None
            if c is not None:
                conf.update({(i, tp): float(v) for i, v in zip(idx, c)})
        try:
            silence = self._silence()
            evidence = self._evidence(now, [int(objects[i]) for i in per_obj])
        except Exception:
            log.exception('молчание каналов и свидетели не собрались')
            silence, evidence = {}, {}
        fres = fres or {}
        stale = {tp: fres[tp] for tp in config.TYPES
                 if fres.get(tp, 0) >= config.STALE_HOURS.get(tp, config.STALE_HOURS['default'])}
        wanted = {int(objects[i]): (tps, [f'{tp}:{r["feature"]}' for tp in tps for r in reasons[(i, tp)]],
                                    {tp: int(applied[tp][1][i]) for tp in tps})
                  for i, tps in per_obj.items()}
        try:
            recs = recommend.batch(self.store.con, now, wanted)
        except Exception:                    # мера — подсказка: без неё тревога всё равно уходит
            log.exception('рекомендации такта не собрались')
            recs = {}
        for i, oid in enumerate(objects):
            o = int(oid)
            msg = outbox.build_message(o, hour_iso(h), model_version, config.TYPES, clock=clock)
            quiet = silence.get(o, set())
            for tp in config.TYPES:
                alarm, since = bool(applied[tp][0][i]), int(applied[tp][1][i])
                sil = sorted(quiet & set(snapmod.FAMILIES[tp]))
                c = conf.get((i, tp))
                if c is not None:
                    for fam in sil:
                        c *= 1.0 - config.SILENCE_COST.get(tp, {}).get(fam, 0.0)
                outbox.fill_type(msg, tp, score=scores[tp][i], threshold=thr[tp], alarm=alarm,
                                 since_hours=since, reasons=reasons.get((i, tp)) if alarm else None,
                                 confidence=c, evidence=evidence.get((o, tp)), silent=sil,
                                 stale_hours=stale.get(tp))
            rec = recs.get(o)
            if rec is not None:
                outbox.attach_recommendation(msg, rec)
            self.sink.send(msg)
        return len(reasons)

    def _emit_facts(self, now, h, eps, coll, model_version, clock, accs=()) -> tuple[int, list]:
        """Канал «по факту» (M8): объявление и его обновления, пока эпизод живой. Н10 по §1.6: газ и
        прочие типы из строки графика — с пометкой «идёт ППР по графику» и номером строки; отказ
        снятого датчика — MUTED в историю главного диспетчера, диспетчеру не уходит. У проникновения —
        маршрут. Аварии и слепота §13.11 (`accs`) в окне любых работ молчат полностью: ни объявления,
        ни MUTED, ни аудита — иначе дублировали бы график и прогноз. Жара при живом пожаре — поле
        `temperature` в блоке пожара, а не отдельный тип."""
        import recommend
        per_obj: dict[int, dict] = {}
        statuses = []
        for e in eps:
            o, tp = e['object_id'], e['type']
            blk = {'started_at': _iso(e['t0']), 'last_at': _iso(e['t1']), 'new': e['new']}
            if 'route' in e:
                blk['route'] = [{**r, 'at': _iso(r['at'], 'seconds')} for r in e['route']]
            if e['noise'] == 'Н10':
                w = self.works.fact_window(o, e['collector_id'] or coll.get(o), tp, e['t0'], e['stype'])
                if w is not None and tp == 'sensor' and w.sensor and e['stype'] == w.sensor:
                    statuses.append((o, tp, 'MUTED', 'works', w.work_id))
                    if e['new']:
                        self._audit_forecast('forecast.muted', (o, tp), h, {
                            'reason': 'works', 'work_id': w.work_id, 'fact': True, 'sensor': e['stype']})
                    continue
                blk.update({'note': worksmod.FACT_NOTE, 'work_id': w.work_id if w else None})
            per_obj.setdefault(o, {})[tp] = blk
        for e in accs:
            o, tp = e['object_id'], e['type']
            if self.works.covers(o, e['collector_id'] or coll.get(o), e['t0']) is not None:
                continue
            blk = {'started_at': _iso(e['t0']), 'last_at': _iso(e['t1']), 'new': e['new']}
            if tp == 'temperature':
                info = {'direction': e['direction'],
                        'channels': [{**x, 'at': _iso(x['at'])} for x in e['channels']]}
                fire = per_obj.get(o, {}).get('fire')
                if e['direction'] == 'hot' and fire is not None:
                    fire['temperature'] = info
                    continue
                blk.update(info)
            else:
                blk.update({'cause': e['cause'], 'share': e['share'], 'possible_accident': True})
            per_obj.setdefault(o, {})[tp] = blk
        wanted = {o: ([tp for tp, b in tps.items() if 'note' not in b and tp in config.TYPES], (), {})
                  for o, tps in per_obj.items()}
        wanted = {o: v for o, v in wanted.items() if v[0]}
        try:
            recs = recommend.batch(self.store.con, now, wanted)
        except Exception:
            log.exception('рекомендации по факту не собрались')
            recs = {}
        for o, tps in per_obj.items():
            msg = outbox.build_fact(o, hour_iso(h), model_version, tps, clock=clock)
            if o in recs:
                outbox.attach_recommendation(msg, recs[o])
            self.sink.send(msg)
        return len(per_obj), statuses

    def run(self, stop: threading.Event) -> None:
        """Такт по часам clock.py до остановки; упавший такт повторяется через минуту."""
        while not stop.is_set():
            t = self.clock.wait(stop)
            if t is None:
                return
            try:
                self.tick(t)
                self.clock.done(t)
            except Exception:
                log.exception('такт %s не прошёл — повтор через минуту', t.isoformat())
                stop.wait(60)

    # ----- состояние ------------------------------------------------------------------------------
    def _save_state(self) -> None:
        _save_json(config.SERVICE_STATE, self.state)

    def _persist(self) -> None:
        self.rules.dump(config.RULES_STATE)
        self._save_state()

    def current_hour(self, env: dict | None = None) -> int:
        """Строка, на которую ложится решение: последний посчитанный час (в проигрыше — модельный)."""
        if self.last_now is not None:
            return hour_index(self.last_now) - 1
        issued = (env or {}).get('issued_at')
        return hour_index(to_msk(issued) if issued else config.now_msk()) - 1

    # ----- аудит ----------------------------------------------------------------------------------
    def _audit(self, event: str, *, env: dict | None = None, object_type=None, object_id=None,
               details=None, outcome: str = 'success') -> None:
        by = (env or {}).get('issued_by') or {}
        try:
            self.audit.event(event, outcome, actor_kind='user' if env else 'service',
                             actor_id=by.get('sub') if env else 'tf-model', actor_login=by.get('login'),
                             request_id=(env or {}).get('request_id'), object_type=object_type,
                             object_id=object_id, details=details or {})
        except Exception:                     # аудит не роняет такт и не держит команду
            log.exception('аудит %s не записан', event)

    def _audit_forecast(self, event, key, h, details, env=None) -> None:
        self._audit(event, env=env, object_type='forecast', object_id=f'{key[0]}:{key[1]}:{hour_iso(h)}',
                    details=details)

    # ----- команды 13.3 ---------------------------------------------------------------------------
    HANDLERS = {'decision.take': '_take', 'decision.reject': '_reject', 'decision.mute': '_mute',
                'decision.reopen': '_reopen', 'decision.confirmed': '_confirmed',
                'settings.operating': '_operating', 'settings.works': '_works',
                'settings.gaps': '_gaps', 'model.switch': '_switch', 'retrain.request': '_retrain'}

    def handle(self, env: dict) -> dict:
        """Одна команда. CommandError — сломанная (reject → DLQ); прочие исключения — временная
        ошибка (nack с повтором). Состояние пишется на том до возврата — до ack."""
        if not isinstance(env, dict) or env.get('schema') != 1:
            raise CommandError('конверт не schema 1')
        cid, kind, payload = env.get('command_id'), env.get('kind'), env.get('payload')
        if not cid or kind not in self.HANDLERS:
            raise CommandError(f'нет command_id или вид {kind!r} неизвестен')
        if not isinstance(payload, dict):
            raise CommandError('payload не объект')
        with self.lock:
            if cid in self.seen:
                return {'status': 'duplicate'}
            res = getattr(self, self.HANDLERS[kind])(payload, env)
            self._persist()
            self.seen[cid] = None
            while len(self.seen) > SEEN_KEEP:
                self.seen.pop(next(iter(self.seen)))
            _save_json(config.COMMANDS_SEEN, list(self.seen))
        log.info('команда %s %s: %s', kind, cid, res)
        return res

    @staticmethod
    def _pair(p: dict) -> tuple[int, str]:
        try:
            o, tp = int(p['object_id']), str(p['type'])
        except (KeyError, TypeError, ValueError):
            raise CommandError('нужны object_id и type') from None
        if tp not in config.TYPES:
            raise CommandError(f'тип {tp!r} не из {config.TYPES}')
        return o, tp

    def _ref(self, env, h, **extra) -> dict:
        return {'command_id': env['command_id'], 'login': (env.get('issued_by') or {}).get('login'),
                'hour': h, **extra}

    def _take(self, p, env):
        o, tp = self._pair(p)
        self.rules.on_decision(o, tp, 'TAKE', self.current_hour(env), ref=self._ref(env, self.current_hour(env)))
        return {'status': 'ok'}

    def _reject(self, p, env):
        o, tp = self._pair(p)
        h = self.current_hour(env)
        code = p.get('reason_code')
        changed = self.rules.on_decision(o, tp, 'REJECT', h, ref=self._ref(env, h, reason_code=code))
        k = self.settings.reject_k(tp)
        until = ('только история: правило отклонения у типа выключено' if k is None else
                 'до эпизода этого типа или снятия отклонения' if k == 0 else
                 f'пока оценка ниже квантиля 1−доля·{k} за 90 суток, до эпизода или снятия, '
                 f'не дольше {config.REJECT_N_HOURS} ч')
        self._audit_forecast('forecast.rejected', (o, tp), h,
                             {'reason_code': code, 'reject_k': k, 'until': until, 'rule': changed}, env)
        return {'status': 'ok', 'rule': changed}

    def _mute(self, p, env):
        o, tp = self._pair(p)
        h = self.current_hour(env)
        try:
            until = to_msk(p['until'])
        except (KeyError, TypeError, ValueError):
            raise CommandError('until — время конца молчания ISO 8601') from None
        uh = hour_index(until)                        # молчат строки, чей час кончается не позже until
        if uh <= h + 1:
            raise CommandError('until уже прошёл')
        self.rules.on_decision(o, tp, 'MUTE', h, until=uh, ref=self._ref(env, h))
        self._audit_forecast('forecast.muted', (o, tp), h,
                             {'reason': 'decision', 'command_id': env['command_id'],
                              'until': until.isoformat(timespec='minutes') + config.TZ}, env)
        return {'status': 'ok'}

    def _reopen(self, p, env):
        o, tp = self._pair(p)
        h = self.current_hour(env)
        changed = self.rules.on_decision(o, tp, 'REOPEN', h, ref=self._ref(env, h))
        self._audit_forecast('forecast.reopened', (o, tp), h, {'changed': changed}, env)
        return {'status': 'ok', 'changed': changed}

    def _confirmed(self, p, env):
        """Подтверждённое происшествие — метка для аудита отклонений (§9.2): если пара была под
        отклонением или молчанием, это «переросло в эпизод» (forecast.recurred)."""
        o, tp = self._pair(p)
        h = self.current_hour(env)
        was = self.rules.status(o, tp, h)
        self.rules.on_decision(o, tp, 'CONFIRMED', h, ref=self._ref(env, h, incident_id=p.get('incident_id')))
        if was is not None:
            back = h - int(was[1].get('hour', h))
            self._audit_forecast('forecast.recurred', (o, tp), h, {
                'was': was[0], 'command_id': was[1].get('command_id'), 'incident_id': p.get('incident_id'),
                'occurred_at': p.get('occurred_at'), 'hours_after_decision': back,
                'within_horizon': back <= config.HORIZON, 'source': 'confirmed'}, env)
        return {'status': 'ok', 'recurred': was is not None}

    def _stale(self, table: str, version, current: int) -> dict | None:
        try:
            version = int(version)
        except (TypeError, ValueError):
            raise CommandError(f'{table}: нет version') from None
        if version <= current:
            log.warning('%s: версия %s не новее текущей %s — снимок отброшен', table, version, current)
            return {'status': 'stale', 'version': version, 'current': current}
        return None

    def _operating(self, p, env):
        if (skip := self._stale('settings.operating', p.get('version'), self.settings.version)):
            return skip
        by = (env.get('issued_by') or {}).get('login') or ''
        raw = {'version': int(p['version']), 'changed': p.get('changed') or env.get('issued_at'),
               'changed_by': p.get('changed_by') or by, 'reason': p.get('reason'),
               'types': p.get('types')}
        try:
            _validate(raw)
        except (AssertionError, KeyError, TypeError, AttributeError) as e:
            raise CommandError(f'settings.operating не прошёл проверку: {e}') from None
        old = self.settings
        vdir = self.dir / 'operating_versions'
        vdir.mkdir(exist_ok=True)
        _save_json(vdir / f'v{old.version}.json', old.as_dict())
        new = OperatingSettings._from(raw)
        new.save(self.settings_path)
        self.settings = new
        self.rules.settings = new
        diff = {tp: {k: [old.types[tp][k], new.types[tp][k]] for k in ('share', 'reject_k')
                     if old.types[tp][k] != new.types[tp][k]} for tp in config.TYPES}
        diff = {tp: v for tp, v in diff.items() if v}
        self._audit('settings.changed', env=env, object_type='settings', object_id=f'operating:v{new.version}',
                    details={'from': old.version, 'to': new.version, 'changes': diff, 'reason': raw['reason']})
        return {'status': 'ok', 'version': new.version, 'changes': diff}

    def _works(self, p, env):
        if (skip := self._stale('settings.works', p.get('version'), self.works.version)):
            return skip
        rows = p.get('rows')
        if not isinstance(rows, list):
            raise CommandError('settings.works: rows — список строк графика')
        old = self.works.version
        try:
            diff = self.works.replace(rows, int(p['version']), (env.get('issued_by') or {}).get('login') or '',
                                      p.get('reason'), config.now_msk())
        except ValueError as e:
            raise CommandError(f'settings.works: {e}') from None
        self.works.point_labels()
        self._audit('works.changed', env=env, object_type='settings', object_id=f'works:v{diff["version"]}',
                    details={'from': old, 'to': diff['version'], 'added': diff['added'],
                             'removed': diff['removed'], 'changed': diff['changed']})
        return {'status': 'ok', **diff}

    def _gaps(self, p, env):
        if (skip := self._stale('settings.gaps', p.get('version'), self.gaps.version)):
            return skip
        rows = p.get('rows')
        if not isinstance(rows, list):
            raise CommandError('settings.gaps: rows — список периодов {a, b, comment}')
        old = self.gaps.version
        try:
            diff = self.gaps.replace(rows, int(p['version']), (env.get('issued_by') or {}).get('login') or '',
                                     p.get('reason'), config.now_msk())
        except ValueError as e:
            raise CommandError(f'settings.gaps: {e}') from None
        dropped = self.history.drop_hours(self._gap_hours()) if self.history is not None else 0
        if dropped:
            self.history.dump()
        # §1.5: брак без переобучения не закрыть — флаг видит админ-панель в /status
        self.state['retrain_needed'] = True
        self.state['retrain_reason'] = f'игнорируемые периоды v{diff["version"]}'
        self._audit('gaps.changed', env=env, object_type='settings', object_id=f'gaps:v{diff["version"]}',
                    details={'from': old, 'to': diff['version'], 'added': diff['added'],
                             'removed': diff['removed'], 'history_rows_dropped': dropped,
                             'retrain_needed': True})
        return {'status': 'ok', **diff, 'retrain_needed': True}

    def _switch(self, p, env):
        tp = p.get('type')
        if tp not in config.TYPES:
            raise CommandError(f'model.switch: тип {tp!r} не из {config.TYPES}')
        v = p.get('version_id')
        n = None if v in (None, '', 0, '0', 'main') else int(str(v).lstrip('v'))
        old = self.state['versions'].get(tp)
        if self.predictor is None:
            raise RuntimeError('модели ещё не подняты')          # временная: повтор после старта
        try:
            self.predictor.use_version(tp, n)
        except (FileNotFoundError, ValueError) as e:
            raise CommandError(f'model.switch: {e}') from None
        if n:
            self.state['versions'][tp] = n
        else:
            self.state['versions'].pop(tp, None)
        # порог новой версии — по её же ретропрогону 2025 (П1), история старой не годится
        hist = self.predictor.bootstrap_history(year=2025)[tp]
        self.history.replace_type(tp, hist, self.settings, self.current_hour(env), self.predictor.version)
        self.history.drop_hours(self._gap_hours())
        self.history.dump()
        self._audit('model.switched', env=env, object_type='model', object_id=tp,
                    details={'from': old or 'основная', 'to': n or 'основная', 'reason': p.get('reason'),
                             'model_version': self.predictor.version})
        return {'status': 'ok', 'type': tp, 'from': old, 'to': n}

    def _retrain(self, p, env):
        strategy, reason = p.get('strategy') or 'all', p.get('reason')
        details = {'strategy': strategy, 'reason': reason}
        if not RETRAIN_ENABLED:
            self._audit('retrain.requested', env=env, object_type='retrain_job', outcome='denied',
                        details={**details, 'why': 'переобучение в этом контуре выключено (TF_MODEL_RETRAIN)'})
            return {'status': 'disabled'}
        from retrainer import Retrainer
        r = Retrainer()
        res = r.request((env.get('issued_by') or {}).get('login') or '', reason or '', strategy=strategy)
        if not res['ok']:
            self._audit('retrain.requested', env=env, object_type='retrain_job', outcome='denied',
                        details={**details, 'why': res['why']})
            return {'status': 'refused', 'why': res['why']}
        job = env['command_id']
        self._audit('retrain.requested', env=env, object_type='retrain_job', object_id=job, details=details)
        self._retrain_thread = threading.Thread(target=self._retrain_run, args=(r, job), daemon=True)
        self._retrain_thread.start()
        return {'status': 'queued', 'job': job}

    def _retrain_run(self, r, job: str) -> None:
        env = {'TF_GAPS': str(self.gaps.path), 'TF_WORKS': str(self.works.path)}
        st = r.run(env=env)
        self._audit('retrain.finished', object_type='retrain_job', object_id=job,
                    outcome='success' if st['status'] == 'done' else 'error',
                    details={'status': st['status'], 'version': st.get('version'), 'error': st.get('error')})
        if st['status'] == 'done':
            with self.lock:
                self.state['retrain_needed'] = False
                self._save_state()

    # ----- чтение для ручек -----------------------------------------------------------------------
    def health(self) -> dict:
        return {'ok': self.predictor is not None, 'clock': self.clock.label if self.clock else 'live',
                'last_hour': self.last.get('hour_end')}

    def status(self) -> dict:
        import predict as predmod
        p = self.predictor
        avail = predmod.available_versions(p.export) if p is not None else {}
        return {
            'env': config.ENV, 'clock': self.clock.label if self.clock else 'live',
            'model': {'version': p.version if p else None,
                      'types': {tp: {'version': self.state['versions'].get(tp) or 'основная',
                                     'available': avail.get(tp, [])} for tp in config.TYPES}},
            'settings': self.settings.as_dict(),
            'settings_bounds': settings_bounds(),
            'works': {'version': self.works.version, 'rows': len(self.works.rows), **self.works.meta},
            'gaps': {'version': self.gaps.version, 'rows': self.gaps.rows},
            'retrain': {'enabled': RETRAIN_ENABLED, 'needed': self.state['retrain_needed'],
                        'reason': self.state.get('retrain_reason')},
            'thresholds': dict(self.history.thresholds) if self.history else {},
            'freshness_hours': self.freshness, 'last_tick': self.last,
            'ingest': dict(self.ingest_stats) if self.ingest_stats is not None else None}

    def estimate(self, tp: str, share: float) -> dict:
        """«Доля часов → тревог в сутки» по истории последних 90 суток (§9.3) — без записи."""
        from settings import estimate
        if tp not in config.TYPES:
            raise ValueError(f'тип {tp!r} не из {config.TYPES}')
        if not 0.0 < share < 1.0:
            raise ValueError('доля — в (0, 1)')
        hh, pp = self.history.history(tp)
        lo = np.searchsorted(hh, hh[-1] - 90 * 24 + 1)
        return {'type': tp, 'share': share, 'current_share': self.settings.share(tp),
                **estimate(pp[lo:], share, 90)}

    def forecast(self, object_id: int) -> dict | None:
        return getattr(self.sink, 'last', {}).get(int(object_id))

    def forecast_history(self, object_id: int, tp: str, a: datetime, b: datetime) -> list[dict]:
        return self.fclog.history(object_id, tp, a, b)
