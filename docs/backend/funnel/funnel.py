"""Воронка (tasks/todo.md Ф1-7): пакеты показаний от шины объекта → проверка → Kafka `tf.ingest.*`.

    python funnel.py [--port 8000]

Шина объекта ходит к нам с долгим токеном техучётки (право `telemetry.push`, права-и-аудит §3) и
шлёт пакет событий в формате журнала — так же, как их отдаёт эмулятор think-test:

    POST /events   {"events": [{"ид_события": 1, "ид_канала_данных": 196771, "дата": "2026-09-26",
                                 "время": "03:09:27", "тревожное": false, "значение_датчика": "0.4"}]}

Ответ 202 `{"accepted": n, "rejected": [{"index": i, "reason": "..."}]}`: хорошие события приняты,
плохие перечислены. Не принято ни одно — 422. Kafka недоступна — 503, шина повторяет пакет (повтор
модель отбрасывает по первичному ключу журнала). Отказ по пакету — `telemetry.rejected` в аудит, без
значений датчиков.

Куда событие: тревожное или текстовое состояние («Норма», «Неисправен», дата охраны) —
`tf.ingest.journal`, его читают и модель, и BFF для ленты; обычное число — `tf.ingest.readings`,
его читает модель. Одно событие — одно сообщение в одном топике, ключ — номер канала: порядок внутри
канала сохраняется, двойного счёта у модели нет.

Молчание. Канал, который слышали хотя бы трижды, молчит, если нового события нет дольше
max(TF_FUNNEL_SILENT_MIN, 4 × его обычный интервал); шина целиком — если нет ни одного пакета дольше
TF_FUNNEL_SILENT_MIN. Переход в «молчит» и обратно — запись `channel.status` в компактный
`tf.ingest.reference` с ключом `channel-status:<канал>` и строка в /status. Сервис при этом не падает.

Стенд: TF_FUNNEL_PULL=http://tf-emulator:8000 — воронка сама забирает поток у эмулятора по курсору
(`/events?cursor=`), вместо того чтобы ждать шину. Без Kafka (TF_KAFKA_BOOTSTRAP=off) события
пишутся строками JSON в TF_FUNNEL_OUT.

Окружение: TF_ENV, TF_KAFKA_BOOTSTRAP (tf-kafka:9092), пароль Kafka — Vault `secret/tf/kafka`
поле `funnel_password` (в dev — TF_KAFKA_FUNNEL_PASSWORD), TF_AUTH_JWKS, техучётки без scope —
Vault `secret/tf/funnel` поле `service_subs`, TF_REDIS_URL — поток аудита.
"""
import argparse
import json
import logging
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _kit in (os.environ.get('TF_KIT'), HERE.parent / 'tfkit', HERE / 'tfkit'):
    if _kit and (Path(_kit) / 'tfkit.py').exists():
        sys.path.insert(0, str(_kit))
        break
import tfkit  # noqa: E402

log = logging.getLogger('funnel')
MSK = timezone(timedelta(hours=3))
TOPIC_READINGS, TOPIC_JOURNAL = 'tf.ingest.readings', 'tf.ingest.journal'
TOPIC_REFERENCE = 'tf.ingest.reference'
SCOPE = 'telemetry.push'
MAX_EVENTS = 10_000                      # как limit у эмулятора; сообщение Kafka — одно событие
MAX_VALUE = 255
TRUE, FALSE = ('true', 't', '1'), ('false', 'f', '0')


class Unavailable(RuntimeError):
    """Kafka не подтвердила запись: пакет не принят, шина повторит."""


# ----- разбор ----------------------------------------------------------------------------------
def normalize(e) -> dict:
    """Событие в формате журнала или ValueError с причиной (без значения датчика)."""
    if not isinstance(e, dict):
        raise ValueError('событие не объект')
    try:
        channel = int(e['ид_канала_данных'])
    except KeyError:
        raise ValueError('нет ид_канала_данных') from None
    except (TypeError, ValueError):
        raise ValueError('ид_канала_данных не число') from None
    if channel <= 0 or isinstance(e['ид_канала_данных'], bool):
        raise ValueError('ид_канала_данных не число')
    try:
        datetime.strptime(f"{e['дата']} {e['время']}", '%Y-%m-%d %H:%M:%S')
    except KeyError:
        raise ValueError('нет даты или времени') from None
    except (TypeError, ValueError):
        raise ValueError('дата или время не в формате ГГГГ-ММ-ДД ЧЧ:ММ:СС') from None
    alarm = e.get('тревожное', False)
    if isinstance(alarm, str) and alarm.strip().lower() in TRUE + FALSE:
        alarm = alarm.strip().lower() in TRUE
    if not isinstance(alarm, bool):
        raise ValueError('тревожное не true/false')
    value = e.get('значение_датчика')
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError('нет значения датчика')
    value = str(value)
    if not value.strip() or len(value) > MAX_VALUE:
        raise ValueError(f'значение датчика пустое или длиннее {MAX_VALUE}')
    event_id = e.get('ид_события')
    if event_id is not None:
        try:
            event_id = int(event_id)
        except (TypeError, ValueError):
            raise ValueError('ид_события не число') from None
    return {'ид_события': event_id, 'ид_канала_данных': channel, 'дата': e['дата'], 'время': e['время'],
            'тревожное': alarm, 'значение_датчика': value}


def numeric(value: str) -> bool:
    """Как модель (storage.clean_event): число — то, что читает float()."""
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def topic_of(ev: dict) -> str:
    return TOPIC_JOURNAL if ev['тревожное'] or not numeric(ev['значение_датчика']) else TOPIC_READINGS


def events_of(body) -> list:
    """Пакет: {"events": [...]}, голый список или ответ эмулятора {"события": [...]}."""
    if isinstance(body, dict):
        body = body.get('events', body.get('события'))
    if not isinstance(body, list):
        raise ValueError('пакет — список событий в поле events')
    if len(body) > MAX_EVENTS:
        raise ValueError(f'в пакете {len(body)} событий, больше {MAX_EVENTS} нельзя')
    return body


# ----- молчание каналов ------------------------------------------------------------------------
class Channels:
    def __init__(self, silent_after: float = 3600.0):
        self.silent_after = silent_after
        self.last: dict[int, float] = {}
        self.gap: dict[int, float] = {}          # обычный интервал канала, сглаженный
        self.count: dict[int, int] = {}
        self.silent: dict[int, float] = {}       # канал → с какого времени молчит
        self.source_last = 0.0
        self.source_silent = False
        self.lock = threading.Lock()

    def seen(self, channel: int, at: float) -> bool:
        """Отметить событие. True — канал молчал и снова заговорил."""
        with self.lock:
            prev = self.last.get(channel)
            back = self.silent.pop(channel, None) is not None
            if prev is not None and at > prev and not back:     # само молчание в интервал не входит
                d = at - prev
                self.gap[channel] = d if channel not in self.gap else 0.8 * self.gap[channel] + 0.2 * d
            self.last[channel] = max(at, prev or 0.0)
            self.count[channel] = self.count.get(channel, 0) + 1
            return back

    def packet(self, at: float) -> bool:
        """Пришёл пакет. True — шина молчала и снова заговорила."""
        with self.lock:
            back = self.source_silent
            self.source_last, self.source_silent = at, False
            return back

    def limit(self, channel: int) -> float:
        return max(self.silent_after, 4 * self.gap.get(channel, 0.0))

    def sweep(self, now: float) -> tuple[list[int], bool]:
        """Кто замолчал с прошлого обхода; второе — замолчала ли шина целиком."""
        went = []
        with self.lock:
            for ch, at in self.last.items():
                if ch not in self.silent and self.count.get(ch, 0) >= 3 and now - at > self.limit(ch):
                    self.silent[ch] = at
                    went.append(ch)
            source_went = bool(self.source_last) and not self.source_silent and \
                now - self.source_last > self.silent_after
            if source_went:
                self.source_silent = True
        return went, source_went

    def snapshot(self, limit: int = 200) -> dict:
        with self.lock:
            silent = sorted(self.silent.items(), key=lambda x: x[1])
            return {'channels': len(self.last), 'silent': len(silent),
                    'silent_channels': [{'channel': c, 'since': iso(at)} for c, at in silent[:limit]],
                    'source_silent': self.source_silent,
                    'source_last': iso(self.source_last) if self.source_last else None}


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, MSK).isoformat(timespec='seconds')


# ----- запись ----------------------------------------------------------------------------------
class KafkaSink:
    def __init__(self, conf: dict, timeout: float = 10.0):
        from confluent_kafka import Producer
        self.p = Producer({**conf, 'acks': 'all', 'enable.idempotence': True, 'compression.type': 'lz4',
                           'linger.ms': 20, 'message.max.bytes': 1_048_576})
        self.timeout = timeout

    def send(self, messages: list[tuple[str, str, dict]]) -> None:
        """(топик, ключ, тело) — все или исключение Unavailable."""
        failed = []

        def done(err, _msg):
            if err is not None:
                failed.append(err)
        try:
            for topic, key, body in messages:
                self.p.produce(topic, key=key.encode(), on_delivery=done,
                               value=json.dumps(body, ensure_ascii=False).encode())
                self.p.poll(0)
        except BufferError:
            raise Unavailable('очередь отправки Kafka переполнена') from None
        left = self.p.flush(self.timeout)
        if failed or left:
            raise Unavailable(f'Kafka не подтвердила {len(failed) or left} сообщений')


class FileSink:
    """Без Kafka (стенд, отладка): строка JSON на сообщение."""

    def __init__(self, path: Path):
        self.path, self.lock = path, threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, messages) -> None:
        with self.lock, open(self.path, 'a', encoding='utf-8') as f:
            for topic, key, body in messages:
                f.write(json.dumps({'topic': topic, 'key': key, 'value': body}, ensure_ascii=False) + '\n')


# ----- воронка ---------------------------------------------------------------------------------
class Funnel:
    def __init__(self, sink, audit, channels: Channels | None = None):
        self.sink, self.audit = sink, audit
        self.channels = channels or Channels()
        self.stats = {'packets': 0, 'accepted': 0, 'rejected': 0, 'unavailable': 0, 'last_accepted': None}
        self.lock = threading.Lock()

    def take(self, raw: list, *, via: str, actor: dict | None = None, request_id: str | None = None) -> dict:
        now = time.time()
        good, bad = [], []
        for i, e in enumerate(raw):
            try:
                good.append(normalize(e))
            except ValueError as err:
                bad.append({'index': i, 'reason': str(err)})
        if bad:
            self.rejected(bad, len(raw), via=via, actor=actor, request_id=request_id)
        if good:
            try:
                self.sink.send([(topic_of(ev), str(ev['ид_канала_данных']), ev) for ev in good])
            except Unavailable:
                with self.lock:
                    self.stats['unavailable'] += 1
                raise
        back = [ev['ид_канала_данных'] for ev in good if self.channels.seen(ev['ид_канала_данных'], now)]
        if self.channels.packet(now):
            log.info('шина снова на связи')
        if back:
            self.status(back, 'ok', now)
        with self.lock:
            self.stats['packets'] += 1
            self.stats['accepted'] += len(good)
            self.stats['rejected'] += len(bad)
            if good:
                self.stats['last_accepted'] = iso(now)
        return {'accepted': len(good), 'rejected': bad[:100]}

    def rejected(self, bad: list, total: int, *, via, actor, request_id, reason: str | None = None) -> None:
        """telemetry.rejected: почему, сколько; значений датчиков в журнале нет (§6.3)."""
        reasons = sorted({b['reason'] for b in bad})[:5] if bad else [reason]
        actor = actor or {}
        try:
            self.audit.event('telemetry.rejected', 'denied', actor_kind=actor.get('kind', 'service'),
                             actor_id=actor.get('sub'), request_id=request_id, object_type='packet',
                             details={'via': via, 'events': total, 'rejected': len(bad) or total,
                                      'reasons': reasons})
        except Exception:
            log.exception('аудит telemetry.rejected не записан')

    def status(self, channels: list[int], state: str, now: float) -> None:
        msgs = [(TOPIC_REFERENCE, f'channel-status:{c}',
                 {'kind': 'channel.status', 'ид_канала_данных': c, 'status': state, 'at': iso(now),
                  'since': iso(self.channels.silent.get(c, now)) if state == 'silent' else iso(now)})
                for c in channels]
        try:
            self.sink.send(msgs)
        except Unavailable:
            log.warning('статус %d каналов не записан: Kafka недоступна', len(msgs))

    def sweep(self, now: float | None = None) -> list[int]:
        now = time.time() if now is None else now
        went, source = self.channels.sweep(now)
        if source:
            log.warning('шина молчит дольше %d мин', self.channels.silent_after // 60)
        if went:
            log.info('замолчали каналы: %d', len(went))
            self.status(went, 'silent', now)
        return went


# ----- стенд: забор у эмулятора ----------------------------------------------------------------
def fetch(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def pull(funnel: Funnel, base: str, stop: threading.Event, interval: float = 1.0, limit: int = 5000,
         get=fetch) -> None:
    """Забирает поток эмулятора по курсору. Перезапуск эмулятора (его курсор меньше нашего) —
    читать сначала: повтор модель отбросит по первичному ключу."""
    cursor, idle, wait = 0, 0.0, interval
    base = base.rstrip('/')
    while not stop.is_set():
        full = False
        try:
            data = get(f'{base}/events?cursor={cursor}&limit={limit}')
            rows = data.get('события') or []
            if rows:
                funnel.take(rows, via='pull')
                cursor, idle = int(data.get('курсор', cursor)), 0.0
                full = len(rows) >= limit           # полная страница — сразу за следующей
            else:
                idle += wait
                if idle >= 30:
                    if int(get(f'{base}/health').get('курсор', cursor)) < cursor:
                        log.warning('эмулятор перезапущен — читаю поток сначала')
                        cursor = 0
                    idle = 0.0
            wait = interval
        except Unavailable:
            wait = min(wait * 2, 60)                # Kafka лежит — курсор не двигаем
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning('эмулятор недоступен (%s) — снова через %.0f с', type(e).__name__, wait)
            wait = min(wait * 2, 60)
        if not full:
            stop.wait(wait)


# ----- HTTP ------------------------------------------------------------------------------------
def make_app(funnel: Funnel, verifier, audit):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title='Think Faster — воронка показаний', version='1.0.0')
    if isinstance(audit, tfkit.Audit):
        tfkit.request_log(app, audit, verifier)          # строка на запрос (права-и-аудит §6.1)

    def refuse(event: str, status: int, reason: str, request: Request, claims: dict | None = None, jti=None):
        claims = claims or {}
        try:
            audit.event(event, 'denied', actor_kind=verifier.kind(claims) if claims else 'anonymous',
                        actor_id=claims.get('sub'), request_id=request.headers.get('x-request-id'),
                        ip=request.client.host if request.client else None, object_type='route',
                        object_id=request.url.path, details={'reason': reason, 'jti': claims.get('jti', jti)})
        except Exception:
            log.exception('аудит %s не записан', event)
        raise HTTPException(status, reason, headers={'WWW-Authenticate': 'Bearer'} if status == 401 else None)

    def caller(request: Request, scope: str | None) -> dict | None:
        if verifier is None:
            return None
        auth = request.headers.get('authorization', '')
        if not auth.lower().startswith('bearer '):
            refuse('token.refused', 401, 'нужен токен шины: Authorization: Bearer <токен>', request)
        try:
            claims = verifier.verify(auth[7:].strip())
        except tfkit.TokenError as e:
            if not e.audit:
                raise HTTPException(e.status, e.reason) from None
            refuse('token.refused', e.status, e.reason, request, jti=e.jti)
        if scope and not verifier.has_scope(claims, scope):
            refuse('access.denied', 403, f'нужно право {scope}', request, claims)
        return {'sub': claims.get('sub'), 'kind': verifier.kind(claims)}

    @app.get('/health')
    @app.get('/api/funnel/health')
    def health():
        return {'ok': True, **{k: funnel.stats[k] for k in ('accepted', 'last_accepted')},
                'source_silent': funnel.channels.source_silent}

    @app.get('/status')
    @app.get('/api/funnel/status')
    def status(request: Request):
        caller(request, None)
        return {**funnel.stats, **funnel.channels.snapshot()}

    @app.post('/events', status_code=202)
    @app.post('/api/funnel/events', status_code=202)
    def events(request: Request, body=Body(...)):
        who = caller(request, SCOPE)
        rid = request.headers.get('x-request-id')
        try:
            raw = events_of(body)
        except ValueError as e:
            funnel.rejected([], 0, via='http', actor=who, request_id=rid, reason=str(e))
            raise HTTPException(422, str(e)) from None
        try:
            result = funnel.take(raw, via='http', actor=who, request_id=rid)
        except Unavailable as e:
            return JSONResponse(status_code=503, content={'detail': f'{e}; повторите пакет'},
                                headers={'Retry-After': '5'})
        if raw and not result['accepted']:
            return JSONResponse(status_code=422, content=result)
        return result

    return app


# ----- запуск ----------------------------------------------------------------------------------
def make_sink():
    bootstrap = os.environ.get('TF_KAFKA_BOOTSTRAP', 'tf-kafka:9092')
    if bootstrap == 'off':
        return FileSink(Path(os.environ.get('TF_FUNNEL_OUT', '/tmp/tf-funnel.jsonl')))
    conf = {'bootstrap.servers': bootstrap}
    password = tfkit.secret('kafka', 'funnel_password', 'TF_KAFKA_FUNNEL_PASSWORD', required=not tfkit.is_dev())
    if password:
        conf.update({'security.protocol': 'SASL_PLAINTEXT', 'sasl.mechanism': 'PLAIN',
                     'sasl.username': os.environ.get('TF_KAFKA_USER', 'tf-funnel'), 'sasl.password': password})
    return KafkaSink(conf)


def make_verifier():
    pem = tfkit.secret('auth', 'public_key', 'TF_AUTH_PUBLIC_KEY', required=False)
    jwks = os.environ.get('TF_AUTH_JWKS') or None
    if pem is None and jwks is None:
        if tfkit.is_dev():
            log.warning('dev: ключа проверки токенов нет — пакеты принимаются без токена')
            return None
        jwks = 'http://tf-auth:8080/.well-known/jwks'
    subs = tfkit.secret('funnel', 'service_subs', 'TF_FUNNEL_SERVICE_SUBS', required=False) or ''
    return tfkit.Verifier(public_key=pem, jwks_url=None if pem else jwks,
                          service_subs=[s.strip() for s in subs.split(',') if s.strip()])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=int(os.environ.get('TF_FUNNEL_PORT', '8000')))
    args = ap.parse_args()
    import uvicorn
    audit = tfkit.Audit('funnel', spool=Path(os.environ['TF_AUDIT_SPOOL']) if os.environ.get('TF_AUDIT_SPOOL') else None)
    funnel = Funnel(make_sink(), audit, Channels(60 * float(os.environ.get('TF_FUNNEL_SILENT_MIN', '60'))))
    stop = threading.Event()

    def sweeper():
        while not stop.wait(30):
            try:
                funnel.sweep()
                audit.flush()
            except Exception:
                log.exception('обход молчания')
    threading.Thread(target=sweeper, name='sweep', daemon=True).start()
    if os.environ.get('TF_FUNNEL_PULL'):
        threading.Thread(target=pull, args=(funnel, os.environ['TF_FUNNEL_PULL'], stop), name='pull',
                         daemon=True).start()
        log.info('стенд: забираю поток у %s', os.environ['TF_FUNNEL_PULL'])
    try:
        uvicorn.run(make_app(funnel, make_verifier(), audit), host='0.0.0.0', port=args.port, log_level='warning')
    finally:
        stop.set()


if __name__ == '__main__':
    main()
