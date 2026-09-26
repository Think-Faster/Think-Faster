"""Часы такта (INTEGRATION §13.1; Н1 — такт раз в час, Н8 — режим проигрыша).

`TF_MODEL_CLOCK`:
- `live` — граница часа по московскому времени; такт — через `DELAY` после границы, чтобы опоздавшие
  строки часа успели прийти. Пропущенные часы (сервис лежал) не досчитываются: считается только
  текущий, пропуск пишется в лог и в состояние — прогноз задним числом диспетчеру не нужен.
- `replay:<начало ISO>:<скорость>` — проигрыш тестового года: модельный час через 3600/скорость
  реальных секунд, от сохранённого часа или от начала. Сообщения несут `clock: "replay"`.

Последний посчитанный час пишется атомарно (`clock.json`): после рестарта такт не повторяется.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import svc as config

log = logging.getLogger('tf-model')
DELAY = timedelta(seconds=int(os.environ.get('TF_MODEL_TICK_DELAY', '120')))


class Clock:
    def __init__(self, spec: str | None = None, state: Path | None = None, now=None):
        spec = spec or config.CLOCK
        self.state = Path(state or config.CLOCK_STATE)
        self._now = now or config.now_msk
        if spec == 'live':
            self.mode, self.start, self.period = 'live', None, None
        elif spec.startswith('replay:'):
            body, speed = spec[len('replay:'):].rsplit(':', 1)
            self.mode, self.start = 'replay', datetime.fromisoformat(body)
            self.period = 3600.0 / float(speed)
        else:
            raise ValueError(f'TF_MODEL_CLOCK: live или replay:<начало>:<скорость>, а не {spec!r}')
        self.last: datetime | None = None
        self.gaps = 0
        if self.state.exists():
            raw = json.loads(self.state.read_text(encoding='utf-8'))
            if raw.get('mode') == self.mode and raw.get('last_hour'):
                self.last = datetime.fromisoformat(raw['last_hour'])
                self.gaps = int(raw.get('gaps', 0))
        self._real = None

    @property
    def label(self) -> str:
        return self.mode

    def due(self) -> datetime | None:
        """Граница часа, которую пора считать, или None."""
        if self.mode == 'replay':
            nxt = self.last + timedelta(hours=1) if self.last else self.start
            if self._real is not None and time.monotonic() - self._real < self.period:
                return None
            return nxt
        now = self._now()
        b = now.replace(minute=0, second=0, microsecond=0)
        if now - b < DELAY or (self.last is not None and b <= self.last):
            return None
        return b

    def wait(self, stop: threading.Event | None = None, poll: float = 5.0) -> datetime | None:
        """Ждать следующего такта; None — пришла остановка."""
        stop = stop or threading.Event()
        while not stop.is_set():
            t = self.due()
            if t is not None:
                return t
            stop.wait(min(poll, self.period or poll))
        return None

    def done(self, t: datetime) -> None:
        if self.mode == 'live' and self.last is not None and t - self.last > timedelta(hours=1):
            missed = int((t - self.last) / timedelta(hours=1)) - 1
            self.gaps += missed
            log.warning('такт: пропущено часов %d (с %s по %s) — не досчитываются', missed,
                        (self.last + timedelta(hours=1)).isoformat(), (t - timedelta(hours=1)).isoformat())
        self.last = t
        self._real = time.monotonic()
        self.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state.with_suffix('.tmp')
        tmp.write_text(json.dumps({'mode': self.mode, 'last_hour': t.isoformat(), 'gaps': self.gaps}),
                       encoding='utf-8')
        os.replace(tmp, self.state)
