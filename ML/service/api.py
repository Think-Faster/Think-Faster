"""HTTP-ручки сервиса (M11/M12): health, /api/ml/estimate, статус и версия настроек.

FastAPI — маршрут /api/ml обслуживает nginx по суффиксу (INTEGRATION §4). Если fastapi не
установлен, модуль импортируется без ошибок, но app = None — стенд работает без него.
"""
from settings import OperatingSettings

app = None
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title='tf-model')
except ImportError:
    pass


def _bind(history=None, settings: OperatingSettings | None = None, retrainer=None,
          observer=None):
    global _history, _settings, _retrain, _observe
    _history = history
    _settings = settings
    _retrain = retrainer
    _observe = observer


_history = _settings = _retrain = _observe = None


def ensure_app():
    if app is None:
        raise RuntimeError('fastapi не установлен — ручки недоступны, стенд без них')


if app is not None:

    from typing import Optional

    class Estimate(BaseModel):
        share: float
        type: str
        alarms_per_day: Optional[float] = None
        threshold: Optional[float] = None
        limited: int = 0

    @app.get('/health')
    def health():
        last = _observe.buf[-1] if _observe and _observe.buf else None
        return {'ok': True, 'last_hour': last}

    @app.get('/api/ml/estimate')
    def estimate(type: str, share: float):
        ensure_app()
        if _settings is None or _history is None:
            raise HTTPException(503, 'сервис не инициализирован (нет настроек/истории)')
        try:
            _settings.check(share)
        except Exception as e:
            raise HTTPException(422, str(e))
        return _history.estimate(type, share)

    @app.post('/api/ml/settings')
    def update_settings(updates: dict, by: str, reason: str):
        global _settings
        ensure_app()
        if _settings is None:
            raise HTTPException(503, 'нет настроек')
        new = _settings.rebase(updates, by, reason)
        new.save()
        _settings = new
        return {'version': new.version}

    @app.get('/api/ml/status')
    def status():
        ensure_app()
        return {'retrain': _retrain.status() if _retrain else None,
                'settings_version': _settings.version if _settings else None}