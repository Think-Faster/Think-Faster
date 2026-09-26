"""Переобучение (M9/П7): расписание на флагах main, новые модели → новая история порога.

Тяжёлая работа живёт в pipeline: `retrain.py` — исследования стратегий (all/d90/d180/d365/
decay90/decay180/warm/frozen/thrfix — чья стратегия сработала на отрезке, её ключ фиксируется);
производственное переобучение — `export.py` (обучает на всех годах, пять зёрен, без моих флагов).
Здесь — только расписание, статус для админ-панели (§9.5) и переключение прод-версии:

1. запуск export.py в подпроцессе;
2. после готовности — новый манифест и новый Predictor;
3. история оценок принадлежит СТАРОЙ версии: новая пороговая история пересчитывается заново —
   bootstrap-смесью новой версии по витрине (П1), `ScoreHistory.rebootstrap`, никаких
   threshold_override (П7). До первых суток порог нового трима — её же 90-суточное окно из 2025,
   история дописывается с каждого такта.

Статус — OUT_DIR/retrain.json: очередь → идёт → готово/ошибка + ссылка на версию. Прогнозы во
время обучения считает прежняя версия; переключает админ-панель (запись operating.json), не мы.
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import svc as config
from predict import Predictor, load_manifest

STRATEGIES = ('all', 'd90', 'd180', 'd365', 'decay90', 'decay180', 'warm', 'frozen', 'thrfix')


class Retrainer:
    def __init__(self, path: Path | None = None, python=None):
        self.path = path or config.RETRAIN_LOG
        self.python = python or sys.executable
        self.state = {'status': 'idle', 'last_run': None, 'next_run': None,
                      'version': None, 'strategy': None, 'error': None, 'history': []}
        self.load()

    def load(self):
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                pass

    def due(self, now: datetime, interval_days: int = 30) -> bool:
        return (self.state['next_run'] is None or
                datetime.fromisoformat(self.state['next_run']) <= now)

    def schedule(self, now: datetime, interval_days: int = 30) -> None:
        self.state['next_run'] = (now + timedelta(days=interval_days)).isoformat(timespec='minutes')
        self.save()

    def request(self, by: str, reason: str, note: str = '', strategy: str = 'all') -> dict:
        if self.state['status'] == 'running':
            return {'ok': False, 'why': 'уже идёт'}
        if strategy not in STRATEGIES:
            return {'ok': False, 'why': f'стратегия {strategy}; можно: {" ".join(STRATEGIES)}'}
        manifest = load_manifest()
        self.state.update({'status': 'queued', 'requested_by': by, 'reason': reason,
                           'note': note, 'strategy': strategy, 'version': manifest.get('built'),
                           'error': None})
        self.save()
        return {'ok': True}

    def run(self, by: str = 'schedule', env: dict | None = None) -> dict:
        self.state.update({'status': 'running', 'started': datetime.now().isoformat()})
        self.save()
        try:
            # продакшн-переобучение: export.py (все годы, пять зёрен); исследование стратегий —
            # это retrain.py, его крутит MLOps/CI, сервис лишь хранит выбранный ключ для журнала
            # таблицы главного диспетчера — текущие версии сервиса (TF_GAPS, TF_WORKS в config.py)
            import os
            subprocess.run([self.python, 'export.py'], cwd=config.ML / 'pipeline', check=True,
                           env={**os.environ, **(env or {})})
            new_manifest = load_manifest()
            self.state.update({'status': 'done', 'version': new_manifest.get('built'),
                               'changed': datetime.now().isoformat(timespec='seconds'),
                               'error': None})
        except subprocess.CalledProcessError as e:
            self.state.update({'status': 'error', 'error': str(e)})
        self.save()
        return self.state

    def status(self) -> dict:
        return self.state

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding='utf-8')


def apply_version(thresholds, predictor: Predictor, h: int) -> None:
    """После выхода новой выгрузки: пороги новой версии с 90-суточным окном 2025 (П1/П7).

    Пишется ровно тот же bootstrap, что при старте, — новая версия не молчит ни часа и не пользуется
    override из ретропрогона старой.
    """
    import settings as s
    scores = predictor.bootstrap_history(year=2025)
    thresholds.rebootstrap(scores, s.OperatingSettings.load(), h, model_version=predictor.version)