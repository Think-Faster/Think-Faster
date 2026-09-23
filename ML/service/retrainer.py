"""Переобучение (M9): раз в 30 суток, дообучение поверх старого — отброшено.

Тяжёлая работа живёт в pipeline (retrain.py/export.py): здесь только расписание, статусы для
админ-панели (§9.5) и то, что меняется в проде при выходе новой версии:

1. запуск обучения (накопление всего прошлого, период 30 дней);
2. после готовности — export.py в новую выгрузку, порог пересчитывается заново:
   история оценок принадлежит старой версии (§9.4), новую версию сервис накапливает с нуля,
   пока не наберётся 90 суток, порог держится из ретропрогона (ScoreHistory.threshold_override).

Статус — OUT_DIR/retrain.json: очередь → идёт → готово/ошибка + ссылка на версию. Прогнозы в это
время считает прежняя версия, никакого переключения на лету (переключает админ-панель §9.4).
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import svc as config
from predict import Predictor, load_manifest


class Retrainer:
    def __init__(self, path: Path | None = None, python=None):
        self.path = path or config.RETRAIN_LOG
        self.python = python or sys.executable
        self.state = {'status': 'idle', 'last_run': None, 'next_run': None,
                      'version': None, 'error': None, 'history': []}
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

    def request(self, by: str, reason: str, note: str = '') -> dict:
        if self.state['status'] == 'running':
            return {'ok': False, 'why': 'уже идёт'}
        manifest = load_manifest()
        self.state.update({'status': 'queued', 'requested_by': by, 'reason': reason,
                           'note': note, 'version': manifest.get('exported'),
                           'error': None})
        self.save()
        return {'ok': True}

    def run(self, run_tag: str, by: str = 'schedule') -> dict:
        self.state.update({'status': 'running', 'run': run_tag, 'started': datetime.now().isoformat()})
        self.save()
        try:
            subprocess.run([self.python, 'retrain.py', '--run', run_tag], cwd=config.ML / 'pipeline',
                           check=True)
            subprocess.run([self.python, 'export.py', '--run', run_tag, '--overwrite'],
                           cwd=config.ML / 'pipeline', check=True)
            new_manifest = load_manifest()
            self.state.update({'status': 'done', 'version': new_manifest.get('exported'),
                               'changed': datetime.now().isoformat(timespec='seconds'),
                               'run': run_tag, 'error': None})
        except subprocess.CalledProcessError as e:
            self.state.update({'status': 'error', 'error': str(e)})
        self.save()
        return self.state

    def status(self) -> dict:
        return self.state

    def save(self) -> None:
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding='utf-8')


def apply_version(history, predictor: Predictor) -> None:
    """После выхода новой выгрузки: порог пересчитывается (§9.4), истории у версии своей нет."""
    manifest = load_manifest(predictor.export)
    for tp in manifest['models']:
        # до первых суток истории держим порог из ретропрогона новой версии (заглушка):
        # ScoreHistory.threshold() вернёт nan только при пустом окне — override не даёт молчать
        history.threshold_override[tp] = 0.5