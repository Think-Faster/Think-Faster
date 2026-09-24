"""Пути, периоды и пороги ML-контура.

Журналы (15 ГБ) в git не лежат: путь к папке с ext-journal-*.csv задаёт TF_JOURNAL.
Рабочая база, витрина и модели — в TF_WORK (по умолчанию ML/work, git его не видит).
"""
import json
import os
from datetime import datetime
from pathlib import Path

import duckdb

ML = Path(__file__).resolve().parents[1]
DICT = ML.parent / 'docs' / 'dataset'
RESULTS = ML / 'results'
JOURNAL = Path(os.environ.get('TF_JOURNAL', ML / 'journal'))
WORK = Path(os.environ.get('TF_WORK', ML / 'work'))
DB = WORK / 'tf.duckdb'
SETTINGS = ML / 'settings' / 'operating.json'

# 2021 год исключён: в нём параллельно работали две системы мониторинга (plan.md §5)
YEARS = [2019, 2020, 2022, 2023, 2024, 2025, 2026]
TRAIN_END = datetime(2025, 1, 1)   # обучение ≤ 2024
VAL_END = datetime(2026, 1, 1)     # проверка — 2025, тест — 2026 (янв–июн)
DATA_END = datetime(2026, 7, 1)
# Потери данных: в эти часы нет ни событий, ни честной разметки
GAPS = [(datetime(2024, 4, 6), datetime(2024, 4, 11)), (datetime(2026, 6, 1), datetime(2026, 6, 2))]
WARMUP = 7 * 24  # часов: после начала данных и после дыры 2021 окно 7 сут ещё не заполнено

HORIZON = 24  # ч, горизонт прогноза по умолчанию (ТЗ §6: не менее 24 ч)
WINDOWS = [1, 6, 24, 168]  # ч: окна признаков 1 ч / 6 ч / 24 ч / 7 сут

# gas — шестой тип сверх ТЗ: у загазованности своя реакция (вентиляция), в пожар её не смешиваем
TYPES = ['fire', 'gas', 'flood', 'equipment', 'sensor', 'intrusion']
TYPE_NAMES = {'fire': 'пожар', 'gas': 'загазованность', 'flood': 'подтопление',
              'equipment': 'отказ оборудования', 'sensor': 'отказ датчика', 'intrusion': 'проникновение'}


def operating(path: Path = SETTINGS) -> dict:
    """Рабочие параметры, которые меняет главный диспетчер (`settings/operating.md`): по каждому
    типу доля часов под тревогой и k после отклонения (None — правило выключено). Проверяются по
    границам схемы: значение за исследованным диапазоном не молча обрезается, а останавливает
    загрузку."""
    cfg = json.loads(path.read_text(encoding='utf-8'))
    schema = json.loads((SETTINGS.parent / 'operating.schema.json').read_text(encoding='utf-8'))
    props = schema['properties']['types']['properties']
    assert set(cfg['types']) == set(TYPES), f'{path.name}: нужны все типы {TYPES}'
    kb = schema['$defs']['reject_k']
    for tp, v in props.items():
        b, s, k = v['properties']['share'], cfg['types'][tp]['share'], cfg['types'][tp]['reject_k']
        assert b['minimum'] <= s <= b['maximum'], f"{path.name}: {tp} share {s} вне [{b['minimum']}, {b['maximum']}]"
        assert k is None or kb['minimum'] <= k <= kb['maximum'], f"{path.name}: {tp} reject_k {k} вне [{kb['minimum']}, {kb['maximum']}]"
    assert cfg.get('reason'), f'{path.name}: не указана причина изменения'
    return cfg


def shares() -> dict:
    return {tp: v['share'] for tp, v in operating()['types'].items()}


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    WORK.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=read_only)
    con.sql(f"SET memory_limit='{os.environ.get('TF_MEMORY', '6GB')}'")
    con.sql(f"SET temp_directory='{(WORK / 'spill').as_posix()}'")
    con.sql('SET preserve_insertion_order=false')
    return con
