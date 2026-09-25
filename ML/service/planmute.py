"""Молчание по графику плановых работ (M7, INTEGRATION §1.6): прогноз типа в окне строки → MUTED.

Строка графика — объект любого уровня (коллектор накрывает все свои объекты), типы происшествий и
начало/конец с точностью до часа. В окне прогноз типа из строки не показывается диспетчеру, но и не
пропадает: статус MUTED, причина `REASON`, номер строки — `work_id`; оценка модели пишется в историю
главного диспетчера (§9.2). Отказ датчика ('sensor') не глушится — молчание теряет настоящие отказы
других датчиков. Окна строит `labels.works_windows` — то же правило, что у разметки Н10 и
`maintenance.in_works` (§10.1).
"""

REASON = 'плановые работы по графику'


def windows(con, year: int) -> bool:
    """Построить временную таблицу works_win для года (годы заданы явно, `inc` не нужен)."""
    try:
        import labels
    except ImportError:
        return False
    labels.works_windows(con, [year])
    return True


def mask(con, tp: str, at) -> dict:
    """Объект → work_id для прогноза типа `tp` на час `at`, если он в окне работ; 'sensor' никогда."""
    if tp == 'sensor':
        return {}
    rows = con.sql(f"""
        SELECT DISTINCT o.object_id, w.work_id
        FROM obj3 o
        JOIN works_win w ON w.object_id IN (o.object_id, o.collector_id)
         AND list_contains(w.types, '{tp}')
         AND TIMESTAMP '{at.isoformat()}' >= w.a
         AND TIMESTAMP '{at.isoformat()}' < w.b
    """).fetchall()
    return {int(r[0]): str(r[1]) for r in rows}