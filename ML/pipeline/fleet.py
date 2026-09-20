"""След общей причины: что в этот час происходит во всём парке (раздел 27).

Три пакета внешних данных подряд дали ноль (раздел 5), и вывод там записан такой: искать надо
ряд с привязкой к объекту, потому что признак, одинаковый для всех 78 объектов, не объясняет,
почему авария будет здесь, а не у соседа. Этот скрипт заходит с другой стороны и признаёт
ограничение честно: ряд тут тоже один на весь парк. Но меряет он не обстановку, а **её след**.

Мысль простая. Если на происшествия влияет фактор, которого нет ни в одном открытом источнике —
скачок напряжения в районе, жара, ливень, ремонт магистрали, — то сам фактор мы не увидим, а вот
его отпечаток увидим сразу: в этот час загорается не один объект, а несколько. Поэтому берётся
не «сколько эпизодов», а прежде всего **на скольких объектах** они начались: два эпизода на одном
объекте — это местная неисправность, два эпизода на двух объектах — это уже кандидат в общую
причину. Разница между этими двумя случаями и есть всё, что здесь измеряется.

Чем это отличается от погоды по существу: ряд не нужно ниоткуда скачивать и он не может
устареть. В работе диспетчерская видит его в тот же момент, что и мы, — это поток из самой
системы, а не внешний файл, который завтра перестанут публиковать.

Что считается (окно всегда назад, будущего не видно):
- `fl_<тип>_6h/24h` — сколько эпизодов этого типа началось во всём парке;
- `fl_obj_<тип>_24h` — на скольких разных объектах они начались (ширина, а не счёт);
- `fl_any_24h/168h`, `fl_obj_any_24h` — то же по всем типам сразу;
- `fl_events_24h` — сколько вообще событий в журнале: общий уровень активности;
- `fl_burst_24h` — превышение суток над средними сутками недели: всплеск, а не уровень.

Свой вклад объекта из ряда не вычитается: в витрине он join-ится по часу, как погода, и ряд
физически один на всех. При 78 объектах вклад одного — единицы процентов, а свою историю модель
и так видит отдельными признаками; отделить в `fl_*` собственный след она не может.

Выход: work/fleet.parquet (час + признаки). Подключается ключом train.py --fleet.

    python fleet.py
"""
import time

import numpy as np
import polars as pl

import config
import features as ft

WINDOWS = (6, 24, 168)


def series(base: np.ndarray) -> dict[str, np.ndarray]:
    """base — объекты × часы × BASE. Возвращает ряды по всему парку."""
    f: dict[str, np.ndarray] = {}
    any_cnt = np.zeros(base.shape[1], np.float32)
    any_obj = np.zeros(base.shape[1], np.float32)
    for tp in config.TYPES:
        col = base[:, :, ft.IDX[f'onset_{tp}']]
        cnt = col.sum(axis=0)
        obj = (col > 0).sum(axis=0).astype(np.float32)
        any_cnt += cnt
        any_obj += obj
        for w in (6, 24):
            f[f'fl_{tp}_{w}h'] = ft.window_sum(cnt, w)
        f[f'fl_obj_{tp}_24h'] = ft.window_sum(obj, 24)
    for w in (24, 168):
        f[f'fl_any_{w}h'] = ft.window_sum(any_cnt, w)
    f['fl_obj_any_24h'] = ft.window_sum(any_obj, 24)
    f['fl_events_24h'] = ft.window_sum(base[:, :, ft.IDX['n_events']].sum(axis=0), 24)
    # уровень активности у парка свой в каждый сезон, поэтому интересен не он, а превышение над ним
    f['fl_burst_24h'] = f['fl_any_24h'] - f['fl_any_168h'] / 7
    return f


def main() -> None:
    t = time.time()
    con = config.connect(read_only=True)
    base, objects, _ = ft.load_base(con)
    con.close()
    print(f'часовые ряды {base.shape}, объектов {len(objects)}: {round(time.time() - t)} с', flush=True)

    f = series(base)
    out = pl.DataFrame({'h': np.arange(base.shape[1], dtype=np.int32)}
                       | {k: v.astype(np.float32) for k, v in f.items()})
    out.write_parquet(config.WORK / 'fleet.parquet')
    print(f'work/fleet.parquet: {out.height} часов, {out.width - 1} признаков', flush=True)
    busiest = out.sort('fl_obj_any_24h', descending=True).head(3)
    print('самые «широкие» сутки парка:', flush=True)
    for r in busiest.iter_rows(named=True):
        print(f"  час {r['h']}: объектов {r['fl_obj_any_24h']:.0f}, эпизодов {r['fl_any_24h']:.0f}", flush=True)
    print(f'готово за {round(time.time() - t)} с', flush=True)


if __name__ == '__main__':
    main()
