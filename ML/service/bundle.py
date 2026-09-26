"""Пакет модели для контейнера (INTEGRATION §13.6, Н6): всё, что нужно для прогноза, в одной папке.

Сервис в контейнере не видит исследовательскую папку `work/`. Пакет собирается из неё один раз и
монтируется только для чтения (`TF_MODEL_BUNDLE=/models/current`):

    manifest.json, meta.json, importance.json      выгрузка main, описание признаков, основания тревоги
    <тип>/<семейство>_s<зерно>.*                    модели main
    scales/<тип>_<семейство>_s<зерно>.npy           оценки проверки 2025 в порядке витрины (шкала, П1)
    history/h_2025.npy                              час каждой строки витрины 2025 — для первой истории порога
    <тип>_v<N>/...                                  версии типа для главного диспетчера (§9.4), то же устройство
    settings/                                       первая версия таблиц главного диспетчера

    python bundle.py <папка> --export work/export --work work_pa                  собрать
    python bundle.py <папка> --check --export work/export --work work_pa [--hot журнал]

Шкала зерна — его оценки проверки 2025 на тех входах, что даёт сервис. `--work` — рабочая папка,
витрина и ряды которой совпадают с признаками сервиса: сейчас work_pa (счётчики эпизодов без Н10,
вариант A′ раздела 62), а не work_st, где учились модели выгрузки. `--check` это и проверяет —
признаки среза сервиса против строки витрины того же часа (`входы`).

- Бустинг. `export.py` переобучает каждую модель на 2022–2026, и модель выгрузки не совпадает с
  моделью прогона, чьё имя стоит в манифесте: шкала прогона ей не принадлежит. Шкала — оценки самой
  модели выгрузки на всех строках витрины 2025 (`own_scales`). На export_check (2022–2025, тест 2026)
  своя шкала даёт тот же PR-AUC, что шкала прогона, а долю тревоги — ближе к рабочей у всех пяти
  типов бустинга (аналитика, раздел 64).
- Сеть. Шкалу пишет `seqmodel.py --score-val` в `<work>/roll/<прогон>_<тип>_val.npy` (Н20); она
  берётся, только если `<work>/roll/<прогон>.pt` совпадает с сетью выгрузки по весам.

Важность признаков (основания тревоги, ТЗ §5) — из самих моделей выгрузки (`train.importance`), а
не из отчётов прогонов. Модели не обучаются и копируются как есть.
"""
import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

import svc as config
import predict as predmod

SETTINGS = ('operating.json', 'operating.schema.json', 'works_2026.csv', 'recommendations.csv', 'recurrence.csv')


def _same_net(a: Path, b: Path) -> bool:
    import torch
    x, y = (torch.load(p, map_location='cpu', weights_only=True) for p in (a, b))
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


def _scale_name(tp: str, e: dict) -> str:
    return f'{tp}_{e["family"]}_s{e["seed"]}.npy'


def own_scales(p: predmod.Predictor, work: Path, out: Path, entries: list[tuple[str, dict]],
               chunk: int = 100_000) -> None:
    """Оценки бустингов выгрузки `entries` на всех строках витрины 2025 папки `work` → `out`."""
    out.mkdir(parents=True, exist_ok=True)
    lf = pl.scan_parquet(work / 'features' / '2025.parquet').select(p.features)
    n = lf.select(pl.len()).collect().item()
    parts: dict[Path, list] = {e['path']: [] for _, e in entries}
    for i in range(0, n, chunk):
        X = p.matrix(lf.slice(i, chunk).collect())
        for _, e in entries:
            parts[e['path']].append(p._boosters[e['path']](X).astype(np.float32))
    for tp, e in entries:
        f = out / _scale_name(tp, e)
        np.save(f, np.concatenate(parts[e['path']]))
        e['scale'], e['origin'] = f, f'{work.name}/features/2025.parquet: оценки модели выгрузки'


def locate(p: predmod.Predictor, work: Path, out: Path, types: list[str] | None = None) -> list[str]:
    """Каждому зерну — шкала его модели на входах сервиса (`e['scale']`, откуда — `e['origin']`).

    Сети проверяются первыми (дёшево): шкала из `<work>/roll`, веса совпадают с выгрузкой. Шкалы
    бустингов считаются в `out`, только если с сетями всё в порядке. Возвращает, чего не нашлось.
    """
    p._ensure()
    miss, boost = [], []
    for tp in types or list(p.entries):
        p._importance.pop(tp, None)
        for e in p.entries[tp]:
            e.pop('scale', None)
            e.pop('origin', None)
            p._scales.pop((tp, e['path']), None)
            if e['family'] != 'tcn':
                boost.append((tp, e))
                continue
            f, pt = (work / 'roll' / f'{e["from_run"]}{tail}' for tail in (f'_{tp}_val.npy', '.pt'))
            if not (f.exists() and pt.exists()):
                miss.append(f'шкала сети {tp}/{e["key"]}: нужны {f.name} и {pt.name} в {work / "roll"} '
                            f'(seqmodel.py --score-val --init {e["from_run"]}, Н20)')
            elif not _same_net(pt, e['path']):
                miss.append(f'шкала сети {tp}/{e["key"]}: {pt} — не та сеть, что {e["path"]}')
            else:
                e['scale'], e['origin'] = f, f'{work.name}/roll/{f.name}'
    if miss:
        return miss
    if boost:
        own_scales(p, work, out, boost)
    n = pl.scan_parquet(work / 'features' / '2025.parquet').select(pl.len()).collect().item()
    for tp in types or list(p.entries):
        for e in p.entries[tp]:
            m = len(np.load(e['scale'], mmap_mode='r'))
            if m != n:
                miss.append(f'шкала {tp}/{e["key"]} ({e["origin"]}): {m} строк при витрине 2025 {n}')
    return miss


def _importance(tp: str, es: list[dict], features: list[str]) -> dict[str, float]:
    """Важность признаков типа — среднее по бустингам выгрузки (топ-20 каждого, как в отчётах прогонов)."""
    from train import importance
    boost = [e for e in es if e['family'] != 'tcn']
    imp: dict[str, float] = {}
    for e in boost:
        if e['family'] == 'xgb':
            import xgboost as xgb
            m = xgb.Booster()
            m.load_model(str(e['path']))
        else:
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(str(e['path']))
        for name, w in importance(m, e['family'], features):
            imp[name] = round(imp.get(name, 0.0) + float(w) / len(boost), 4)
    return dict(sorted(imp.items(), key=lambda x: -x[1]))


def _models_missing(p: predmod.Predictor, types: list[str] | None = None) -> list[str]:
    return [f'модель {e["path"]}' for tp in types or list(p.entries) for e in p.entries[tp] if not e['path'].exists()]


def _pack(p: predmod.Predictor, types: list[str], src: Path, dst: Path, manifest: dict) -> int:
    """Модели, шкалы и важность типов `types` из выгрузки `src` в папку пакета `dst`."""
    (dst / 'scales').mkdir(parents=True, exist_ok=True)
    imp, size, origin = {}, 0, {}
    for tp in types:
        for e in p.entries[tp]:
            to = dst / e['path'].relative_to(src)
            to.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(e['path'], to)
            sc = dst / 'scales' / _scale_name(tp, e)
            if e['scale'] != sc:
                shutil.copy2(e['scale'], sc)
            origin[f'{tp}/{e["key"]}'] = e['origin']
            size += to.stat().st_size + sc.stat().st_size
        imp[tp] = _importance(tp, p.entries[tp], p.features)
    manifest = dict(manifest)
    manifest['bundle'] = {**manifest.get('bundle', {}), 'scales': origin}
    (dst / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    (dst / 'importance.json').write_text(json.dumps(imp, ensure_ascii=False), encoding='utf-8')
    return size


def build(dest: Path, work: Path | None = None, export: Path | None = None,
          settings: Path | None = None) -> dict:
    work = Path(work or config.WORK)
    export = Path(export or work / 'export')
    settings = Path(settings or config.ML / 'settings')
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'{dest} не пустая — пакет собирается в новую папку')
    p = predmod.Predictor(export=export, work=work)
    miss = _models_missing(p) or locate(p, work, dest / 'scales')
    if miss:
        raise FileNotFoundError('не хватает для пакета: ' + '; '.join(miss))
    manifest = dict(p.manifest)
    manifest['bundle'] = {'built': datetime.now().isoformat(timespec='seconds'), 'from': export.name,
                          'inputs': work.name}
    size = _pack(p, list(p.entries), export, dest, manifest)
    shutil.copy2(work / 'features' / 'meta.json', dest / 'meta.json')
    (dest / 'history').mkdir(exist_ok=True)
    np.save(dest / 'history' / 'h_2025.npy', p.hours_2025())
    versions = {}
    for tp, rows in predmod.available_versions(export).items():
        for r in rows:
            n = r['number']
            src = predmod.version_dir(export, tp, n)
            p.use_version(tp, n)
            miss = _models_missing(p, [tp]) or locate(p, work, dest / f'{tp}_v{n}' / 'scales', [tp])
            if miss:
                raise FileNotFoundError(f'версия {tp} v{n}: ' + '; '.join(miss))
            m = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
            m['bundle'] = {'inputs': work.name}
            size += _pack(p, [tp], src, dest / f'{tp}_v{n}', m)
            versions.setdefault(tp, []).append(n)
        p.use_version(tp, None)
    (dest / 'settings').mkdir(exist_ok=True)
    for name in SETTINGS:
        if (settings / name).exists():
            shutil.copy2(settings / name, dest / 'settings' / name)
    return {'dest': str(dest), 'version': p.version, 'versions': versions, 'mb': round(size / 2 ** 20, 1),
            'inputs': work.name}


def _inputs_diff(frame: pl.DataFrame, work: Path, t: datetime, features: list[str]) -> int:
    """Ячейки, где признаки среза сервиса на момент t расходятся с витриной `work` (строка часа t − 1 ч)."""
    from core import hour_index
    f = work / 'features' / f'{(t - timedelta(hours=1)).year}.parquet'
    ref = (pl.scan_parquet(f).filter(pl.col('h') == hour_index(t) - 1).select(['object_id'] + features)
           .collect())
    both = frame.select(['object_id'] + features).join(ref, on='object_id', suffix='_ref')
    assert both.height, f'в {f} нет строк часа {t - timedelta(hours=1)} по объектам среза'
    bad = 0
    for c in features:
        x, y = (both[k].to_numpy().astype(np.float64) for k in (c, c + '_ref'))
        bad += int((~(np.isclose(x, y, rtol=1e-4, atol=1e-4) | (np.isnan(x) & np.isnan(y)))).sum())
    return bad


def check(dest: Path, when: str = '2026-01-04T12:00', work: Path | None = None,
          export: Path | None = None, store=None) -> dict:
    """Пакет и выгрузка дают одни оценки на срезе (у каждой версии каждого типа), а входы сервиса —
    те, на которых посчитаны шкалы (`входы` — число расходящихся ячеек)."""
    import features as ft
    import snapshot as snapmod
    import storage
    from core import hour_index
    work = Path(work or config.WORK)
    export = Path(export or work / 'export')
    t = datetime.fromisoformat(when)
    a = predmod.Predictor(export=Path(dest))
    b = predmod.Predictor(export=export, work=work)
    assert a.bundle and a.work is None, 'пакет читается без work/'
    store = store or storage.HotStore()
    cal = ft.calendar(max(ft.NH, hour_index(t) + 48))
    frame, seq = snapmod.snapshot(store, t, a.meta, cal, seq=a.has_nets())
    out = {'входы': _inputs_diff(frame, work, t, a.features)}
    with tempfile.TemporaryDirectory() as tmp:
        miss = locate(b, work, Path(tmp) / 'main')
        assert not miss, miss
        main = {e['path']: (e['scale'], e['origin']) for es in b.entries.values() for e in es}
        runs = [(None, None)] + [(tp, r['number']) for tp, rows in predmod.available_versions(Path(dest)).items()
                                 for r in rows]
        for tp, n in runs:
            if tp:
                a.use_version(tp, n)
                b.use_version(tp, n)
                miss = locate(b, work, Path(tmp) / f'{tp}_v{n}', [tp])
                assert not miss, miss
            pa, pb = a.predict(frame, seq), b.predict(frame, seq)
            for k in ([tp] if tp else config.TYPES):
                out[f'{k}' + (f'_v{n}' if tp else '')] = float(np.abs(pa[k] - pb[k]).max())
            if tp:
                a.use_version(tp, None)
                b.use_version(tp, None)
                for e in b.entries[tp]:
                    e['scale'], e['origin'] = main[e['path']]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Пакет модели для контейнера (INTEGRATION §13.6)')
    ap.add_argument('dest', type=Path)
    ap.add_argument('--check', action='store_true', help='сравнить оценки пакета и выгрузки на срезе')
    ap.add_argument('--when', default='2026-01-04T12:00')
    ap.add_argument('--export', type=Path, help='выгрузка main (по умолчанию <--work>/export)')
    ap.add_argument('--work', type=Path,
                    help='рабочая папка, чьи витрина и ряды совпадают со входами сервиса (по умолчанию TF_WORK)')
    ap.add_argument('--hot', type=Path, help='горячий журнал для --check (сервис в это время остановлен)')
    args = ap.parse_args()
    if args.check:
        import storage
        store = storage.HotStore(args.hot) if args.hot else None      # срез пишет вид ev — не только чтение
        diff = check(args.dest, args.when, work=args.work, export=args.export, store=store)
        print(json.dumps(diff, ensure_ascii=False, indent=1))
        sys.exit(0 if max(diff.values()) < 1e-6 else 1)
    print(json.dumps(build(args.dest, work=args.work, export=args.export), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
