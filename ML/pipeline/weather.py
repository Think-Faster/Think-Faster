"""Шаг 9. Погода по Москве: проверка, даёт ли она что-то сверх журнала (Ф5-5, SPEC §9).

Архив Open-Meteo (ERA5), точка — центр Москвы, часовой шаг, время местное. Два набора:

- `base` — то, что записано в ТЗ: температура, осадки, приземное давление;
- `ext` — то, что ближе к механике аварий в коллекторе: снег и его таяние, влажность грунта,
  порывы ветра, расход реки (GloFAS, суточный). Подтопление коллектора идёт не от дождя за сутки,
  а от талой воды и насыщенного грунта; порывы ветра — от них рвётся питание;
- `hum` — конденсат и сухость: влажность, точка росы, солнечная радиация, дефицит давления пара, а
  с ними производственный календарь (isdayoff.ru). Стенки коллектора держат температуру месячного
  среднего, и когда точка росы выше неё — на них выпадает роса: это заливает контакты и даёт отказы
  датчиков, не видные ни по дождю, ни по грунту. Обратный край того же ряда — сухой воздух, при
  котором горит легче. Календарь отвечает за людей: в выходные и праздники на объектах никого;
- `air` — качество воздуха (CAMS): CO, NO2, SO2, PM10, PM2.5. Гипотеза не «ряд увидит утечку» —
  городской фон утечку под землёй не видит, — а «ряд увидит застой». Загазованность в замкнутом
  объёме требует и утечки, и отсутствия проветривания; при инверсии над городом фон растёт по всем
  веществам разом, и тот же застой стоит в коллекторе. Поэтому главные признаки набора —
  превышение над собственным недельным фоном и длительность этого превышения.

Признаки общие для всех объектов — зависят только от часа, поэтому лежат отдельной таблицей
«час → погода» и приклеиваются к витрине по `h` (`train.py --weather base|ext|both`). Витрина при
этом не пересобирается: сравнение с прогоном без погоды идёт на тех же строках и тех же параметрах.

Значение за час h известно к концу этого часа — момент прогноза тот же, что у витрины, в будущее
погода не заглядывает. Суточный расход реки берётся за прошлые сутки по той же причине.

    python weather.py base     # скачать и собрать набор из ТЗ
    python weather.py ext      # скачать и собрать расширенный набор
    python weather.py base --build-only    # пересобрать признаки из скачанного
"""
import json
import sys
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import config
from features import NH, T0, window_ext, window_sum

LAT, LON = 55.7558, 37.6173   # Москва, Красная площадь
BACK = 60                     # суток до начала данных: окно осадков в 30 суток должно быть полным
START = T0 - timedelta(days=BACK)
END = config.DATA_END - timedelta(days=1)
ARCHIVE = ('https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
           '&start_date={a}&end_date={b}&hourly={vars}&timezone=Europe%2FMoscow')
FLOOD = ('https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}'
         '&start_date={a}&end_date={b}&daily=river_discharge')
PACKS = {'base': ['temperature_2m', 'precipitation', 'surface_pressure'],
         'ext': ['snow_depth', 'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm', 'wind_gusts_10m'],
         'hum': ['relative_humidity_2m', 'dew_point_2m', 'shortwave_radiation',
                 'vapour_pressure_deficit'],
         'air': ['carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'pm10', 'pm2_5']}
SHORT = {'temperature_2m': 'temp', 'precipitation': 'prec', 'surface_pressure': 'press',
         'snow_depth': 'snow', 'soil_moisture_0_to_7cm': 'soil0', 'soil_moisture_7_to_28cm': 'soil28',
         'wind_gusts_10m': 'gust', 'relative_humidity_2m': 'rh', 'dew_point_2m': 'dew',
         'shortwave_radiation': 'rad', 'vapour_pressure_deficit': 'vpd',
         'carbon_monoxide': 'co', 'nitrogen_dioxide': 'no2', 'sulphur_dioxide': 'so2',
         'pm10': 'pm10', 'pm2_5': 'pm25'}
AIR = ('https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}'
       '&start_date={a}&end_date={b}&hourly={vars}&timezone=Europe%2FMoscow')
CALENDAR = 'https://isdayoff.ru/api/getdata?year={y}'   # производственный календарь России
RAIN = 0.5  # мм/ч: ниже этого — морось, «дождь был» не считаем


def path(pack: str, ext: str) -> 'config.Path':
    return config.WORK / (f'weather.{ext}' if pack == 'base' else f'weather_{pack}.{ext}')


def get(url: str) -> dict:
    print('качаю', url, flush=True)
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.loads(r.read().decode('utf-8'))


def fetch(pack: str) -> dict:
    url = AIR if pack == 'air' else ARCHIVE
    raw = get(url.format(lat=LAT, lon=LON, a=START.date(), b=END.date(), vars=','.join(PACKS[pack])))
    if pack == 'ext':
        raw['flood'] = get(FLOOD.format(lat=LAT, lon=LON, a=START.date(), b=END.date()))['daily']
    if pack == 'hum':
        raw['dayoff'] = {y: get_text(CALENDAR.format(y=y))
                         for y in range(START.year, config.DATA_END.year + 1)}
    config.WORK.mkdir(parents=True, exist_ok=True)
    path(pack, 'json').write_text(json.dumps(raw, ensure_ascii=False), encoding='utf-8')
    print('скачано', len(raw['hourly']['time']), 'часов →', path(pack, 'json'), flush=True)
    return raw


def get_text(url: str) -> str:
    print('качаю', url, flush=True)
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read().decode('utf-8').strip()


def series(raw: dict, pack: str) -> dict[str, np.ndarray]:
    """Ряды по часам START … DATA_END. Пропуски (их единицы) заполняются предыдущим значением."""
    n = int((config.DATA_END - START).total_seconds() // 3600)
    assert datetime.fromisoformat(raw['hourly']['time'][0]) == START, raw['hourly']['time'][0]
    out = {}
    for src in PACKS[pack]:
        x = np.array([np.nan if v is None else v for v in raw['hourly'][src]], np.float32)
        x = np.concatenate([x, np.full(max(0, n - len(x)), np.nan, np.float32)])[:n]
        miss = int(np.isnan(x).sum())
        if miss:
            idx = np.maximum.accumulate(np.where(np.isnan(x), 0, np.arange(len(x))))
            x = x[idx]
            print(f'  {SHORT[src]}: заполнено {miss} пропусков', flush=True)
        out[SHORT[src]] = x
    if 'dayoff' in raw:
        # календарь по суткам: 1 — выходной или праздник; год START может начаться не с 1 января
        by_year = {int(y): v for y, v in raw['dayoff'].items()}
        days = []
        for y in sorted(by_year):
            days.extend(1.0 if c == '1' else 0.0 for c in by_year[y])
        skip = (START - datetime(min(by_year), 1, 1)).days
        x = np.repeat(np.array(days[skip:], np.float32), 24)
        out['dayoff'] = np.concatenate([x, np.zeros(max(0, n - len(x)), np.float32)])[:n]
    if 'flood' in raw:
        # суточный расход: в час h берём значение за прошлые сутки, сегодняшнего система ещё не знает
        d = np.array([np.nan if v is None else v for v in raw['flood']['river_discharge']], np.float32)
        days = np.repeat(d, 24)[:n + 24]
        out['river'] = np.concatenate([np.full(24, np.nan, np.float32), days])[:n]
    return out


def shift(x: np.ndarray, d: int) -> np.ndarray:
    return np.concatenate([np.full(d, np.nan, np.float32), x[:-d]])


def base_features(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    temp, prec, press = s['temp'], s['prec'], s['press']
    f = {'wx_temp': temp, 'wx_press': press, 'wx_prec_1h': prec}
    for w in (24, 168):
        f[f'wx_temp_mean_{w}h'] = window_sum(temp, w) / w
    f['wx_temp_min_24h'] = window_ext(temp, 24, np.min)
    f['wx_temp_max_24h'] = window_ext(temp, 24, np.max)
    f['wx_temp_delta_24h'] = temp - shift(temp, 24)
    # переходы через ноль: лёд и оттепель — основной сезонный механизм подтоплений
    cross = np.zeros_like(temp)
    cross[1:] = (np.sign(temp[1:]) != np.sign(temp[:-1])).astype(np.float32)
    f['wx_cross0_72h'] = window_sum(cross, 72)
    f['wx_thaw_72h'] = window_sum(np.maximum(temp, 0), 72)      # градусо-часы тепла: таяние
    f['wx_frost_168h'] = window_sum(np.maximum(-temp, 0), 168)  # градусо-часы мороза: запас льда
    for w in (6, 24, 72, 168, 720):
        f[f'wx_prec_{w}h'] = window_sum(prec, w)
    f['wx_prec_max1h_24h'] = window_ext(prec, 24, np.max)
    idx = np.arange(len(prec))
    last = np.maximum.accumulate(np.where(prec >= RAIN, idx, -2161))
    f['wx_since_rain'] = np.minimum(idx - last, 2160).astype(np.float32)
    for d in (3, 24):
        f[f'wx_press_delta_{d}h'] = press - shift(press, d)
    f['wx_press_min_24h'] = window_ext(press, 24, np.min)
    return f


def ext_features(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    snow, soil0, soil28, gust, river = s['snow'], s['soil0'], s['soil28'], s['gust'], s['river']
    f = {'wx_snow': snow, 'wx_soil0': soil0, 'wx_soil28': soil28, 'wx_gust': gust, 'wx_river': river}
    f['wx_snow_max_168h'] = window_ext(snow, 168, np.max)
    for d in (24, 72):
        # убыль снежного покрова: сколько талой воды ушло в грунт за сутки и за трое
        f[f'wx_snow_melt_{d}h'] = np.maximum(shift(snow, d) - snow, 0)
    f['wx_soil0_delta_24h'] = soil0 - shift(soil0, 24)
    f['wx_soil28_delta_168h'] = soil28 - shift(soil28, 168)
    f['wx_soil0_max_168h'] = window_ext(soil0, 168, np.max)
    for w in (24, 72):
        f[f'wx_gust_max_{w}h'] = window_ext(gust, w, np.max)
    f['wx_river_delta_24h'] = river - shift(river, 24)
    f['wx_river_delta_168h'] = river - shift(river, 168)
    f['wx_river_max_720h'] = window_ext(river, 720, np.max)
    if 'temp' in s:
        # отопительный сезон в Москве включают после пяти суток холоднее +8: горячая вода в трубах
        # под коллектором — отдельный механизм подтоплений, а рвёт трубы обычно на пуске и на отключении
        heat_on = (window_sum(s['temp'], 120) / 120 < 8).astype(np.float32)
        f['wx_heating'] = heat_on
        switch = np.zeros_like(heat_on)
        switch[1:] = (heat_on[1:] != heat_on[:-1]).astype(np.float32)
        f['wx_heating_switch_720h'] = window_sum(switch, 720)
        f['wx_melt_index'] = np.minimum(snow, 0.5) * window_sum(np.maximum(s['temp'], 0), 72)
    return f


def hum_features(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rh, dew, rad, vpd, temp = s['rh'], s['dew'], s['rad'], s['vpd'], s['temp']
    f = {'wx_rh': rh, 'wx_dew': dew, 'wx_rad': rad, 'wx_vpd': vpd}
    # стенки коллектора под землёй держат температуру примерно месячного среднего воздуха; когда
    # точка росы выше неё, на них выпадает конденсат — вода на контактах без единой капли дождя
    wall = window_sum(temp, 720) / 720
    cond = dew - wall
    f['wx_cond'] = cond
    f['wx_wall'] = wall
    for w in (24, 72, 168):
        f[f'wx_cond_hours_{w}h'] = window_sum((cond > 0).astype(np.float32), w)
    f['wx_cond_max_24h'] = window_ext(cond, 24, np.max)
    f['wx_rh_mean_24h'] = window_sum(rh, 24) / 24
    f['wx_rh_max_24h'] = window_ext(rh, 24, np.max)
    f['wx_vpd_mean_24h'] = window_sum(vpd, 24) / 24
    f['wx_vpd_max_72h'] = window_ext(vpd, 72, np.max)
    f['wx_rad_24h'] = window_sum(rad, 24)
    if 'dayoff' in s:
        off = s['dayoff']
        f['wx_dayoff'] = off
        f['wx_dayoff_run'] = window_sum(off, 72)          # сколько нерабочих часов подряд позади
        back = np.zeros_like(off)
        back[24:] = ((off[24:] == 0) & (off[:-24] == 1)).astype(np.float32)
        f['wx_back_to_work'] = window_sum(back, 24)       # первый рабочий день после выходных
    return f


def air_features(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Качество воздуха по центру Москвы: CO, NO2, SO2, пыль (Open-Meteo CAMS).

    Важно понимать, что этот ряд измеряет и чего не измеряет. Это **городской фон**, одна точка на
    весь город, а не воздух в конкретном коллекторе. Утечку под землёй он не увидит никогда.

    Полезен он другим. Загазованность в замкнутом объёме — это всегда две вещи сразу: что-то
    подтекает и **не выветривается**. Первое ряд не видит, зато второе видит хорошо: когда над
    городом стоит инверсия и воздух не перемешивается, фоновые концентрации ползут вверх все разом,
    на всех веществах и по всему городу. Тот же застой стоит и в коллекторе. Поэтому главными здесь
    сделаны не уровни, а **превышения над собственным недельным фоном** и то, сколько часов подряд
    это превышение держится: ряд используется как индикатор режима проветривания, а не как
    измеритель утечки.

    Побочно то же самое может пригодиться двум другим типам: CO и мелкая пыль поднимаются при
    горении (пожар), крупная пыль PM10 — это запылённость, от которой слепнут оптические датчики.
    Эти два применения проверяются заодно, отдельной гипотезы под них не строилось.
    """
    f = {}
    for k in ('co', 'no2', 'so2', 'pm10', 'pm25'):
        x = s[k]
        f[f'aq_{k}'] = x
        base = window_sum(x, 168) / 168          # недельный фон самого ряда
        f[f'aq_{k}_base'] = base
        # превышение над фоном в долях: безразмерно, поэтому сравнимо между веществами и сезонами
        over = np.divide(x, base, out=np.ones_like(x), where=base > 0) - 1.0
        f[f'aq_{k}_over'] = over
        f[f'aq_{k}_over_max_24h'] = window_ext(over, 24, np.max)
        # часы застоя: сколько за сутки и за трое было выше фона. Длительность важнее пика —
        # разовый выброс проветрится, а трое суток над фоном означают, что не проветривается ничего
        high = (over > 0.25).astype(np.float32)
        for w in (24, 72):
            f[f'aq_{k}_high_{w}h'] = window_sum(high, w)
    # согласованность: застой поднимает все вещества сразу, локальный выброс — одно
    f['aq_all_over'] = np.mean([f[f'aq_{k}_over'] for k in ('co', 'no2', 'so2', 'pm10')], axis=0)
    f['aq_all_high_24h'] = np.mean([f[f'aq_{k}_high_24h'] for k in ('co', 'no2', 'so2', 'pm10')],
                                   axis=0)
    # доля мелкой фракции: горение даёт мелкую пыль, пыление и песок — крупную
    f['aq_fine_share'] = np.divide(s['pm25'], s['pm10'], out=np.zeros_like(s['pm10']),
                                   where=s['pm10'] > 0)
    return f


def build(raw: dict, pack: str) -> None:
    s = series(raw, pack)
    if pack in ('ext', 'hum') and path('base', 'json').exists():   # нужна температура
        s.update(series(json.loads(path('base', 'json').read_text(encoding='utf-8')), 'base'))
    f = {'base': base_features, 'ext': ext_features, 'hum': hum_features,
         'air': air_features}[pack](s)
    lo = BACK * 24  # первые 60 суток были нужны только для окон
    cols = {'h': np.arange(NH, dtype=np.int32)}
    cols.update({k: v[lo:lo + NH].astype(np.float32) for k, v in f.items()})
    out = path(pack, 'parquet')
    pq.write_table(pa.table(cols), out, compression='zstd')
    print(f'{out}: {NH} часов, {len(cols) - 1} признаков', flush=True)
    print('  ' + ', '.join(k for k in cols if k != 'h'))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    pack = args[0] if args else 'base'
    raw = (json.loads(path(pack, 'json').read_text(encoding='utf-8'))
           if '--build-only' in sys.argv else fetch(pack))
    build(raw, pack)


if __name__ == '__main__':
    main()
