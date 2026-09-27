"""Синтетические базы для стенда и карты: объекты и пикеты, датчики, слои карты, люди, группы и
права, бригады и допуски, графики смен и отпусков на 2026 год, план обслуживания объектов и график
работ. Описание таблиц и что в них выдумано — docs/backend/домены-и-сущности.md, §13.

    python seed.py profile   # сессии снятия с охраны 2024–2025 из tf.duckdb → service_observed.csv
    python seed.py build     # всё остальное из справочников и service_observed.csv → *.csv, *.jsonl, seed.sql

profile нужен один раз: журнал в репозиторий не кладётся, а service_observed.csv кладётся. Путь к
базе — TF_DUCKDB (по умолчанию ML/work/tf.duckdb). build детерминирован: зёрна фиксированы, uuid —
uuid5 от ключа, повторный прогон даёт те же файлы.

Люди, телефоны и номера документов выдуманы. Места из названий датчиков заменены на «место N».
Координаты схематичные, в метрах, а не WGS84: пикет — 10 м трассы, коллекторы разложены сеткой.
"""
import csv
import json
import math
import os
import random
import re
import sys
import uuid
import zlib
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(os.path.abspath(__file__)).parent
ROOT = HERE.parents[2]
DATA = ROOT / 'docs' / 'dataset'
WORKS = ROOT / 'ML' / 'settings' / 'works_2026.csv'
NS = uuid.UUID('6f1c1d2e-7a55-4c1b-9d7e-2f5e0b7c9a11')
TODAY = date(2026, 9, 27)                       # день, на который считаются статусы и сроки допусков
STAMP = '2026-09-27T00:00:00+03:00'
YEAR = 2026
PK_M = 10                                       # длина пикета, м: смещения «+N» почти все меньше 10
# нерабочие праздничные дни 2026 года для режима 5/2 (с переносами)
HOLIDAYS = {date(2026, 1, d) for d in range(1, 12)} | {
    date(2026, 2, 23), date(2026, 3, 9), date(2026, 5, 1), date(2026, 5, 11), date(2026, 6, 12),
    date(2026, 11, 4), date(2026, 12, 31)}
# для профиля 2024–2025 — без переносов, этого хватает, чтобы отличить будни от выходных
FIXED_HOLIDAYS = {(1, d) for d in range(1, 9)} | {(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)}


def uid(key):
    return str(uuid.uuid5(NS, key))


def rng_for(key):
    return random.Random(zlib.crc32(key.encode()))


def read_csv(path, **kw):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, **kw))


def write_csv(name, rows, cols):
    with open(HERE / name, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, '') for c in cols})
    print(f'{name}: {len(rows)}')


# ---------------------------------------------------------------- профиль обслуживания по журналу

def profile():
    import duckdb
    db = os.environ.get('TF_DUCKDB', str(ROOT / 'ML' / 'work' / 'tf.duckdb'))
    c = duckdb.connect(db, read_only=True)
    rows = c.sql("""select object_id, collector_id, ts, armed from guard
                    where ts >= '2024-01-01' and ts < '2026-01-01' order by object_id, ts""").fetchall()
    out, open_ = [], {}
    for obj, coll, ts, armed in rows:
        t0 = open_.get(obj)
        if not armed:
            if t0 is None or ts - t0 > timedelta(hours=24):
                open_[obj] = ts                  # снятие; повторное снятие в открытой сессии не сдвигает начало
        elif t0 is not None:
            if ts - t0 <= timedelta(hours=24):
                out.append({'object_id': obj, 'collector_id': coll, 'starts_at': t0.isoformat(sep=' '),
                            'ends_at': ts.isoformat(sep=' ')})
            open_.pop(obj)
    write_csv('service_observed.csv', out, ['object_id', 'collector_id', 'starts_at', 'ends_at'])


# ---------------------------------------------------------------- названия: места → «место N»

PLACES = [
    r'Андреевская\s+набережная|Андреевская\s+наб\.?|Андр\.?\s*наб\.?|Анд\.\s*наб\.?|Анд\.\s*Н\.|Анд\.',
    r'Площадь\s+Гагарина|Пл\.\s*Гагарина',
    r'Гагаринский\s+тоннель|Гагар\.\s*тоннель|Гагар\.\s*т\.?|Гагаринский\s*2|Гагар-2|Гагаринский|Гагар\.?|Гаг\.',
    r'Новые\s+Чер[её]мушки|Н\.\s*Чер[её]мушки|Нов\.\s*Чер[её]мушки|Новочер[её]мушкинский|Новочер',
    r'Цюрупы',
    r'Профсоюзн[а-яё]*|Проф\.?(?![а-яё])',
    r'Донско[йм]',
    r'Никулино',
    r'Ломоносовский|Ломонос\.?',
    r'Зюзино',
    r'Академическ[а-яё]*|Академ\.?(?![а-яё])',
    r'Ленинский\s*83|Лен\.\s*83',
    r'Ленинский\s*131|Лен\.\s*131',
    r'Гарибальди|Гариб\.?',
    r'Беляево',
    r'Ясенево',
    r'Зелинского',
    r'Вавиловск[а-яё]*',
    r'Мосрентген',
    r'Коньково',
    r'Ком+унарка',
    r'Т\.\s*Стан',
    r'Крупск[а-яё]*',
    r'Иволга',
    r'Тэц-20|ТЭЦ-20',
    r'ЦДТ',
    r'\(26БК\)|26БК',
]
PLACE_RX = re.compile('|'.join(f'(?P<p{i}>(?<![А-Яа-яЁё]){p})' for i, p in enumerate(PLACES)))
# основы, которых после замены остаться не должно
LEFT_RX = re.compile(r'Цюруп|Гагар|Гаг\.|Черемуш|Черёмуш|Донск|Андр|Никулин|Ломонос|Зюзин|Академ|Ленинск|Лен\.|'
                     r'Гариб|Беляев|Проф|Ясенев|Зелинск|Вавилов|Мосрентген|Коньков|Коммунар|Комунар|Стан|'
                     r'Крупск|Иволг|Тэц|ЦДТ|Анд\.')


def clean(name):
    def sub(m):
        n = int(m.lastgroup[1:]) + 1
        tail = m.string[m.end():m.end() + 1]
        return f'место {n}' + (' ' if tail and tail.isalnum() else '')
    s = PLACE_RX.sub(sub, name or '')
    s = s.replace('[SECURE ZONE]', '[Охранная зона]')
    return re.sub(r'\s+', ' ', s).strip()


# ---------------------------------------------------------------- разбор пикета из названия датчика

PK_RX = re.compile(r'(?<![А-Яа-яЁёA-Za-z])(?:(Г)\s*(\d)\s*)?[Пп][Кк]\s*(\d+)(?:\s*\+\s*(\d+(?:[.,]\d+)?))?')


def parse_pk(name):
    """(ответвление_от, номер_ответвления, пикет, смещение_м) или None. Ответвление_от — пикет основной
    трассы, None — ответвление без привязки; номер_ответвления None — основная трасса."""
    ms = list(PK_RX.finditer(name or ''))
    if not ms:
        return None
    m0 = ms[0]
    off = lambda m: float(m.group(4).replace(',', '.')) if m.group(4) else 0.0
    if m0.group(1):
        return (None, int(m0.group(2)), int(m0.group(3)), off(m0))
    if len(ms) > 1 and ms[1].group(1) and ms[1].start() - m0.end() <= 2:
        return (int(m0.group(3)), int(ms[1].group(2)), int(ms[1].group(3)), off(ms[1]))
    return (None, None, int(m0.group(3)), off(m0))


def side_of(name):
    n = (name or '').lower()
    return 'L' if 'лев' in n else 'R' if 'прав' in n else ''


# ---------------------------------------------------------------- топология: объекты, трассы, пикеты

def role_of(o):
    n, lvl = o['name'], o['level']
    if lvl == 1:
        return 'district'
    if lvl == 2:
        return 'collector'
    low = n.lower()
    if 'шкаф' in low or 'щит' in low:
        return 'cabinet'
    if re.search(r'(?<![А-Яа-яЁё])ДП(?![А-Яа-яЁё])', n):
        return 'base'
    if o['kind'] == 'guardObject':
        return 'trace_guard'
    return 'control'


def letter_of(name):
    m = re.search(r'объект\s+([А-ЯЁ][а-яё]+)', name)
    return m.group(1) if m else None


def walk(n, x0, y0, h0, rng, sigma=0.0035, maxturn=0.025):
    """Плавная ломаная из n точек с шагом PK_M: курс меняется медленно, как у трассы под улицей."""
    pts, hs, x, y, h, t = [(x0, y0)], [h0], x0, y0, h0, 0.0
    for _ in range(n - 1):
        t = max(-maxturn, min(maxturn, 0.96 * t + rng.gauss(0, sigma)))
        h += t
        x, y = x + PK_M * math.cos(h), y + PK_M * math.sin(h)
        pts.append((x, y))
        hs.append(h)
    return pts, hs


def r1(v):
    return round(v, 1)


def pt(p):
    return [r1(p[0]), r1(p[1])]


def thin(pts, k):
    out = pts[::k]
    return out if out[-1] == pts[-1] else out + [pts[-1]]


def build_topology():
    objs = []
    for r in read_csv(next(DATA.glob('справочник_объектов*.csv'))):
        o = {'id': int(r['ид_объект']), 'level': int(r['иерархия_уровень']), 'parent': int(r['родитель']),
             'kind': r['вид_объекта'], 'name': r['диспетчерское_название_объекта']}
        o['role'] = role_of(o)
        objs.append(o)
    by_id = {o['id']: o for o in objs}
    district = next(o for o in objs if o['level'] == 1)
    district['parent'] = None                    # родителя 3831 в справочнике нет
    collectors = sorted((o for o in objs if o['level'] == 2), key=lambda o: o['id'])
    for o in objs:
        if o['level'] == 3:
            o['collector'] = o['parent']
            o['letter'] = letter_of(o['name'])
            o['trace'] = (o['parent'], o['letter'])
    sens = []
    for r in read_csv(next(DATA.glob('справочник_каналов*.csv'))):
        s = {'id': int(r['ид_канала_данных']), 'object_id': int(r['ид_объект']), 'system': r['тип_инж_системы'],
             'stype': r['тип_датчика'], 'tag': r['тег_инженерной_системы'], 'raw': r['название_датчика']}
        s['name'] = clean(s['raw'])
        s['pk'] = parse_pk(s['raw'])
        s['side'] = side_of(s['raw'])
        s['junction'] = bool(re.search(r'сопр', s['raw'] or '', re.I))
        sens.append(s)
    left = sorted({s['name'] for s in sens if LEFT_RX.search(s['name'])})
    assert not left, f'остались места в названиях: {left[:10]}'

    # трассы: коллектор × буква из названия объекта 3-го уровня
    traces = {}
    for o in objs:
        if o['level'] == 3 and o['trace'] not in traces:
            traces[o['trace']] = {'key': o['trace'], 'main': set(), 'br': defaultdict(set), 'objects': []}
        if o['level'] == 3:
            traces[o['trace']]['objects'].append(o['id'])
    for s in sens:
        o = by_id[s['object_id']]
        s['trace'] = o['trace']
        if s['pk']:
            frm, g, pk, _ = s['pk']
            t = traces[o['trace']]
            if g is None:
                t['main'].add(pk)
            else:
                t['br'][(frm, g)].add(pk)
    for t in traces.values():                    # ответвление без привязки — к единственному с тем же номером
        for (frm, g) in [k for k in t['br'] if k[0] is None]:
            anchors = {k[0] for k in t['br'] if k[1] == g and k[0] is not None}
            if len(anchors) == 1:
                t['br'][(anchors.pop(), g)] |= t['br'].pop((None, g))
                for s in sens:
                    if s['trace'] == t['key'] and s['pk'] and s['pk'][:2] == (None, g):
                        s['pk'] = (next(k[0] for k in t['br'] if k[1] == g), g) + s['pk'][2:]
        froms = {k[0] for k in t['br'] if k[0] is not None}
        allm = t['main'] | froms or {0}
        t['lo'], t['hi'] = min(allm), max(allm)

    # пикеты: принадлежат коллектору; у коллектора с несколькими трассами код с буквой трассы
    pickets, pk_index = [], {}
    for c in collectors:
        ctr = [t for k, t in traces.items() if k[0] == c['id']]
        multi = len(ctr) > 1
        ordinal = 0
        for t in ctr:
            pre = f"{t['key'][1]} " if multi else ''
            seq = [(None, None, pk) for pk in range(t['lo'], t['hi'] + 1)]
            for (frm, g) in sorted(t['br'], key=lambda k: (k[0] is None, k[0] or 0, k[1])):
                seq += [(frm, g, pk) for pk in range(0, max(t['br'][(frm, g)]) + 1)]
            for frm, g, pk in seq:
                ordinal += 1
                code = pre + (f'ПК{pk}' if g is None else
                              (f'ПК{frm} ' if frm is not None else '') + f'Г{g} ПК{pk}')
                p = {'id': c['id'] * 100000 + ordinal, 'object_id': c['id'], 'code': code, 'ordinal': ordinal,
                     'trace': t['key'][1], 'pk': pk, 'branch': '' if g is None else f'Г{g}',
                     'branch_from': '' if frm is None else frm, 'sensors': 0, 'junction': ''}
                pickets.append(p)
                pk_index[(t['key'], frm, g, pk)] = p
            assert ordinal < 100000
    for s in sens:
        s['picket'] = pk_index.get((s['trace'],) + s['pk'][:3]) if s['pk'] else None
        if s['picket']:
            s['picket']['sensors'] += 1
            if s['junction']:
                m = re.search(r'место \d+', s['name'])
                s['picket']['junction'] = 'сопряжение' + (f' с {m.group(0)}' if m else '')

    # геометрия трасс в локальных координатах коллектора, затем раскладка коллекторов сеткой
    geo = {}
    for c in collectors:
        ctr = [t for k, t in traces.items() if k[0] == c['id']]
        for i, t in enumerate(ctr):
            rng = rng_for(f"trace:{c['id']}:{t['key'][1]}")
            h0 = rng.uniform(0, 2 * math.pi)
            x0, y0 = (0.0, 0.0) if i == 0 else (1300.0 * i * math.cos(h0 + 1.3), 1300.0 * i * math.sin(h0 + 1.3))
            pts, hs = walk(t['hi'] - t['lo'] + 1, x0, y0, h0, rng)
            t['pts'] = {pk: pts[pk - t['lo']] for pk in range(t['lo'], t['hi'] + 1)}
            t['hd'] = {pk: hs[pk - t['lo']] for pk in range(t['lo'], t['hi'] + 1)}
            t['bpts'] = {}
            for j, (frm, g) in enumerate(sorted(t['br'], key=lambda k: (k[0] is None, k[0] or 0, k[1]))):
                at = frm if frm is not None else t['lo']
                sign = 1 if (g + j) % 2 else -1
                b, bh = walk(max(t['br'][(frm, g)]) + 1, *t['pts'][at], t['hd'][at] + sign * rng.uniform(1.2, 1.9),
                             rng, sigma=0.002)
                t['bpts'][(frm, g)] = (b, bh)
        xs = [p[0] for t in ctr for p in list(t['pts'].values()) + [q for b, _ in t['bpts'].values() for q in b]]
        ys = [p[1] for t in ctr for p in list(t['pts'].values()) + [q for b, _ in t['bpts'].values() for q in b]]
        geo[c['id']] = (min(xs), min(ys), max(xs), max(ys))
    gap, per_row, cy, order = 1500.0, 4, 0.0, [c['id'] for c in collectors]
    shift = {}
    for r0 in range(0, len(order), per_row):
        row = order[r0:r0 + per_row]
        hmax = max(geo[i][3] - geo[i][1] for i in row)
        cx = 0.0
        for i in row:
            x0, y0, x1, y1 = geo[i]
            shift[i] = (cx - x0, cy - hmax - y0 + (hmax - (y1 - y0)) / 2)
            cx += x1 - x0 + gap
        cy -= hmax + gap
    sites = {cid: 1 + k // per_row for k, cid in enumerate(order)}   # участок = ряд сетки
    for k, t in traces.items():
        dx, dy = shift[k[0]]
        t['pts'] = {pk: (x + dx, y + dy) for pk, (x, y) in t['pts'].items()}
        t['bpts'] = {kk: ([(x + dx, y + dy) for x, y in b], bh) for kk, (b, bh) in t['bpts'].items()}
    for p in pickets:
        t = traces[(p['object_id'], p['trace'])]
        if p['branch']:
            frm = p['branch_from'] if p['branch_from'] != '' else None
            b, bh = t['bpts'][(frm, int(p['branch'][1:]))]
            p['xy'], p['h'] = b[p['pk']], bh[p['pk']]
        else:
            p['xy'], p['h'] = t['pts'][p['pk']], t['hd'][p['pk']]
    return objs, by_id, district, collectors, traces, sens, pickets, sites


def place_objects(objs, by_id, traces, sens, pickets):
    """Геометрия объектов 3-го уровня: участок трассы по пикетам его датчиков или точка у трассы."""
    per_obj = defaultdict(list)
    for s in sens:
        if s['picket'] and not s['picket']['branch']:
            per_obj[s['object_id']].append(s['picket']['pk'])
    for o in objs:
        if o['level'] != 3:
            continue
        t = traces[o['trace']]
        pks = sorted(per_obj[o['id']])
        o['pk_from'], o['pk_to'] = (pks[0], pks[-1]) if pks else ('', '')
        if o['role'] in ('trace_guard', 'control') and pks and pks[-1] > pks[0]:
            line = [t['pts'][k] for k in range(pks[0], pks[-1] + 1)]
            o['geom'] = {'type': 'LineString', 'coordinates': [pt(p) for p in thin(line, 5)]}
            continue
        at = pks[len(pks) // 2] if pks else t['lo']
        (x, y), h = t['pts'][at], t['hd'][at]
        d = 30.0 if o['role'] == 'base' else 12.0
        o['xy'] = (x - d * math.sin(h), y + d * math.cos(h))
        o['geom'] = {'type': 'Point', 'coordinates': pt(o['xy'])}
    # датчики: пикет + смещение вдоль трассы + сторона; без пикета — вокруг точки объекта
    ring = defaultdict(int)
    for s in sens:
        p = s['picket']
        if p:
            (x, y), h = p['xy'], p['h']
            off, lat = s['pk'][3], {'L': 3.0, 'R': -3.0}.get(s['side'], 0.0)
            s['xy'] = (x + off * math.cos(h) - lat * math.sin(h), y + off * math.sin(h) + lat * math.cos(h))
            s['offset_m'] = off
            continue
        o = by_id[s['object_id']]
        if 'xy' in o:
            cx, cy = o['xy']
        else:
            c = o['geom']['coordinates']
            cx, cy = c[len(c) // 2]
        k = ring[o['id']]
        ring[o['id']] += 1
        a, r = k * 2.39996, 4.0 + 1.6 * math.sqrt(k)       # подсолнух вокруг точки объекта
        s['xy'] = (cx + r * math.cos(a), cy + r * math.sin(a))
        s['offset_m'] = ''


SYSTEMS = ['Пожарная охрана', 'Газовая охрана', 'Охранная подсистема', 'Диспетчерский контроль',
           'Температурная подсистема', 'Диагностическая подсистема']


def fc(level, obj_id, features):
    return {'type': 'FeatureCollection', 'tf': {'crs': 'schematic-m', 'level': level, 'object_id': obj_id},
            'features': features}


def feat(geom, **props):
    return {'type': 'Feature', 'geometry': geom, 'properties': props}


def map_layers(objs, district, collectors, traces, sens, pickets):
    layers = []
    xs = [p['xy'][0] for p in pickets]
    ys = [p['xy'][1] for p in pickets]
    m = 2000.0
    box = [[r1(min(xs) - m), r1(min(ys) - m)], [r1(max(xs) + m), r1(min(ys) - m)],
           [r1(max(xs) + m), r1(max(ys) + m)], [r1(min(xs) - m), r1(max(ys) + m)]]
    district['geom'] = {'type': 'Polygon', 'coordinates': [box + [box[0]]]}
    for c in collectors:
        lines = [[pt(p) for p in thin([t['pts'][k] for k in range(t['lo'], t['hi'] + 1)], 10)]
                 for kk, t in traces.items() if kk[0] == c['id']]
        c['geom'] = {'type': 'MultiLineString', 'coordinates': lines}
    # 1 — город
    f1 = [feat(district['geom'], id=district['id'], name=district['name'], level=1)]
    f1 += [feat(c['geom'], id=c['id'], name=c['name'], level=2) for c in collectors]
    f1 += [feat(o['geom'], id=o['id'], name=o['name'], level=3, role=o['role'], collector=o['collector'])
           for o in objs if o['level'] == 3 and o['role'] == 'base']
    layers.append((1, district['id'], 'city', fc(1, district['id'], f1)))
    for c in collectors:
        cid = c['id']
        ctr = [(kk, t) for kk, t in traces.items() if kk[0] == cid]
        # 2 — карта коллектора: трассы, ответвления, объекты, пикеты
        f2 = []
        for kk, t in ctr:
            f2.append(feat({'type': 'LineString', 'coordinates':
                            [pt(t['pts'][k]) for k in range(t['lo'], t['hi'] + 1)]},
                           kind='trace', trace=kk[1], pk_from=t['lo'], pk_to=t['hi']))
            for (frm, g), (b, _) in t['bpts'].items():
                f2.append(feat({'type': 'LineString', 'coordinates': [pt(p) for p in b]},
                               kind='branch', trace=kk[1], branch=f'Г{g}', branch_from=frm))
        f2 += [feat(o['geom'], kind='object', id=o['id'], name=o['name'], role=o['role'],
                    pk_from=o['pk_from'], pk_to=o['pk_to'])
               for o in objs if o['level'] == 3 and o['collector'] == cid]
        f2 += [feat({'type': 'Point', 'coordinates': pt(p['xy'])}, kind='picket', id=p['id'], code=p['code'],
                    sensors=p['sensors'], **({'junction': p['junction']} if p['junction'] else {}))
               for p in pickets if p['object_id'] == cid]
        layers.append((2, cid, 'object', fc(2, cid, f2)))
        # 3 — датчики
        f3 = [feat({'type': 'Point', 'coordinates': pt(s['xy'])}, id=s['id'], object_id=s['object_id'],
                   picket_id=s['picket']['id'] if s['picket'] else None, system=s['system'], stype=s['stype'],
                   name=s['name'])
              for s in sens if s['trace'][0] == cid]
        layers.append((3, cid, 'sensors', fc(3, cid, f3)))
        # 4 — внутренняя схема: трасса в линию, x = пикет × 10 м, дорожки по системам
        f4, yb = [], 0.0
        for kk, t in ctr:
            X = lambda pk: (pk - t['lo']) * PK_M
            f4.append(feat({'type': 'LineString', 'coordinates': [[0, yb], [X(t['hi']), yb]]},
                           kind='trace', trace=kk[1]))
            by = {}
            for j, ((frm, g), (b, _)) in enumerate(t['bpts'].items()):
                y = yb - 30 - 15 * j
                x0 = X(frm if frm is not None else t['lo'])
                by[(frm, g)] = (x0, y)
                f4.append(feat({'type': 'LineString', 'coordinates': [[x0, yb], [x0, y], [x0 + (len(b) - 1) * PK_M, y]]},
                               kind='branch', trace=kk[1], branch=f'Г{g}', branch_from=frm))
            for p in pickets:
                if p['object_id'] != cid or p['trace'] != kk[1]:
                    continue
                if p['branch']:
                    frm = p['branch_from'] if p['branch_from'] != '' else None
                    x0, y = by[(frm, int(p['branch'][1:]))]
                    p['sx'] = (x0 + p['pk'] * PK_M, y)
                else:
                    p['sx'] = (X(p['pk']), yb)
                f4.append(feat({'type': 'Point', 'coordinates': [r1(p['sx'][0]), r1(p['sx'][1])]},
                               kind='picket', id=p['id'], code=p['code']))
            for s in sens:
                if s['trace'] != kk or not s['picket']:
                    continue
                x, y = s['picket']['sx']
                lane = SYSTEMS.index(s['system']) if s['system'] in SYSTEMS else len(SYSTEMS)
                f4.append(feat({'type': 'Point', 'coordinates': [r1(x + s['pk'][3]), r1(y + 5 + 4 * lane)]},
                               kind='sensor', id=s['id'], system=s['system']))
            yb -= 80 + 15 * len(t['bpts'])
        layers.append((4, cid, 'schema', fc(4, cid, f4)))
    return layers


# ---------------------------------------------------------------- люди

M_FIRST = ['Алексей', 'Андрей', 'Антон', 'Артём', 'Борис', 'Вадим', 'Валерий', 'Василий', 'Виктор', 'Виталий',
           'Владимир', 'Геннадий', 'Георгий', 'Григорий', 'Денис', 'Дмитрий', 'Евгений', 'Егор', 'Иван', 'Игорь',
           'Илья', 'Кирилл', 'Константин', 'Леонид', 'Максим', 'Михаил', 'Никита', 'Николай', 'Олег', 'Павел',
           'Роман', 'Руслан', 'Сергей', 'Станислав', 'Степан', 'Тимур', 'Фёдор', 'Юрий', 'Ярослав']
F_FIRST = ['Анна', 'Валентина', 'Галина', 'Дарья', 'Екатерина', 'Елена', 'Ирина', 'Юлия', 'Людмила', 'Марина',
           'Надежда', 'Наталья', 'Ольга', 'Светлана', 'Татьяна', 'Вера', 'Ксения', 'Полина']
PATR = ['Александров', 'Алексеев', 'Андреев', 'Борисов', 'Васильев', 'Викторов', 'Владимиров', 'Дмитриев',
        'Иванов', 'Михайлов', 'Николаев', 'Олегов', 'Павлов', 'Петров', 'Сергеев', 'Юрьев', 'Фёдоров',
        'Геннадьев', 'Константинов', 'Романов']
SURN = ['Абрамов', 'Белов', 'Волков', 'Воронин', 'Гусев', 'Данилов', 'Егоров', 'Жуков', 'Зайцев', 'Ильин',
        'Калинин', 'Киселёв', 'Козлов', 'Комаров', 'Крылов', 'Кузнецов', 'Лебедев', 'Макаров', 'Медведев',
        'Морозов', 'Никитин', 'Орлов', 'Осипов', 'Поляков', 'Сорокин', 'Степанов', 'Тарасов', 'Титов',
        'Федотов', 'Филиппов', 'Фролов', 'Цветков', 'Шаров', 'Широков', 'Яковлев', 'Беляков', 'Громов',
        'Дроздов', 'Ершов', 'Журавлёв', 'Карпов', 'Лапин', 'Мельников', 'Носов', 'Панов', 'Рябов', 'Соболев',
        'Тихонов', 'Уваров', 'Хохлов', 'Чернов', 'Щукин', 'Герасимов', 'Денисов', 'Ефимов', 'Зуев', 'Исаев',
        'Кравец', 'Бондаренко', 'Клименко', 'Мартынов', 'Гаврилов', 'Суханов', 'Лукин', 'Быков', 'Трофимов']


class Names:
    def __init__(self):
        self.rng, self.used = random.Random(20260927), set()

    def make(self, female_share):
        while True:
            f = self.rng.random() < female_share
            s = self.rng.choice(SURN)
            if f and s.endswith(('ов', 'ев', 'ёв', 'ин')):
                s += 'а'
            first = self.rng.choice(F_FIRST if f else M_FIRST)
            mid = self.rng.choice(PATR) + ('на' if f else 'ич')
            mid = mid.replace('ьевна', 'ьевна').replace('ьевич', 'ьевич')
            if (s, first) not in self.used:
                self.used.add((s, first))
                return s, first, mid


# режимы: дни работы от 1 января 2026 и часы смены
REGIMES = {
    '5/2': ('08:00–17:00', lambda d, k: d.weekday() < 5 and d not in HOLIDAYS),
    '2/2': ('07:00–19:00', lambda d, k: ((d - date(YEAR, 1, 1)).days // 2) % 2 == k),
    '1/3': ('08:00–08:00+1', lambda d, k: (d - date(YEAR, 1, 1)).days % 4 == k),
}


def people(collectors, sites, observed):
    names, ppl, teams, brigades = Names(), [], defaultdict(list), []

    def add(login, position, team, regime, k=0, female=0.2, spec=(), unit='', brigade=None, leave=28):
        s, f, m = names.make(female)
        p = {'id': uid(f'user:{login}'), 'auth_user_id': uid(f'auth:{login}'), 'login': login,
             'last_name': s, 'first_name': f, 'middle_name': m, 'phone': f'+7000000{len(ppl) + 1:04d}',
             'position': position, 'team': team, 'regime': regime, 'cycle': k, 'shift': REGIMES[regime][0],
             'spec': list(spec), 'unit': unit, 'brigade': brigade, 'leave_days': leave}
        ppl.append(p)
        teams[team].append(p)
        return p

    add('admin', 'Администратор системы', 'admin', '5/2')
    add('chief', 'Главный диспетчер', 'chief', '5/2', female=0.5)
    add('chief.deputy', 'Заместитель главного диспетчера', 'chief', '5/2', female=0.5)
    for k, sh in enumerate('АБВГ'):
        for j in (1, 2):
            add(f'disp.{"abvg"[k]}{j}', 'Старший диспетчер смены' if j == 1 else 'Диспетчер',
                f'Смена {sh}', '1/3', k=k, female=0.6)
    # режим линейной бригады — по доле сессий в выходные у объектов коллектора (2024–2025)
    wk = defaultdict(lambda: [0, 0])
    for r in observed:
        d = datetime.fromisoformat(r['starts_at']).date()
        wk[int(r['collector_id'])][0] += 1
        wk[int(r['collector_id'])][1] += not workday_hist(d)
    for c in collectors:
        n, w = wk[c['id']]
        c['regime'] = '2/2' if n and w / n >= 0.18 else '5/2'
        c['weekend_share'] = round(w / n, 3) if n else ''
        b = {'id': uid(f"brigade:{c['id']}"), 'name': f"Бригада {c['name']}", 'unit': f"Участок {sites[c['id']]}",
             'regime': c['regime'], 'collector': c['id']}
        brigades.append(b)
        size = 4 if c['regime'] == '2/2' else 3
        for j in range(size):
            p = add(f"line.{c['id']}.{j + 1}", 'Бригадир' if j == 0 else ('Электромонтёр' if j % 2 else 'Слесарь-ремонтник'),
                    b['name'], c['regime'], k=j % 2, spec=['MAINTENANCE'], unit=b['unit'], brigade=b, leave=35)
            if j == 0:
                b['leader'] = p['id']
    for s in range(1, 5):
        add(f'site.{s}', 'Инженер участка', 'Инженеры участков', '5/2', spec=['SITE_ENGINEER'],
            unit=f'Участок {s}', leave=35)
    extra = [('mobile', 'Мобильная бригада ППР и ТО АКМ', 5, ['MAINTENANCE', 'COMMS'], 35, 'Мастер'),
             ('power', 'Энергетики', 3, ['POWER'], 28, 'Энергетик'),
             ('comms', 'Связь и автоматика', 3, ['COMMS'], 28, 'Инженер связи и автоматики')]
    for code, name, n, spec, leave, pos in extra:
        b = {'id': uid(f'brigade:{code}'), 'name': name, 'unit': 'Район', 'regime': '5/2', 'collector': ''}
        brigades.append(b)
        for j in range(n):
            p = add(f'{code}.{j + 1}', pos if j == 0 or code != 'mobile' else 'Электромонтёр', name, '5/2',
                    spec=spec, brigade=b, leave=leave)
            if j == 0:
                b['leader'] = p['id']
    return ppl, teams, brigades


def workday_hist(d):
    return d.weekday() < 5 and (d.month, d.day) not in FIXED_HOLIDAYS


def days_of_year():
    d = date(YEAR, 1, 1)
    while d.year == YEAR:
        yield d
        d += timedelta(days=1)


def leaves(teams):
    """Отпуск 28 дней (под землёй 35) двумя частями: 14 летом и остаток в другой сезон. В одной смене,
    бригаде или команде одновременно в отпуске не больше одного человека."""
    rng = random.Random(2026)
    mondays = [d for d in days_of_year() if d.weekday() == 0]
    out = []
    for team, members in sorted(teams.items()):
        busy = []                                # занятые отрезки команды
        order = members[:]
        rng.shuffle(order)
        for p in order:
            parts = [14, p['leave_days'] - 14]
            own = []
            for i, n in enumerate(parts):
                summer = [d for d in mondays if 6 <= d.month <= 8]
                other = [d for d in mondays if d.month in (2, 3, 4, 5, 9, 10, 11) or (d.month == 1 and d.day > 11)]
                rng.shuffle(summer)
                rng.shuffle(other)
                cand = (summer + other) if i == 0 else (other + summer)
                for st in cand:
                    en = st + timedelta(days=n - 1)
                    if en.year != YEAR:
                        continue
                    if any(not (en < a or st > b) for a, b in busy):
                        continue
                    if any(abs((st - a).days) < 45 for a, _ in own):
                        continue
                    busy.append((st, en))
                    own.append((st, en))
                    out.append({'user': p, 'from': st, 'to': en, 'days': n})
                    break
                else:
                    raise RuntimeError(f'не уложили отпуск: {team} {p["login"]}')
    return out


def schedules(ppl, lv, chief):
    on_leave = defaultdict(set)
    for v in lv:
        d = v['from']
        while d <= v['to']:
            on_leave[v['user']['id']].add(d)
            d += timedelta(days=1)
    works = {}
    rows = []
    for p in ppl:
        fn = REGIMES[p['regime']][1]
        wd = [d for d in days_of_year() if fn(d, p['cycle']) and d not in on_leave[p['id']]]
        works[p['id']] = set(wd)
        runs, start, prev = [], None, None
        for d in wd:
            if start is None:
                start = prev = d
            elif (d - prev).days == 1 and p['regime'] != '1/3':
                prev = d
            else:
                runs.append((start, prev))
                start = prev = d
        if start:
            runs.append((start, prev))
        for a, b in runs:
            rows.append({'id': uid(f"sched:{p['login']}:{a}"), 'user_id': p['id'], 'login': p['login'],
                         'date_from': a, 'date_to': b, 'status': 1, 'status_name': 'WORKING',
                         'shift': p['shift'], 'source': 'график смен', 'changed_by': chief, 'changed_at': STAMP})
    for v in lv:
        p = v['user']
        rows.append({'id': uid(f"leave:{p['login']}:{v['from']}"), 'user_id': p['id'], 'login': p['login'],
                     'date_from': v['from'], 'date_to': v['to'], 'status': 3, 'status_name': 'ON_LEAVE',
                     'shift': '', 'source': 'график отпусков', 'changed_by': chief, 'changed_at': STAMP})
    rows.sort(key=lambda r: (r['login'], r['date_from']))
    return rows, works, on_leave


PERMITS = {  # вид → (срок, лет; префикс номера)
    'CONFINED_SPACE': (3, 'ОЗП'), 'GAS_HAZARD': (1, 'ГОР'), 'ELECTRICAL': (1, 'ЭБ')}


def permits(ppl):
    rng = random.Random(902)
    site_eng = {p['unit']: p['id'] for p in ppl if p['login'].startswith('site.')}
    chief = next(p['id'] for p in ppl if p['login'] == 'chief')
    out, n = [], 0
    for p in ppl:
        lead = p['position'] in ('Бригадир', 'Мастер')
        # ОЗП группы 2 у бригадира и у второго номера: звену CREW3 (§6.6) нужны двое с группой 2, у 2/2
        # второй номер в другой смене — так в каждой смене есть производитель работ
        second = p['login'].endswith('.2')
        if p['login'].startswith('line.') or p['login'].startswith('mobile.'):
            kinds = [('CONFINED_SPACE', 2 if lead or second else 1), ('GAS_HAZARD', ''), ('ELECTRICAL', 3 if lead else 2)]
        elif p['login'].startswith('site.'):
            kinds = [('CONFINED_SPACE', 3), ('GAS_HAZARD', ''), ('ELECTRICAL', 4)]
        elif p['login'].startswith('power.'):
            kinds = [('CONFINED_SPACE', 1), ('ELECTRICAL', 5 if lead else 4)]
        elif p['login'].startswith('comms.'):
            kinds = [('CONFINED_SPACE', 1), ('ELECTRICAL', 3)]
        else:
            continue
        for kind, level in kinds:
            years, pre = PERMITS[kind]
            until = TODAY + timedelta(days=rng.randint(50, 365 * years - 5))
            n += 1
            out.append({'id': uid(f'permit:{p["login"]}:{kind}'), 'user_id': p['id'], 'login': p['login'],
                        'kind': kind, 'level': level, 'valid_until': until,
                        'checked_at': until.replace(year=until.year - years),
                        'document_no': f'{pre}-{until.year - years}-{n:04d}',
                        'checked_by': site_eng.get(p['unit'], chief)})
    # предупреждение главному диспетчеру: три допуска кончаются в ближайшие 30 дней, один просрочен
    for i, (row, d) in enumerate(zip(rng.sample(out, 4), [10, 17, 24, -15])):
        row['valid_until'] = TODAY + timedelta(days=d)
        row['checked_at'] = row['valid_until'].replace(year=row['valid_until'].year - PERMITS[row['kind']][0])
        row['note'] = 'просрочен' if d < 0 else 'кончается в ближайшие 30 дней'
    return out


# ---------------------------------------------------------------- план обслуживания и график работ

def season(d):
    return (d.month % 12) // 3                   # 0 зима, 1 весна, 2 лето, 3 осень


def service_plan(observed, by_id, collectors, sites, ppl, works):
    """Сессии 2026 года: на каждый день каждого охраняемого объекта берётся случайный день 2024–2025
    того же типа (будни/выходные) и сезона, и его сессии переносятся на этот день."""
    sess = defaultdict(lambda: defaultdict(list))
    for r in observed:
        a, b = datetime.fromisoformat(r['starts_at']), datetime.fromisoformat(r['ends_at'])
        sess[int(r['object_id'])][a.date()].append((a, b))
    # библиотека — все дни 2024–2025, и пустые тоже: иначе редкий объект получал бы сессию каждый день
    lib = defaultdict(list)
    d = date(2024, 1, 1)
    while d.year < YEAR:
        lib[(workday_hist(d), season(d))].append(d)
        d += timedelta(days=1)
    brig = defaultdict(list)
    for p in ppl:
        if p['brigade'] and p['brigade']['collector'] != '':
            brig[p['brigade']['collector']].append(p)
    out = []
    for o in sorted(sess):
        rng = rng_for(f'plan:{o}')
        obj = by_id[o]
        coll = obj.get('collector') or o
        for day in days_of_year():
            wd = day.weekday() < 5 and day not in HOLIDAYS
            src = rng.choice(lib[(wd, season(day))])
            for a, b in sess[o].get(src, []):
                st = datetime.combine(day, a.time())
                en = st + (b - a)
                need = 1 if obj.get('role') == 'base' else 2
                crew, note = pick_crew(brig, collectors, sites, coll, day, works, need, rng)
                out.append({'object_id': o, 'object_name': obj['name'], 'collector_id': coll,
                            'role': obj.get('role', ''), 'starts_at': st.isoformat(sep=' ', timespec='minutes'),
                            'ends_at': en.isoformat(sep=' ', timespec='minutes'),
                            'hours': round((en - st).total_seconds() / 3600, 2),
                            'day_type': 'будни' if wd else 'выходной', 'source_date': src,
                            'brigade_id': crew[0]['brigade']['id'] if crew else '',
                            'crew': ' '.join(p['login'] for p in crew), 'note': note})
    return out


def pick_crew(brig, collectors, sites, coll, day, works, need, rng):
    own = [p for p in brig[coll] if day in works[p['id']]]
    if own:
        return rng.sample(own, min(need, len(own))), ''
    for c in collectors:                         # соседняя бригада того же участка
        if c['id'] != coll and sites[c['id']] == sites.get(coll):
            other = [p for p in brig[c['id']] if day in works[p['id']]]
            if len(other) >= need + 1:           # у соседей остаётся хотя бы один на своём коллекторе
                return rng.sample(other, need), f"из бригады {c['name']}"
    return [], 'нет людей в графике'


def work_schedule(ppl, works):
    rows = read_csv(WORKS, delimiter=';')
    mobile = [p for p in ppl if p['login'].startswith('mobile.')]
    out = []
    for r in rows:
        out.append({'work_id': r['work_id'], 'version': 1, 'object_id': r['object_id'], 'work_kind': r['work_kind'],
                    'incident_types': r['incident_types'], 'removed_sensor': r['removed_sensor'],
                    'starts_at': r['starts_at'], 'ends_at': r['ends_at'], 'source': 'ORGANIZER',
                    'comment': r['comment'], 'crew': ''})
    wid = 100
    for r in rows:
        m = re.search(r'ТО в месяцы ([\d ]+)', r['comment'])
        if not m:
            continue
        rng = rng_for(f"to:{r['work_id']}")
        k = re.search(r'Объект (\d+) графика ТО', r['comment']).group(1)
        for month in map(int, m.group(1).split()):
            days = [date(YEAR, month, d) for d in range(8, 22) if date(YEAR, month, d).weekday() in (1, 2, 3)
                    and date(YEAR, month, d) not in HOLIDAYS]
            rng.shuffle(days)
            day = next((d for d in days if sum(d in works[p['id']] for p in mobile) >= 3), days[0])
            crew = sorted(rng.sample([p['login'] for p in mobile if day in works[p['id']]], 3))
            wid += 1
            out.append({'work_id': wid, 'version': 1, 'object_id': r['object_id'], 'work_kind': 'ТО АКМ и ДУ',
                        'incident_types': '', 'removed_sensor': '', 'starts_at': f'{day} 09:00',
                        'ends_at': f'{day} 17:00', 'source': 'ORGANIZER',
                        'comment': f'Объект {k} графика ТО АКМ и ДУ, месяцы {m.group(1).strip()}; '
                                   f'день и часы выбраны генератором', 'crew': ' '.join(crew)})
    return out


# ---------------------------------------------------------------- группы и права

MASK = {'C': 1, 'R': 2, 'U': 4, 'D': 8, 'E': 16, 'I': 32, 'M': 64}
GRANTS = {  # группа → ресурс → действия; матрица ролей docs/common/права-и-аудит.md §3
    'engineers': {'objects': 'R', 'sensors': 'R', 'predictions': 'R', 'tasks': 'RU', 'incidents': 'R',
                  'schedule': 'R', 'assigned_objects': 'R', 'engineers': 'R', 'presence': 'R'},
    # readings — окно «Логи» по любому объекту; инженеру показания открывают его заявки (BFF /readings/scope)
    'dispatchers': {'objects': 'R', 'sensors': 'R', 'readings': 'R', 'predictions': 'RU', 'tasks': 'CRU',
                    'incidents': 'CRU', 'schedule': 'R', 'assigned_objects': 'R', 'engineers': 'R', 'presence': 'R'},
    'chief_dispatchers': {'predictions': 'RUE', 'model_settings': 'RU', 'schedule': 'CRUD',
                          'assigned_objects': 'CRUD', 'engineers': 'CRUD', 'groups': 'RU', 'users': 'R'},
}


def groups(ppl, teams):
    G = [('chief_dispatchers', 'Главные диспетчеры'), ('dispatchers', 'Диспетчеры'),
         ('dispatch_shift_a', 'Смена А'), ('dispatch_shift_b', 'Смена Б'), ('dispatch_shift_v', 'Смена В'),
         ('dispatch_shift_g', 'Смена Г'), ('engineers', 'Инженеры'), ('line_staff', 'Линейный персонал'),
         ('site_engineers', 'Инженеры участков'), ('mobile_brigade', 'Мобильная бригада ППР и ТО АКМ'),
         ('power_engineers', 'Энергетики'), ('comms_engineers', 'Связь и автоматика')]
    G += [(f'site_{s}', f'Участок {s}') for s in range(1, 5)]
    gs = [{'id': uid(f'group:{c}'), 'code': c, 'name': n, 'is_system': False} for c, n in G]
    nest = [('dispatchers', f'dispatch_shift_{x}') for x in 'abvg'] + [('dispatchers', 'chief_dispatchers')]
    nest += [('engineers', c) for c in ('line_staff', 'site_engineers', 'mobile_brigade', 'power_engineers',
                                        'comms_engineers')]
    mem = []
    for p in ppl:
        lg = p['login']
        codes = []
        if lg == 'admin':
            codes = ['admins']
        elif lg.startswith('chief'):
            codes = ['chief_dispatchers']
        elif lg.startswith('disp.'):
            codes = [f'dispatch_shift_{lg[5]}']
        elif lg.startswith('line.'):
            codes = ['line_staff', f"site_{p['unit'][-1]}"]
        elif lg.startswith('site.'):
            codes = ['site_engineers', f"site_{p['unit'][-1]}"]
        elif lg.startswith('mobile.'):
            codes = ['mobile_brigade']
        elif lg.startswith('power.'):
            codes = ['power_engineers']
        elif lg.startswith('comms.'):
            codes = ['comms_engineers']
        mem += [{'group': c, 'member_type': 1, 'member': p['id'], 'login': lg} for c in codes]
    mem += [{'group': a, 'member_type': 2, 'member': b, 'login': ''} for a, b in nest]
    # замыкание: предок — группа, в которую вложена потомок; права идут от предка к потомку
    codes = ['admins'] + [g['code'] for g in gs]
    closure = {(c, c): 0 for c in codes}
    changed = True
    while changed:
        changed = False
        for a, b in nest:
            for (x, y), dep in list(closure.items()):
                if x == b and ((a, y) not in closure or closure[(a, y)] > dep + 1):
                    closure[(a, y)] = dep + 1
                    changed = True
    grants = []
    for g, res in GRANTS.items():
        for r, acts in res.items():
            grants.append({'id': uid(f'grant:{g}:{r}'), 'group': g, 'resource': r, 'actions': acts,
                           'mask': sum(MASK[a] for a in acts),
                           'scope': {'engineers': 'свой коллектор или участок', 'dispatchers': 'весь район',
                                     'chief_dispatchers': 'весь район'}[g]})
    return gs, mem, closure, grants


def assignments(ppl, collectors, sites, chief):
    out = []
    by_site = defaultdict(list)
    for c in collectors:
        by_site[sites[c['id']]].append(c['id'])
    for p in ppl:
        lg = p['login']
        if lg.startswith('disp.'):
            objs = [c for s in ((1, 2) if lg.endswith('1') else (3, 4)) for c in by_site[s]]
            note = 'дежурство: участки 1–2' if lg.endswith('1') else 'дежурство: участки 3–4'
        elif lg.startswith('line.'):
            objs, note = [p['brigade']['collector']], 'своя трасса'
        elif lg.startswith('site.'):
            objs, note = by_site[int(lg[-1])], 'коллекторы участка'
        else:
            continue
        out += [{'user_id': p['id'], 'login': lg, 'object_id': o, 'assigned_by': chief, 'assigned_at': STAMP,
                 'note': note} for o in objs]
    return out


# ---------------------------------------------------------------- SQL под схему bff

def q(v):
    if v is None or v == '':
        return 'NULL'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def insert(f, table, cols, rows, conflict='DO NOTHING', target=''):
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        f.write(f'INSERT INTO {table} ({", ".join(cols)}) VALUES\n')
        f.write(',\n'.join('(' + ', '.join(r) + ')' for r in chunk))
        f.write(f'\nON CONFLICT {target}{conflict};\n\n')


def shift_of(text):
    """'07:00–19:00' → ('07:00', 12); '08:00–08:00+1' → ('08:00', 24)."""
    if not text:
        return None, None
    a, b = text.split('–')
    h = (int(b[:2]) * 60 + int(b[3:5]) - int(a[:2]) * 60 - int(a[3:5])) / 60 + (24 if b.endswith('+1') else 0)
    return a, int(h)


PERMIT_KIND = {'CONFINED_SPACE': 1, 'GAS_HAZARD': 2, 'ELECTRICAL': 3}
WORK_SOURCE = {'ORGANIZER': 1, 'CHIEF_DISPATCHER': 2, 'CARRIED_OVER': 3}


def write_sql(objs, pickets, sens, layers, ppl, gs, mem, closure, grants, brigades, profiles, assigned, sched,
              prm, ws):
    ts = q(STAMP)
    with open(HERE / 'seed.sql', 'w', encoding='utf-8', newline='\n') as f:
        f.write('-- Синтетические данные стенда: docs/backend/seed/seed.py build, руками не править.\n'
                '-- Выполнять после миграций и 001_seed_initial_data.sql (группа admins и ресурсы).\n'
                '-- Повторный прогон безопасен: ON CONFLICT DO NOTHING.\n\n'
                'SET search_path TO bff;\n\nBEGIN;\n\n')
        insert(f, 'objects', ['id', 'level', 'parent_id', 'kind', 'name', 'address', 'geometry_geojson', 'status',
                              'status_at', 'created_at', 'updated_at'],
               [[q(o['id']), q(o['level']), q(o['parent']), q(o['kind']), q(o['name']), 'NULL',
                 q(json.dumps(o['geom'], ensure_ascii=False, separators=(',', ':'))), '1', ts, ts, ts] for o in objs])
        insert(f, 'pickets', ['id', 'object_id', 'code', 'ordinal', 'geometry_geojson', 'created_at', 'updated_at'],
               [[q(p['id']), q(p['object_id']), q(p['code']), q(p['ordinal']),
                 q(json.dumps({'type': 'Point', 'coordinates': pt(p['xy'])})), ts, ts] for p in pickets])
        f.write("SELECT setval(pg_get_serial_sequence('bff.pickets', 'id'), (SELECT max(id) FROM pickets));\n\n")
        insert(f, 'sensors', ['id', 'object_id', 'picket_id', 'system', 'stype', 'tag', 'name', 'is_active',
                              'created_at', 'updated_at'],
               [[q(s['id']), q(s['object_id']), q(s['picket']['id'] if s['picket'] else None), q(s['system']),
                 q(s['stype']), q(s['tag']), q(s['name']), 'true', ts, ts] for s in sens])
        insert(f, 'map_layers', ['id', 'level', 'object_id', 'kind', 'geojson', 'updated_at'],
               [[q(uid(f'layer:{lv}:{oid}:{k}')), q(lv), q(oid), q(k),
                 q(json.dumps(g, ensure_ascii=False, separators=(',', ':'))), ts] for lv, oid, k, g in layers])
        insert(f, 'users', ['id', 'auth_user_id', 'last_name', 'first_name', 'middle_name', 'is_active',
                            'created_at', 'updated_at'],
               [[q(p['id']), q(p['auth_user_id']), q(p['last_name']), q(p['first_name']), q(p['middle_name']),
                 'true', ts, ts] for p in ppl])
        insert(f, 'groups', ['id', 'code', 'name', 'is_system', 'created_at', 'updated_at'],
               [[q(g['id']), q(g['code']), q(g['name']), 'false', ts, ts] for g in gs], target='(code) ')
        # ссылки на группы — по коду: admins заводит 001_seed_initial_data.sql со случайным id
        um = [m for m in mem if m['member_type'] == 1]
        f.write('INSERT INTO group_members (group_id, member_type, member_id, created_at)\n'
                'SELECT g.id, 1, v.m::uuid, ' + ts + '::timestamptz FROM (VALUES\n'
                + ',\n'.join(f"({q(m['group'])}, {q(m['member'])})" for m in um)
                + '\n) v(code, m) JOIN groups g ON g.code = v.code\nON CONFLICT DO NOTHING;\n\n')
        gm = [m for m in mem if m['member_type'] == 2]
        f.write('INSERT INTO group_members (group_id, member_type, member_id, created_at)\n'
                'SELECT p.id, 2, c.id, ' + ts + '::timestamptz FROM (VALUES\n'
                + ',\n'.join(f"({q(m['group'])}, {q(m['member'])})" for m in gm)
                + '\n) v(parent, child) JOIN groups p ON p.code = v.parent JOIN groups c ON c.code = v.child\n'
                  'ON CONFLICT DO NOTHING;\n\n')
        f.write('INSERT INTO group_closure (ancestor_id, descendant_id, depth)\n'
                'SELECT a.id, d.id, v.depth FROM (VALUES\n'
                + ',\n'.join(f'({q(a)}, {q(b)}, {dep})' for (a, b), dep in sorted(closure.items()))
                + '\n) v(anc, des, depth) JOIN groups a ON a.code = v.anc JOIN groups d ON d.code = v.des\n'
                  'ON CONFLICT (ancestor_id, descendant_id) DO NOTHING;\n\n')
        f.write('INSERT INTO access_grants (id, principal_type, principal_id, resource_id, permission_mask, '
                'created_at, updated_at)\n'
                'SELECT v.id::uuid, 2, g.id, r.id, v.mask, ' + ts + '::timestamptz, ' + ts + '::timestamptz FROM (VALUES\n'
                + ',\n'.join(f"({q(x['id'])}, {q(x['group'])}, {q(x['resource'])}, {x['mask']})" for x in grants)
                + '\n) v(id, code, res, mask) JOIN groups g ON g.code = v.code JOIN resources r ON r.code = v.res\n'
                  'ON CONFLICT (principal_type, principal_id, resource_id) DO NOTHING;\n\n')
        insert(f, 'brigades', ['id', 'name', 'unit', 'leader_id'],
               [[q(b['id']), q(b['name']), q(b['unit']), q(b.get('leader'))] for b in brigades])
        insert(f, 'engineer_profiles', ['user_id', 'brigade_id', 'phone', 'telegram', 'specialization', 'status'],
               [[q(r['user_id']), q(r['brigade_id']), q(r['phone']), 'NULL',
                 q('{' + ','.join(r['specialization'].split()) + '}'), q(r['status'])] for r in profiles])
        insert(f, 'assigned_objects', ['user_id', 'object_id', 'assigned_by', 'assigned_at', 'note'],
               [[q(a['user_id']), q(a['object_id']), q(a['assigned_by']), ts, q(a['note'])] for a in assigned])
        insert(f, 'schedule_entries', ['id', 'user_id', 'date_from', 'date_to', 'status', 'shift_start',
                                       'shift_hours', 'source', 'changed_by', 'changed_at'],
               [[q(s['id']), q(s['user_id']), q(str(s['date_from'])), q(str(s['date_to'])), q(s['status']),
                 q(shift_of(s['shift'])[0]), q(shift_of(s['shift'])[1]), q(s['source']), q(s['changed_by']), ts]
                for s in sched])
        insert(f, 'engineer_permits', ['id', 'user_id', 'kind', 'level', 'valid_until', 'document_no', 'checked_by',
                                       'checked_at'],
               [[q(r['id']), q(r['user_id']), q(PERMIT_KIND[r['kind']]), q(r['level']), q(str(r['valid_until'])),
                 q(r['document_no']), q(r['checked_by']), q(str(r['checked_at']))] for r in prm])
        # время графика — местное, как в works_2026.csv; created_by пуст: строки от организатора, не из интерфейса
        insert(f, 'work_schedule', ['work_id', 'version', 'object_id', 'work_kind', 'incident_types', 'removed_sensor',
                                    'starts_at', 'ends_at', 'source', 'comment', 'deleted', 'created_by', 'created_at'],
               [[q(int(w['work_id'])), q(w['version']), q(int(w['object_id']) if w['object_id'] else None),
                 q(w['work_kind']), q('{' + ','.join(w['incident_types'].split()) + '}'), q(w['removed_sensor']),
                 q(w['starts_at'].replace(' ', 'T') + ':00+03:00'), q(w['ends_at'].replace(' ', 'T') + ':00+03:00'), q(WORK_SOURCE[w['source']]), q(w['comment']),
                 'false', 'NULL', ts] for w in ws])
        f.write("SELECT setval('work_schedule_work_id_seq', (SELECT max(work_id) FROM work_schedule));\n\n")
        f.write('INSERT INTO rbac_version (id, value) VALUES (1, 1)\n'
                'ON CONFLICT (id) DO UPDATE SET value = rbac_version.value + 1;\n\nCOMMIT;\n')
    print('seed.sql:', round((HERE / 'seed.sql').stat().st_size / 1e6, 1), 'МБ')


# ---------------------------------------------------------------- сборка

def build():
    objs, by_id, district, collectors, traces, sens, pickets, sites = build_topology()
    place_objects(objs, by_id, traces, sens, pickets)
    layers = map_layers(objs, district, collectors, traces, sens, pickets)
    observed = read_csv(HERE / 'service_observed.csv')
    ppl, teams, brigades = people(collectors, sites, observed)
    chief = next(p['id'] for p in ppl if p['login'] == 'chief')
    lv = leaves(teams)
    sched, works, on_leave = schedules(ppl, lv, chief)
    prm = permits(ppl)
    gs, mem, closure, grants = groups(ppl, teams)
    assigned = assignments(ppl, collectors, sites, chief)
    plan = service_plan(observed, by_id, collectors, sites, ppl, works)
    ws = work_schedule(ppl, works)

    for c in collectors:
        c['site'] = sites[c['id']]
    write_csv('objects.csv', [dict(o, parent_id=o['parent'], geometry_geojson=json.dumps(o['geom'], ensure_ascii=False,
                                                                                        separators=(',', ':')),
                                   status=1, status_at=STAMP, site=o.get('site', sites.get(o.get('collector'), '')),
                                   trace=o.get('letter', '')) for o in objs],
              ['id', 'level', 'parent_id', 'kind', 'name', 'role', 'trace', 'site', 'regime', 'weekend_share',
               'pk_from', 'pk_to', 'status', 'status_at', 'geometry_geojson'])
    write_csv('pickets.csv', [dict(p, x_m=r1(p['xy'][0]), y_m=r1(p['xy'][1])) for p in pickets],
              ['id', 'object_id', 'code', 'ordinal', 'trace', 'pk', 'branch', 'branch_from', 'x_m', 'y_m',
               'sensors', 'junction'])
    write_csv('sensors.csv', [dict(s, picket_id=s['picket']['id'] if s['picket'] else '', is_active='true',
                                   x_m=r1(s['xy'][0]), y_m=r1(s['xy'][1])) for s in sens],
              ['id', 'object_id', 'picket_id', 'system', 'stype', 'tag', 'name', 'offset_m', 'side', 'is_active',
               'x_m', 'y_m'])
    with open(HERE / 'map_layers.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for lv_, oid, k, g in layers:
            f.write(json.dumps({'id': uid(f'layer:{lv_}:{oid}:{k}'), 'level': lv_, 'object_id': oid, 'kind': k,
                                'geojson': g}, ensure_ascii=False, separators=(',', ':')) + '\n')
    print('map_layers.jsonl:', len(layers))
    write_csv('users.csv', ppl, ['id', 'auth_user_id', 'login', 'last_name', 'first_name', 'middle_name', 'phone',
                                 'position', 'team', 'unit', 'regime', 'shift', 'leave_days'])
    write_csv('groups.csv', gs, ['id', 'code', 'name', 'is_system'])
    write_csv('group_members.csv', mem, ['group', 'member_type', 'member', 'login'])
    write_csv('group_closure.csv', [{'ancestor': a, 'descendant': b, 'depth': d} for (a, b), d in sorted(closure.items())],
              ['ancestor', 'descendant', 'depth'])
    write_csv('grants.csv', grants, ['id', 'group', 'resource', 'actions', 'mask', 'scope'])
    write_csv('brigades.csv', [dict(b, leader_id=b.get('leader', '')) for b in brigades],
              ['id', 'name', 'unit', 'leader_id', 'regime', 'collector'])
    profiles = []
    for p in ppl:
        if not p['spec']:
            continue
        st = 1 if TODAY in works[p['id']] else 4
        profiles.append({'user_id': p['id'], 'login': p['login'], 'brigade_id': p['brigade']['id'] if p['brigade'] else '',
                         'phone': p['phone'], 'specialization': ' '.join(p['spec']), 'status': st,
                         'status_name': 'AVAILABLE' if st == 1 else ('UNAVAILABLE, отпуск' if TODAY in on_leave[p['id']]
                                                                     else 'UNAVAILABLE, не его смена')})
    write_csv('engineer_profiles.csv', profiles, ['user_id', 'login', 'brigade_id', 'phone', 'specialization',
                                                  'status', 'status_name'])
    write_csv('engineer_permits.csv', prm, ['id', 'user_id', 'login', 'kind', 'level', 'valid_until', 'document_no',
                                            'checked_by', 'checked_at', 'note'])
    write_csv('assigned_objects.csv', assigned, ['user_id', 'login', 'object_id', 'assigned_by', 'assigned_at', 'note'])
    write_csv('schedule_entries.csv', sched, ['id', 'user_id', 'login', 'date_from', 'date_to', 'status',
                                              'status_name', 'shift', 'source', 'changed_by', 'changed_at'])
    write_csv('service_plan_2026.csv', plan, ['object_id', 'object_name', 'collector_id', 'role', 'starts_at',
                                              'ends_at', 'hours', 'day_type', 'source_date', 'brigade_id', 'crew',
                                              'note'])
    write_csv('work_schedule_2026.csv', ws, ['work_id', 'version', 'object_id', 'work_kind', 'incident_types',
                                             'removed_sensor', 'starts_at', 'ends_at', 'source', 'comment', 'crew'])
    write_sql(objs, pickets, sens, layers, ppl, gs, mem, closure, grants, brigades, profiles, assigned, sched,
              prm, ws)


if __name__ == '__main__':
    {'profile': profile, 'build': build}[sys.argv[1] if len(sys.argv) > 1 else 'build']()
