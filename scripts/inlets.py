"""Inlet conditions: what the app needs from a server to forecast how rough each ocean inlet is,
hour by hour for two days, in one small JSON file (inlets.json, next to the currents tiles).

  python3 scripts/inlets.py --out site                  # every inlet in scripts/inlets.json
  python3 scripts/inlets.py --out d --now 1763150400000   # pretend it is then (tests)

For each inlet (scripts/inlets.json, the same list as the app's src/data/inlets.json):
  - the waves offshore of it, hourly from 3 hours ago to 48 hours ahead: NOAA's GFS-Wave
    (WAVEWATCH III, the Atlantic 0.16° grid, atlocn.0p16, on AWS Open Data, bucket
    noaa-gfs-bdp-pds): the combined height, the peak period and direction, the wind sea and the
    two biggest swells (height, period, direction each), and the model's wind. Each field is read
    by byte range (the .idx says where it is); nothing is downloaded whole. The values are packed
    as JPEG 2000, which Pillow decodes (grib2.py).
  - its tide station's highs and lows (NOAA CO-OPS predictions, MLLW), and its current station's
    slacks and strongest flood and ebb, where it has one (CO-OPS current predictions)
  - the nearest wave buoy's last day (NDBC realtime2, its newest rows only, by byte range): the
    "right now" check. NDBC serves no CORS headers, so the app can't ask it itself.
The app does the rest (the waves on the bar, the rating, the words): src/lib/inlets.

Format (all times ms since 1970 UTC):
  { v: 1, updated, run (the model cycle), t0, step (3600000), n,
    fields: ['hs', 'tp', 'dp', 'wsh', 'wst', 'wsd', 's1h', 's1t', 's1d', 's2h', 's2t', 's2d', 'ws', 'wd'],
    scale: { field: divisor },     # value = stored / divisor (heights cm, periods 0.1 s, wind 0.1 m/s)
    cells: [{ at: [lat, lng], d: [[n values] per field] }],    # null where the model has none
    waves: { inletId: index into cells },
    tides: { station: [[t, ft × 100, 1 high | 0 low], ...] },
    currents: { station: { flood: deg, ebb: deg, ev: [[t, 'f' | 'e' | 's', kn × 100], ...] } },
    buoys: { id: { obs: [[t, wave height cm, dominant period 0.1 s, direction, wind 0.1 m/s, wind dir], ...] } },
    missing: { what: [ids] } }
"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grib2  # noqa: E402

HOUR = 3600 * 1000
UA = {'User-Agent': 'inlet-conditions/1.0'}
GFS = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
GRID = 'atlocn.0p16'
COOPS = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
NDBC = 'https://www.ndbc.noaa.gov/data/realtime2/'
HERE = os.path.dirname(os.path.abspath(__file__))

# the fields kept, each from (variable, level) in the .idx, and how each is stored
FIELDS = [
    ('hs', 'HTSGW', 'surface', 100), ('tp', 'PERPW', 'surface', 10), ('dp', 'DIRPW', 'surface', 1),
    ('wsh', 'WVHGT', 'surface', 100), ('wst', 'WVPER', 'surface', 10), ('wsd', 'WVDIR', 'surface', 1),
    ('s1h', 'SWELL', '1 in sequence', 100), ('s1t', 'SWPER', '1 in sequence', 10), ('s1d', 'SWDIR', '1 in sequence', 1),
    ('s2h', 'SWELL', '2 in sequence', 100), ('s2t', 'SWPER', '2 in sequence', 10), ('s2d', 'SWDIR', '2 in sequence', 1),
    ('ws', 'WIND', 'surface', 10), ('wd', 'WDIR', 'surface', 1),
]


def http_get(url, headers=None, timeout=60, tries=3):
    """bytes, or None when it isn't there (404)"""
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout).read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            if i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(1.5 * (i + 1))


# ── GFS-Wave ──
def cycle_url(cycle_ms, fh, ext=''):
    d = dt.datetime.fromtimestamp(cycle_ms / 1000, dt.timezone.utc)
    return f'{GFS}/gfs.{d:%Y%m%d}/{d:%H}/wave/gridded/gfswave.t{d:%H}z.{GRID}.f{fh:03d}.grib2{ext}'


def pick_cycle(now, hours, get=http_get, back=5):
    """the newest cycle (00, 06, 12, 18 UTC) whose forecast reaches `hours` past now, or None"""
    c = (now // (6 * HOUR)) * 6 * HOUR
    for k in range(back):
        cyc = c - k * 6 * HOUR
        last = int(math.ceil((now + hours * HOUR - cyc) / HOUR))
        if last > 120:   # (hourly to 120 h, then every 3)
            continue
        if get(cycle_url(cyc, last, '.idx')) is not None:
            return cyc
    return None


def ranges_for(idx_rows):
    """the byte ranges to read for FIELDS, adjacent fields merged: [(start, end_exclusive, [(name, start, end)])]"""
    want = {(v, lv): name for name, v, lv, _ in FIELDS}
    parts = []
    for r in idx_rows:
        name = want.get((r['var'], r['level']))
        if name:
            parts.append((name, r['start'], r['end']))
    parts.sort(key=lambda p: p[1])
    merged = []
    for p in parts:
        if merged and merged[-1][1] == p[1] and p[2] is not None:
            merged[-1] = (merged[-1][0], p[2], merged[-1][2] + [p])
        else:
            merged.append((p[1], p[2], [p]))
    return merged


def read_hour(cycle_ms, fh, cells, get=http_get):
    """one forecast hour: { field: [value at each cell] } (NaN where the model has none)"""
    idx = get(cycle_url(cycle_ms, fh, '.idx'))
    if idx is None:
        raise FileNotFoundError(cycle_url(cycle_ms, fh, '.idx'))
    rows = grib2.parse_idx(idx.decode())
    out = {}
    for a, b, parts in ranges_for(rows):
        rng = f'bytes={a}-{"" if b is None else b - 1}'
        buf = get(cycle_url(cycle_ms, fh), {'Range': rng})
        for name, s, e in parts:
            f = grib2.decode(buf[s - a: None if e is None else e - a])
            out[name] = [f.at(la, ln) for la, ln in cells]
    return out


def waves(inlets, now, past=3, hours=48, get=http_get, workers=6, say=print):
    cyc = pick_cycle(now, hours, get)
    if cyc is None:
        raise RuntimeError('no GFS-Wave cycle reaches two days ahead')
    t0 = ((now - past * HOUR) // HOUR) * HOUR
    n = int((now + hours * HOUR - t0) // HOUR) + 1
    cells, index = [], {}
    for x in inlets:
        if not x.get('offshore'):
            continue
        k = (round(x['offshore'][0], 4), round(x['offshore'][1], 4))
        if k not in index:
            index[k] = len(cells)
            cells.append(k)
    fhs = [int((t0 + i * HOUR - cyc) // HOUR) for i in range(n)]
    say(f'GFS-Wave {dt.datetime.fromtimestamp(cyc / 1000, dt.timezone.utc):%Y-%m-%d %HZ}: f{fhs[0]:03d}–f{fhs[-1]:03d}, {len(cells)} cells')

    def one(fh):
        if fh < 0:
            return None
        try:
            return read_hour(cyc, fh, cells, get)
        except Exception as e:
            say(f'  f{fh:03d}: skipped ({type(e).__name__}: {e})')
            return None
    with ThreadPoolExecutor(workers) as pool:
        hrs = list(pool.map(one, fhs))
    data = [[[None] * n for _ in FIELDS] for _ in cells]
    for i, h in enumerate(hrs):
        if not h:
            continue
        for fi, (name, _, _, scale) in enumerate(FIELDS):
            for ci, v in enumerate(h.get(name, [])):
                if v == v and v is not None and abs(v) < 1e6:
                    data[ci][fi][i] = int(round(v * scale))
    waves_by_inlet = {x['id']: index[(round(x['offshore'][0], 4), round(x['offshore'][1], 4))] for x in inlets if x.get('offshore')}
    return {'run': int(cyc), 't0': int(t0), 'step': HOUR, 'n': n, 'cells': [{'at': list(c), 'd': data[i]} for i, c in enumerate(cells)], 'waves': waves_by_inlet,
            'hours_read': sum(1 for h in hrs if h)}


# ── NOAA CO-OPS: tides and currents ──
def gmt(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%Y%m%d %H:%M')


def parse_t(s):
    return int(dt.datetime.strptime(s[:16], '%Y-%m-%d %H:%M').replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def tide_url(station, begin, hours):
    return COOPS + '?' + urllib.parse.urlencode({'product': 'predictions', 'station': station, 'begin_date': gmt(begin), 'range': hours, 'datum': 'MLLW',
                                                  'interval': 'hilo', 'units': 'english', 'time_zone': 'gmt', 'format': 'json', 'application': '302waterways'})


def parse_hilo(j):
    rows = (j or {}).get('predictions') or []
    return [[parse_t(r['t']), int(round(float(r['v']) * 100)), 1 if r.get('type') == 'H' else 0] for r in rows if r.get('t') and r.get('v') not in (None, '')]


def current_url(station, bin_, begin, hours):
    q = {'product': 'currents_predictions', 'station': station, 'begin_date': gmt(begin), 'range': hours, 'interval': 'MAX_SLACK',
         'units': 'english', 'time_zone': 'gmt', 'format': 'json', 'application': '302waterways'}
    if bin_:
        q['bin'] = bin_
    return COOPS + '?' + urllib.parse.urlencode(q)


def parse_max_slack(j):
    cp = ((j or {}).get('current_predictions') or {}).get('cp') or []
    ev, flood, ebb = [], None, None
    for r in cp:
        try:
            v = float(r['Velocity_Major'])
        except (KeyError, TypeError, ValueError):
            continue
        t = (r.get('Type') or '').lower()
        ev.append([parse_t(r['Time']), {'flood': 'f', 'ebb': 'e', 'slack': 's'}.get(t, 's' if abs(v) < 0.3 else 'f' if v > 0 else 'e'), int(round(v * 100))])
        if flood is None and r.get('meanFloodDir') not in (None, ''):
            flood, ebb = float(r['meanFloodDir']), float(r['meanEbbDir'])
    return {'flood': flood, 'ebb': ebb, 'ev': ev}


def coops(inlets, now, get=http_get, workers=6, say=print):
    begin = ((now - 12 * HOUR) // HOUR) * HOUR
    tides = sorted({x['tide'] for x in inlets if x.get('tide')})
    curs = sorted({(x['current'], x.get('currentBin')) for x in inlets if x.get('current')})

    def tide(st):
        try:
            return st, parse_hilo(json.loads(get(tide_url(st, begin, 72)) or b'{}'))
        except Exception as e:
            say(f'  tide {st}: {type(e).__name__}: {e}')
            return st, []

    def cur(sb):
        try:
            return sb[0], parse_max_slack(json.loads(get(current_url(sb[0], sb[1], begin, 72)) or b'{}'))
        except Exception as e:
            say(f'  current {sb[0]}: {type(e).__name__}: {e}')
            return sb[0], {'ev': []}
    with ThreadPoolExecutor(workers) as pool:
        t = dict(pool.map(tide, tides))
        c = dict(pool.map(cur, curs))
    return {k: v for k, v in t.items() if v}, {k: v for k, v in c.items() if v.get('ev')}


# ── NDBC buoys ──
def parse_ndbc(text, now, keep_ms=26 * HOUR):
    """realtime2 rows (newest first) → [[t, wvht cm, dpd 0.1 s, mwd, wspd 0.1 m/s, wdir]] oldest first,
    the rows with a wave height, within keep_ms of now"""
    out = []
    num = lambda s, k=1: None if s in ('MM', '', None) else int(round(float(s) * k))
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) < 12:
            continue
        try:
            t = int(dt.datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), tzinfo=dt.timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
        if t < now - keep_ms:
            break
        if p[8] == 'MM':
            continue
        out.append([t, num(p[8], 100), num(p[9], 10), num(p[11]), num(p[6], 10), num(p[5])])
    return out[::-1]


def buoys(inlets, now, get=http_get, workers=6, say=print):
    ids = sorted({x['buoy'] for x in inlets if x.get('buoy')})

    def one(b):
        try:
            raw = get(NDBC + b.upper() + '.txt', {'Range': 'bytes=0-24575'})
            return b, parse_ndbc((raw or b'').decode('latin-1'), now)
        except Exception as e:
            say(f'  buoy {b}: {type(e).__name__}: {e}')
            return b, []
    with ThreadPoolExecutor(workers) as pool:
        got = dict(pool.map(one, ids))
    return {k: {'obs': v} for k, v in got.items() if v}


def build(inlets, now, get=http_get, say=print, hours=48, past=3):
    w = waves(inlets, now, past=past, hours=hours, get=get, say=say)
    tides, currents = coops(inlets, now, get=get, say=say)
    bz = buoys(inlets, now, get=get, say=say)
    missing = {
        'waves': sorted(x['id'] for x in inlets if x['id'] not in w['waves']),
        'tides': sorted({x['tide'] for x in inlets if x.get('tide') and x['tide'] not in tides}),
        'currents': sorted({x['current'] for x in inlets if x.get('current') and x['current'] not in currents}),
        'buoys': sorted({x['buoy'] for x in inlets if x.get('buoy') and x['buoy'] not in bz}),
    }
    return {
        'v': 1, 'updated': dt.datetime.fromtimestamp(now / 1000, dt.timezone.utc).isoformat(timespec='seconds'),
        'model': 'NOAA GFS-Wave (WAVEWATCH III) atlocn 0.16°', 'run': w['run'], 't0': w['t0'], 'step': w['step'], 'n': w['n'],
        'fields': [f[0] for f in FIELDS], 'scale': {f[0]: f[3] for f in FIELDS},
        'cells': w['cells'], 'waves': w['waves'], 'tides': tides, 'currents': currents, 'buoys': bz,
        'missing': {k: v for k, v in missing.items() if v},
    }


def default_list():
    """scripts/inlets.json here; in the app's repository (scripts/currents), its src/data/inlets.json"""
    here = os.path.join(HERE, 'inlets.json')
    app = os.path.join(HERE, '..', '..', 'src', 'data', 'inlets.json')
    return here if os.path.exists(here) or not os.path.exists(app) else app


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--list', default=default_list())
    ap.add_argument('--now', type=float, default=None, help='pretend it is this time (ms)')
    ap.add_argument('--hours', type=int, default=48)
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))
    t0 = time.time()
    now = int(a.now or time.time() * 1000)
    inlets = json.load(open(a.list))
    out = build(inlets, now, say=say, hours=a.hours)
    os.makedirs(a.out, exist_ok=True)
    body = json.dumps(out, separators=(',', ':'))
    with open(os.path.join(a.out, 'inlets.json'), 'w') as fh:
        fh.write(body)
    say(f'inlets.json: {len(inlets)} inlets, {len(out["cells"])} cells, {out["n"]} hours, {len(out["tides"])} tide and {len(out["currents"])} current stations, '
        f'{len(out["buoys"])} buoys, {len(body) / 1e3:.0f} kB; missing {out["missing"] or "nothing"}; {time.time() - t0:.0f} s')
    return 0 if out['cells'] else 1


if __name__ == '__main__':
    sys.exit(main())
