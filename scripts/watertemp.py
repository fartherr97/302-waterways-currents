"""Water temperature for the 302 Waterways Water Temp layer (the app's components/map/weather/WaterTempLayer):

- the satellite surface temperature: MUR SST (GHRSST Level 4, NASA JPL; 0.01°, daily, no cloud
  gaps: the analysis fills them), from NOAA CoastWatch's ERDDAP (dataset jplMURSST41), public
  domain. The newest day (about a day behind). It covers the ocean, the sounds, the bays and the
  big lakes; its land mask leaves out the narrowest back bays and small lakes.
- the latest reading at every buoy and shore station that measures it: NDBC's latest_obs.txt
  (NDBC buoys, C-MAN and NOS stations), names from NDBC's activestations.xml.

  python3 scripts/watertemp.py --out site      → site/watertemp/: manifest.json, o.png,
                                                 t<x>_<y>.png, stations.json

The tiles: PNG, grey + alpha. Pixel (col, row) is the temperature at lon = west + col × res,
lat = north − row × res (the satellite's own grid points), as grey = round((°C − T0) / STEP);
alpha 255 where there's water, 0 where there isn't.
  t<x>_<y>  4° squares at 0.01° (400 × 400): x counts from 98°W, y from 23°N; only those with water
  o         the whole area at 0.05° (641 × 561), for the map zoomed out
manifest.json: { updated, day (the satellite's day, ms), t0, step, res, tile, west, south,
north, tiles: [keys], o: { west, north, res, w, h }, stations }.
stations.json: { updated, fields: [id, name, lat, lng, c, t], rows } (c: °C, t: ms of the reading).

Its own step in the workflow: trouble here never holds up the currents.
"""
import argparse
import datetime as dt
import json
import os
import re
import struct
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdf3 import CDF3          # noqa: E402
from remote import _open, UA   # noqa: E402
from tiles import _chunk         # noqa: E402

ERDDAPS = ['https://coastwatch.pfeg.noaa.gov/erddap', 'https://upwell.pfeg.noaa.gov/erddap']
DATASET = 'jplMURSST41'
LATEST_OBS = 'https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt'
ACTIVE = 'https://www.ndbc.noaa.gov/activestations.xml'

WEST, EAST, SOUTH, NORTH = -98.0, -66.0, 23.0, 51.0   # (from 23°N: the Keys well inside it, so the app's fade at the edge misses them)
TILE, RES, OV_RES = 4, 0.01, 0.05
N_T = round(TILE / RES)                                # 400 points a side
T0, STEP = -2.0, 0.15                                  # grey 0 → −2 °C … 255 → 36.25 °C
STATION_HOURS = 6                                      # a reading older than this is left out


# ── the satellite ──

def encode(c):
    """°C (NaN: no water) → (grey, alpha) uint8 arrays."""
    c = np.asarray(c, float)
    ok = np.isfinite(c)
    g = np.zeros(c.shape, np.uint8)
    g[ok] = np.clip(np.round((c[ok] - T0) / STEP), 0, 255).astype(np.uint8)
    return g, np.where(ok, 255, 0).astype(np.uint8)


def decode(g, a):
    return np.where(np.asarray(a) > 0, T0 + np.asarray(g, float) * STEP, np.nan)


def png_la(grey, alpha):
    """a grey + alpha PNG (colour type 4), Paeth-filtered rows (as tiles.encode_png)"""
    px = np.dstack([grey, alpha]).astype(np.int16)
    h, w, _ = px.shape
    a = np.zeros_like(px); a[:, 1:] = px[:, :-1]
    b = np.zeros_like(px); b[1:] = px[:-1]
    c = np.zeros_like(px); c[1:, 1:] = px[:-1, :-1]
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    filt = ((px - pred) & 255).astype(np.uint8).reshape(h, w * 2)
    raw = np.concatenate([np.full((h, 1), 4, np.uint8), filt], 1).tobytes()
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 4, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) + _chunk(b'IDAT', zlib.compress(raw, 9)) + _chunk(b'IEND', b'')


def griddap_url(base, lat0, lat1, lon0, lon1, stride=1, when='last'):
    """when: 'last' (the newest day) or that day's time, '2026-11-22T09:00:00Z'"""
    q = f'analysed_sst[({when})][({lat0:.2f}):{stride}:({lat1:.2f})][({lon0:.2f}):{stride}:({lon1:.2f})]'
    return f'{base}/griddap/{DATASET}.nc?' + urllib.parse.quote(q, safe='():,')


class _Bytes:
    def __init__(self, b):
        self.b = b

    def read_at(self, off, n):
        return self.b[off:off + n]


def read_grid(data):
    """An ERDDAP .nc answer → (lat ascending, lon ascending, °C [lat, lon] with NaN off the water, day ms)."""
    c = CDF3(_Bytes(data))
    lat = np.asarray(c.read('latitude'), float)
    lon = np.asarray(c.read('longitude'), float)
    t = float(np.asarray(c.read('time'), float).ravel()[0]) * 1000
    v = np.asarray(c.read('analysed_sst', (0,)), float)
    fill = c.vars['analysed_sst'].attrs.get('_FillValue')
    bad = ~np.isfinite(v) | (v < T0 - 1) | (v > 45)
    if fill is not None:
        bad |= np.isclose(v, float(np.asarray(fill).ravel()[0]))
    v = np.where(bad, np.nan, v)
    if lat[0] > lat[-1]:
        lat, v = lat[::-1], v[::-1]
    return lat, lon, v, t


def fetch(url, timeout=120):
    with _open(urllib.request.Request(url, headers=UA), timeout) as r:
        return r.read()


def fetch_grid(lat0, lat1, lon0, lon1, stride=1, when='last'):
    err = None
    for base in ERDDAPS:
        try:
            return read_grid(fetch(griddap_url(base, lat0, lat1, lon0, lon1, stride, when)))
        except Exception as e:   # the next server
            err = e
    raise err


def tile_box(x, y):
    """the tile's grid points: (lat0, lat1, lon0, lon1) inclusive, 400 of each"""
    lat0, lon0 = SOUTH + y * TILE, WEST + x * TILE
    return lat0, round(lat0 + TILE - RES, 2), lon0, round(lon0 + TILE - RES, 2)


def tiles_with_water(lat, lon, v):
    """from the overview: the tiles that hold any water"""
    out = set()
    wet = np.isfinite(v)
    la, lo = np.meshgrid(lat, lon, indexing='ij')
    tx = np.floor((lo[wet] - WEST) / TILE).astype(int)
    ty = np.floor((la[wet] - SOUTH) / TILE).astype(int)
    for x, y in zip(tx.tolist(), ty.tolist()):
        if 0 <= x < round((EAST - WEST) / TILE) and 0 <= y < round((NORTH - SOUTH) / TILE):
            out.add((x, y))
    return sorted(out)


def north_first(lat, v, n):
    """rows north → south, padded to n rows (a tile the server answered short)"""
    v = v[::-1]
    if v.shape[0] < n:
        v = np.vstack([np.full((n - v.shape[0], v.shape[1]), np.nan), v])
    return v[:n]


# ── the stations ──

def parse_latest(text, now_ms, hours=STATION_HOURS):
    """NDBC latest_obs.txt → [(id, lat, lng, °C, ms)] in the area, read within `hours`."""
    out = []
    cols = None
    for line in text.splitlines():
        if line.startswith('#'):
            if cols is None:
                cols = line.lstrip('#').split()
            continue
        f = line.split()
        if not cols or len(f) < len(cols):
            continue
        r = dict(zip(cols, f))
        try:
            lat, lng = float(r['LAT']), float(r['LON'])
            c = float(r['WTMP'])
            t = dt.datetime(int(r['YYYY']), int(r['MM']), int(r['DD']), int(r['hh']), int(r['mm']), tzinfo=dt.timezone.utc).timestamp() * 1000
        except (KeyError, ValueError):
            continue
        if not (SOUTH <= lat <= NORTH and WEST <= lng <= EAST) or not (-3 < c < 40):
            continue
        if now_ms - t > hours * 3600_000 or t - now_ms > 3600_000:
            continue
        out.append((r['STN'].upper(), lat, lng, c, int(t)))
    return out


def tidy_name(s):
    """NDBC's names: without the station numbers they carry, SHOUTED ones in ordinary case"""
    s = re.sub(r'\s+', ' ', (s or '').strip())
    s = re.sub(r'^\d{5,}\s*-\s*', '', s)        # ("8557380 - Lewes, DE": the NOS number)
    s = re.sub(r'\s*\(\d+\)$', '', s)           # ("Bethany Beach, DE (263)": the CDIP number)
    if s.isupper():
        s = s.title()
        s = re.sub(r"\b(Nm|Ne|Nw|Se|Sw|Usa)\b", lambda m: m.group(1).upper(), s)
        s = re.sub(r"\b([A-Z][a-z])$", lambda m: m.group(1).upper(), s)   # (", De" → ", DE")
        s = re.sub(r"'S\b", "'s", s)
    return s


def parse_names(xml_text):
    names = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return names
    for st in root.iter('station'):
        sid, name = (st.get('id') or '').upper(), st.get('name')
        if sid and name:
            names[sid] = tidy_name(name)
    return names


def stations(now_ms, updated, say):
    obs = parse_latest(fetch(LATEST_OBS, 60).decode('utf-8', 'replace'), now_ms)
    try:
        names = parse_names(fetch(ACTIVE, 60).decode('utf-8', 'replace'))
    except Exception as e:
        say(f'station names: none ({e})'); names = {}
    rows = [[sid, names.get(sid, f'Station {sid}'), round(lat, 4), round(lng, 4), round(c, 1), t] for sid, lat, lng, c, t in obs]
    rows.sort(key=lambda r: r[0])
    return {'updated': updated, 'source': 'NOAA National Data Buoy Center', 'fields': ['id', 'name', 'lat', 'lng', 'c', 't'], 'rows': rows}


def dumps(obj):
    return json.dumps(obj, separators=(',', ':'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', default='', help='tiles x_y,x_y (local tests)')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))
    t0 = time.time()
    out = os.path.join(a.out, 'watertemp')
    os.makedirs(out, exist_ok=True)
    now = time.time() * 1000
    updated = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

    # the stations first (small, and worth having even when the satellite's server is down)
    n_st = 0
    try:
        st = stations(now, updated, say)
        n_st = len(st['rows'])
        with open(os.path.join(out, 'stations.json'), 'w') as fh:
            fh.write(dumps(st))
        say(f'stations: {n_st} with a reading in the last {STATION_HOURS} h')
    except Exception as e:
        say(f'stations: none ({type(e).__name__}: {e})')

    # the overview, and from it the tiles with water
    stride = round(OV_RES / RES)
    lat, lon, v, day = fetch_grid(SOUTH, NORTH, WEST, EAST, stride)
    # the tiles from the overview's day, by its time (asked for as "the newest", a day that came
    # in during the run would put some tiles a day ahead of the manifest's)
    when = dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    ov = north_first(lat, v, len(lat))
    with open(os.path.join(out, 'o.png'), 'wb') as fh:
        fh.write(png_la(*encode(ov)))
    o = {'west': round(float(lon[0]), 4), 'north': round(float(lat[-1]), 4), 'res': OV_RES, 'w': int(ov.shape[1]), 'h': int(ov.shape[0])}
    want = tiles_with_water(lat, lon, v)
    if a.only:
        keep = {tuple(int(n) for n in k.split('_')) for k in a.only.split(',')}
        want = [t for t in want if t in keep]
    say(f'satellite day {dt.datetime.fromtimestamp(day / 1000, dt.timezone.utc):%Y-%m-%d}; overview {ov.shape}; {len(want)} tiles with water; {time.time() - t0:.0f} s')

    def one(xy):
        x, y = xy
        lat0, lat1, lon0, lon1 = tile_box(x, y)
        la, lo, vv, d = fetch_grid(lat0, lat1, lon0, lon1, when=when)
        if d != day:
            raise ValueError(f'a tile from another day ({d}, not {day})')
        grid = north_first(la, vv, N_T)
        if grid.shape[1] < N_T:
            grid = np.hstack([grid, np.full((N_T, N_T - grid.shape[1]), np.nan)])
        if not np.isfinite(grid).any():
            return None
        png = png_la(*encode(grid[:, :N_T]))
        with open(os.path.join(out, f't{x}_{y}.png'), 'wb') as fh:
            fh.write(png)
        return f't{x}_{y}', len(png)

    done, total = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for xy, r in zip(want, pool.map(lambda t: _try(one, t, say), want)):
            if r:
                done.append(r[0]); total += r[1]
    say(f'tiles: {len(done)} of {len(want)}, {total / 1e6:.1f} MB; {time.time() - t0:.0f} s')
    if not done:
        return 1
    man = {'updated': updated, 'day': int(day), 'source': 'MUR SST (NASA JPL, GHRSST) via NOAA CoastWatch', 't0': T0, 'step': STEP,
           'res': RES, 'tile': TILE, 'west': WEST, 'south': SOUTH, 'north': NORTH, 'tiles': sorted(done), 'o': o, 'stations': n_st}
    with open(os.path.join(out, 'manifest.json'), 'w') as fh:
        fh.write(dumps(man))
    return 0


def _try(fn, xy, say):
    try:
        return fn(xy)
    except Exception as e:   # one tile's trouble: that tile left out
        say(f'  tile {xy}: left out ({type(e).__name__}: {e})')
        return None


if __name__ == '__main__':
    sys.exit(main())
