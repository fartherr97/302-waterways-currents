"""Build the currents tiles: NOAA's forecast systems' surface currents, now → +48 h, as the tiles
the app's Currents layer draws (tiles.py says the format), plus the station index.

  python3 scripts/build.py --out /some/dir                 # every model, every hour
  python3 scripts/build.py --out d --models cbofs,dbofs --hours 12 --every 2
  python3 scripts/build.py --out d --workers 4            # the hours in four processes

Writes <out>/<YYYYMMDDHH>/<tile>.png, <out>/manifest.json and <out>/stations.json: a folder to
publish as it is (the 302-waterways-currents repository publishes it to GitHub Pages every six
hours). Nothing is downloaded whole: each hour reads only the surface layer of u and v
(remote.py); the raw model data never touches the disk. Hour by hour, so memory holds one hour of
every model at a time (per worker).
"""
import argparse
import datetime as dt
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Model, MODELS, DEFAULT, HOUR       # noqa: E402
from tiles import LEVELS, OV, WEST, SOUTH, NORTH, Grid, tiles_over, tile_box, encode_png, manifest, dumps  # noqa: E402
import stations as station_index                      # noqa: E402


def hour_key(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%Y%m%d%H')


def fine_tiles(mesh, L):
    """The fine level's tiles where this model is finer than L['fine_m'] (on water)."""
    wet_tri = mesh.wet[mesh.tri].max(1) > 0.5
    small = wet_tri & (mesh.size < L['fine_m'])
    if not small.any():
        return set()
    cx = mesh.lon[mesh.tri[small]].mean(1); cy = mesh.lat[mesh.tri[small]].mean(1)
    tx = np.floor((cx - WEST) / L['tile']).astype(int); ty = np.floor((cy - SOUTH) / L['tile']).astype(int)
    ok = (tx >= 0) & (tx < L['cols']) & (ty >= 0) & (ty < L['rows'])
    return set(zip(tx[ok].tolist(), ty[ok].tolist()))


def overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


_JOB = {}   # what every hour needs (shared with forked workers)


def _hour(t):
    """One hour: every model's surface current read (in parallel), laid on every grid, written as
    PNGs. → (the manifest's entry for it or None, the bytes written, the bytes read from NOAA)."""
    J = _JOB
    before = sum(m.fetched for m in J['models'])
    todo = {mid: d[t] for mid, d in J['per_model'].items() if t in d}
    fields = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {mid: pool.submit(fn) for mid, fn in todo.items()}
        for mid, fu in futs.items():
            try:
                u, v = fu.result()
                fields[mid] = (u.astype(np.float32), v.astype(np.float32))
            except Exception as e:
                J['say'](f'  {hour_key(t)} {mid}: skipped ({type(e).__name__}: {e})')
    read = sum(m.fetched for m in J['models']) - before
    if not fields:
        return None, 0, read
    key = hour_key(t)
    os.makedirs(os.path.join(J['out'], key), exist_ok=True)
    keys, total = [], 0
    for g in J['grids']:
        rgba, any_ = g.composite(fields)
        if not any_:
            continue
        png = encode_png(rgba)
        total += len(png)
        with open(os.path.join(J['out'], key, g.key + '.png'), 'wb') as fh:
            fh.write(png)
        keys.append(g.key)
    J['say'](f'  {key}: {len(fields)} models, {len(keys)} tiles, {total / 1e6:.1f} MB; {time.time() - J["t0"]:.0f} s')
    return {'t': int(t), 'key': key, 'run': J['run'], 'tiles': keys, 'models': sorted(fields)}, total, read


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default=','.join(DEFAULT))
    ap.add_argument('--hours', type=int, default=48, help='forecast hours after each cycle')
    ap.add_argument('--every', type=int, default=1, help='keep every n-th hour (local tests)')
    ap.add_argument('--past', type=float, default=3, help='hours before now to keep')
    ap.add_argument('--no-stations', action='store_true')
    ap.add_argument('--now', type=float, default=None, help='pretend it is this time (ms), for tests')
    ap.add_argument('--workers', type=int, default=1, help='processes for the hours (forked; the build machine\'s cores)')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    now = int(a.now or time.time() * 1000)
    os.makedirs(a.out, exist_ok=True)
    say = (lambda *x: None) if a.quiet else (lambda *x: print(*x, flush=True))

    # the models that have a complete run, and their meshes
    models = []
    for mid in [m.strip() for m in a.models.split(',') if m.strip()]:
        if mid not in MODELS:
            say('unknown model', mid); continue
        try:
            m = Model(mid, now, hours=a.hours, verbose=not a.quiet)
            if m.cycle is None:
                say(f'{mid}: no complete run in the last day and a half: left out'); continue
            m.mesh()
            models.append(m)
            say(f'{mid}: run {dt.datetime.fromtimestamp(m.cycle / 1000, dt.timezone.utc):%Y-%m-%d %HZ}, bbox {tuple(round(x, 2) for x in m.mesh().bbox)}')
        except Exception as e:  # one model's trouble never stops the rest
            say(f'{mid}: left out ({type(e).__name__}: {e})')
    if not models:
        say('no models'); return 1

    # the grids: fine tiles where a model is fine, the middle level wherever there's a model, the overview
    grids = []
    F, M = LEVELS
    fine = set()
    for m in models:
        fine |= fine_tiles(m.mesh(), F)
    for (tx, ty) in sorted(fine):
        w, s, e, n = tile_box(F, tx, ty)
        grids.append(Grid(f"f{tx}_{ty}", w, n, F['res'], F['px'], F['px']))
    mid_t = set()
    for m in models:
        mid_t |= set(tiles_over(M, m.mesh().bbox))
    for (tx, ty) in sorted(mid_t):
        w, s, e, n = tile_box(M, tx, ty)
        grids.append(Grid(f"m{tx}_{ty}", w, n, M['res'], M['px'], M['px']))
    grids.append(Grid('o', WEST, NORTH, OV['res'], OV['w'], OV['h']))
    for g in grids:
        gb = (g.west, g.north - g.h * g.res, g.west + g.w * g.res, g.north)
        for m in models:
            if overlaps(gb, m.mesh().bbox):
                g.add(MODELS[m.id]['prio'], m.id, m.mesh())
    grids = [g for g in grids if g.maps]
    say(f'grids: {sum(g.key[0] == "f" for g in grids)} fine, {sum(g.key[0] == "m" for g in grids)} middle, overview; {time.time() - t0:.0f} s')

    # the hours: every model's, from a little before now to the end of the newest runs
    per_model = {m.id: dict(m.hours()) for m in models}
    all_t = sorted({t for d in per_model.values() for t in d if t >= now - a.past * HOUR})
    if a.every > 1:
        all_t = all_t[::a.every]
    mesh_read = sum(m.fetched for m in models)
    _JOB.update(per_model=per_model, grids=grids, out=a.out, run=now, say=say, t0=t0, models=models)
    if a.workers > 1:
        # (forked: the workers share the meshes and grids already built; each takes whole hours)
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            done = list(pool.imap(_hour, all_t))
    else:
        done = [_hour(t) for t in all_t]
    hours = [h for h, _, _ in done if h]
    total = sum(n for _, n, _ in done)

    info = [{'id': m.id, 'name': MODELS[m.id]['name'], 'run': int(m.cycle), 'bbox': [round(x, 3) for x in m.mesh().bbox]} for m in models]
    updated = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    with open(os.path.join(a.out, 'manifest.json'), 'w') as fh:
        fh.write(dumps(manifest(hours, info, updated)))
    if not a.no_stations:
        try:
            with open(os.path.join(a.out, 'stations.json'), 'w') as fh:
                fh.write(dumps(station_index.build(updated)))
        except Exception as e:
            say(f'stations: not refreshed ({e})')
    fetched = mesh_read + sum(r for _, _, r in done)
    say(f'done: {len(hours)} hours, {total / 1e6:.1f} MB of tiles, {fetched / 1e6:.0f} MB read from NOAA, {time.time() - t0:.0f} s')
    return 0 if hours else 1


if __name__ == '__main__':
    sys.exit(main())
