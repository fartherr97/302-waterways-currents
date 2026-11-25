"""When a step that makes part of the site doesn't go through, put back the copy of that part that's
published now. The site is made from scratch each run and replaces the whole published site, so
without this the part would be gone from the app until the next good run.

  python3 scripts/keep_last.py --out site watertemp   # watertemp/: o.png, its tiles, stations.json, manifest.json
  python3 scripts/keep_last.py --out site inlets      # inlets.json

The published site is SITE_URL (default this repository's GitHub Pages). A whole copy or none:
the manifest is written last, after everything it lists.
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request

SITE = os.environ.get('SITE_URL', 'https://fartherr97.github.io/302-waterways-currents/')


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': '302-waterways-currents'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _write(d, name, b):
    with open(os.path.join(d, name), 'wb') as fh:
        fh.write(b)


def keep_watertemp(out, site=SITE, say=print):
    d = os.path.join(out, 'watertemp')
    shutil.rmtree(d, ignore_errors=True)   # (what a step that stopped partway left)
    os.makedirs(d)
    man = fetch(site + 'watertemp/manifest.json')
    m = json.loads(man)
    files = ['o.png'] + [f'{t}.png' for t in m['tiles']]
    for f in files:
        _write(d, f, fetch(site + 'watertemp/' + f))
    try:
        _write(d, 'stations.json', fetch(site + 'watertemp/stations.json'))
    except Exception as e:   # the readings go stale anyway; the picture is the part worth keeping
        say(f'stations.json not kept ({type(e).__name__}: {e})')
    _write(d, 'manifest.json', man)
    say(f'watertemp/: kept the published copy from {m.get("updated")} ({len(files)} pictures)')


def keep_inlets(out, site=SITE, say=print):
    b = fetch(site + 'inlets.json')
    m = json.loads(b)
    _write(out, 'inlets.json', b)
    say(f'inlets.json: kept the published copy from {m.get("updated")}')


PARTS = {'watertemp': keep_watertemp, 'inlets': keep_inlets}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('part', choices=sorted(PARTS))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    PARTS[a.part](a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
