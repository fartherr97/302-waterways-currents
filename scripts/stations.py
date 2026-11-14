"""NOAA's tidal current prediction stations on our coasts, as one small index for the app
(currents/stations.json). The app asks NOAA (CO-OPS) itself for a station's predictions when it's
on screen; this only says where the stations are.

From the CO-OPS metadata API (all current prediction stations, every depth bin): those inside our
box, one bin each (the one nearest the surface: what a boat feels), and only stations that have
predictions (harmonic 'H' or subordinate 'S'; the weak-and-variable 'W' ones have none).
"""
import json
import urllib.request

from tiles import WEST, SOUTH, EAST, NORTH

META = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=currentpredictions&units=english'
FIELDS = ['id', 'name', 'lat', 'lng', 'bin', 'type', 'depth']


def pick(stations, box=(WEST, SOUTH, EAST, NORTH)):
    w, s, e, n = box
    best = {}
    for x in stations:
        if x.get('type') not in ('H', 'S'):
            continue
        lat, lng = x.get('lat'), x.get('lng')
        if lat is None or lng is None or not (s <= lat <= n and w <= lng <= e):
            continue
        d = x.get('depth')
        rank = (d if isinstance(d, (int, float)) else 999, x.get('currbin') or 0)
        cur = best.get(x['id'])
        if cur is None or rank < cur[0]:
            best[x['id']] = (rank, x)
    rows = []
    for _, x in sorted(best.values(), key=lambda r: r[1]['id']):
        d = x.get('depth')
        rows.append([x['id'], x['name'].strip(), round(float(x['lat']), 5), round(float(x['lng']), 5), int(x.get('currbin') or 1), x['type'],
                     round(float(d), 1) if isinstance(d, (int, float)) else None])
    return {'fields': FIELDS, 'rows': rows}


def build(updated):
    req = urllib.request.Request(META, headers={'User-Agent': 'currents-tiles/1.0'})
    data = json.loads(urllib.request.urlopen(req, timeout=120).read())
    out = pick(data['stations'])
    out['updated'] = updated
    return out
