"""Tests for the inlet conditions step: GRIB2 decoding (simple and JPEG 2000 packing, the land
bitmap, the scanning), the .idx byte ranges, picking a model cycle, NOAA's tide, current and buoy
answers, and the whole file built from fake servers.   python3 -m unittest discover -s scripts -p 'test_*.py'"""
import datetime as dt
import io
import json
import os
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grib2  # noqa: E402
import inlets  # noqa: E402

HOUR = 3600 * 1000


def sec(n, body):
    return struct.pack('>IB', 5 + len(body), n) + body


def message(values, ni, nj, lat0=40.0, lon0=280.0, d=0.5, packing='simple', ref=0.0, D=2, nbits=12, mask=None, north_first=True, cat=0, par=3):
    """a GRIB2 message for a regular lat/lon grid (rows north → south unless north_first is False)"""
    vals = np.asarray(values, float).ravel()
    keep = np.ones(vals.size, bool) if mask is None else np.asarray(mask, bool).ravel()
    x = np.round((vals[keep] * 10 ** D) - ref).astype(np.int64)
    s1 = sec(1, b'\x00' * 16)
    la1, la2 = (lat0, lat0 - (nj - 1) * d) if north_first else (lat0 - (nj - 1) * d, lat0)
    g = struct.pack('>B', 6) + b'\x00' * 15 + struct.pack('>II', ni, nj) + b'\x00' * 8
    g += struct.pack('>iI', int(la1 * 1e6), int(lon0 * 1e6)) + b'\x30'
    g += struct.pack('>iI', int(la2 * 1e6), int((lon0 + (ni - 1) * d) * 1e6))
    g += struct.pack('>II', int(d * 1e6), int(d * 1e6)) + bytes([0 if north_first else 0x40])
    s3 = sec(3, b'\x00' + struct.pack('>I', vals.size) + b'\x00\x00' + struct.pack('>H', 0) + g)
    s4 = sec(4, struct.pack('>HH', 0, 0) + bytes([cat, par]) + b'\x00' * 20)
    E = 0
    if packing == 'simple':
        bits = ''.join(format(int(v), f'0{nbits}b') for v in x)
        bits += '0' * (-len(bits) % 8)
        data = int(bits, 2).to_bytes(len(bits) // 8, 'big') if bits else b''
        tmpl = 0
    else:
        from PIL import Image
        im = Image.frombytes('I;16', (x.size, 1), x.astype('<u2').tobytes())
        b = io.BytesIO(); im.save(b, format='JPEG2000', irreversible=False, no_jp2=True)
        data, tmpl = b.getvalue(), 40
    dsc = lambda v: (0x8000 | -v) if v < 0 else v
    s5 = sec(5, struct.pack('>IH', int(keep.sum()), tmpl) + struct.pack('>fHHBB', ref, dsc(E), dsc(D), nbits, 0) + b'\x00\x00')
    s6 = sec(6, b'\x00' + np.packbits(keep.astype(np.uint8)).tobytes()) if mask is not None else sec(6, b'\xff')
    s7 = sec(7, data)
    body = s1 + s3 + s4 + s5 + s6 + s7
    total = 16 + len(body) + 4
    return b'GRIB' + b'\x00\x00' + bytes([10, 2]) + struct.pack('>Q', total) + body + b'7777'


class Grib2(unittest.TestCase):
    def test_simple_packing_with_land(self):
        v = np.array([[1.25, 2.5, 0.0], [3.75, 0.0, 4.5]])
        land = np.array([[1, 1, 0], [1, 0, 1]])
        f = grib2.decode(message(v, 3, 2, mask=land))
        self.assertEqual(f.shape, (2, 3))
        self.assertAlmostEqual(f.values[0, 0], 1.25, places=4)
        self.assertAlmostEqual(f.values[1, 2], 4.5, places=4)
        self.assertTrue(np.isnan(f.values[0, 2]) and np.isnan(f.values[1, 1]))
        # the cell nearest a point, west longitudes as east
        self.assertAlmostEqual(f.at(39.6, -79.9), 3.75, places=4)    # (39.5, 280.0)
        self.assertTrue(np.isnan(f.at(39.6, -79.4)))   # (39.5, 280.5: land)
        self.assertTrue(np.isnan(f.at(10, -79)))

    def test_jpeg2000(self):
        v = np.array([[0.31, 1.07, 2.4, 0.0], [0.7, 4.15, 0.02, 1.3]])
        f = grib2.decode(message(v, 4, 2, packing='jpeg2000'))
        np.testing.assert_allclose(f.values, v, atol=1e-6)

    def test_south_first_rows_come_out_north_first(self):
        v = np.array([[1.0, 2.0], [3.0, 4.0]])   # (as stored: the southern row first)
        f = grib2.decode(message(v, 2, 2, lat0=40.0, north_first=False))
        self.assertEqual(f.lat0, 40.0)
        np.testing.assert_allclose(f.values, [[3.0, 4.0], [1.0, 2.0]])

    def test_narrow_pillows_widened_samples(self):
        x = np.array([0, 128, 13696, 52864])
        self.assertEqual(grib2.narrow(x, 'I;16', 9).tolist(), [0, 1, 107, 413])
        self.assertEqual(grib2.narrow(x, 'I;16', 16).tolist(), x.tolist())
        self.assertEqual(grib2.narrow(np.array([0, 64, 255]), 'L', 6).tolist(), [0, 16, 63])

    def test_rejects(self):
        with self.assertRaises(ValueError):
            grib2.decode(b'GRIB\x00\x00\x00\x01' + b'\x00' * 20)
        m = message([[1.0]], 1, 1)
        with self.assertRaises(ValueError):
            grib2.decode(m[:-30])


IDX = """1:0:d=2026111412:WIND:surface:1 hour fcst:
2:100:d=2026111412:WDIR:surface:1 hour fcst:
3:200:d=2026111412:UGRD:surface:1 hour fcst:
4:300:d=2026111412:VGRD:surface:1 hour fcst:
5:400:d=2026111412:HTSGW:surface:1 hour fcst:
6:500:d=2026111412:PERPW:surface:1 hour fcst:
7:600:d=2026111412:DIRPW:surface:1 hour fcst:
8:700:d=2026111412:WVHGT:surface:1 hour fcst:
9:800:d=2026111412:SWELL:1 in sequence:1 hour fcst:
10:900:d=2026111412:SWELL:2 in sequence:1 hour fcst:
11:1000:d=2026111412:SWELL:3 in sequence:1 hour fcst:
12:1100:d=2026111412:WVPER:surface:1 hour fcst:
"""


class Index(unittest.TestCase):
    def test_parse_and_merge(self):
        rows = grib2.parse_idx(IDX)
        self.assertEqual(rows[4], {'n': 5, 'start': 400, 'var': 'HTSGW', 'level': 'surface', 'fcst': '1 hour fcst', 'end': 500})
        self.assertIsNone(rows[-1]['end'])
        r = inlets.ranges_for(rows)
        # WIND+WDIR, then HTSGW..SWELL 2 (UGRD, VGRD and SWELL 3 left out), then WVPER to the end
        self.assertEqual([(a, b) for a, b, _ in r], [(0, 200), (400, 1000), (1100, None)])
        self.assertEqual([p[0] for p in r[1][2]], ['hs', 'tp', 'dp', 'wsh', 's1h', 's2h'])


class Cycle(unittest.TestCase):
    def test_newest_that_reaches_two_days(self):
        now = int(dt.datetime(2026, 11, 14, 21, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
        have = {'/12/'}
        get = lambda url, h=None: b'x' if any(k in url for k in have) else None
        cyc = inlets.pick_cycle(now, 48, get)
        self.assertEqual(dt.datetime.fromtimestamp(cyc / 1000, dt.timezone.utc).hour, 12)
        self.assertIn('gfs.20261114/12/wave/gridded/gfswave.t12z.atlocn.0p16.f058.grib2.idx', inlets.cycle_url(cyc, 58, '.idx'))
        self.assertIsNone(inlets.pick_cycle(now, 48, lambda u, h=None: None))


class Noaa(unittest.TestCase):
    def test_hilo(self):
        j = {'predictions': [{'t': '2026-11-14 12:13', 'v': '0.551', 'type': 'L'}, {'t': '2026-11-14 19:16', 'v': '2.912', 'type': 'H'}]}
        h = inlets.parse_hilo(j)
        self.assertEqual(h[0][1:], [55, 0]); self.assertEqual(h[1][1:], [291, 1])
        self.assertEqual(h[1][0] - h[0][0], (7 * 60 + 3) * 60000)
        self.assertEqual(inlets.parse_hilo({'error': {'message': 'No data'}}), [])

    def test_max_slack(self):
        j = {'current_predictions': {'cp': [
            {'Time': '2026-11-14 12:21', 'Velocity_Major': -1.01, 'meanFloodDir': 299, 'meanEbbDir': 118, 'Type': 'ebb'},
            {'Time': '2026-11-14 17:00', 'Velocity_Major': 0.0, 'meanFloodDir': 299, 'meanEbbDir': 118, 'Type': 'slack'},
            {'Time': '2026-11-14 19:40', 'Velocity_Major': 1.78, 'meanFloodDir': 299, 'meanEbbDir': 118, 'Type': 'flood'}]}}
        c = inlets.parse_max_slack(j)
        self.assertEqual((c['flood'], c['ebb']), (299.0, 118.0))
        self.assertEqual([e[1:] for e in c['ev']], [['e', -101], ['s', 0], ['f', 178]])

    def test_ndbc(self):
        txt = """#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS PTDY  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi  hPa    ft
2026 11 14 20 26  MM   MM   MM   0.7     8   5.1  67     MM    MM  16.0    MM   MM   MM    MM
2026 11 14 20 00 300 10.0 13.0    MM    MM    MM  MM 1011.8   9.0  16.0   2.3   MM +0.8    MM
2026 11 14 19 56 310  9.5 12.0   0.8     9   5.0  70     MM    MM  16.0    MM   MM   MM    MM
2026 11 12 19 56 310  9.5 12.0   1.8     9   5.0  70     MM    MM  16.0    MM   MM   MM    MM
"""
        now = int(dt.datetime(2026, 11, 14, 21, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
        o = inlets.parse_ndbc(txt, now)
        self.assertEqual(len(o), 2)   # (the rows without waves and the one two days old left out)
        self.assertEqual(o[0][1:], [80, 90, 70, 95, 310])
        self.assertEqual(o[1][1:], [70, 80, 67, None, None])
        self.assertLess(o[0][0], o[1][0])


class Build(unittest.TestCase):
    def test_whole_file_from_fake_servers(self):
        now = int(dt.datetime(2026, 11, 14, 21, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
        L = [{'id': 'a-inlet', 'offshore': [39.5, -79.5], 'tide': '1', 'current': 'C1', 'currentBin': 1, 'buoy': 'b1'},
             {'id': 'b-inlet', 'offshore': [39.5, -79.5], 'tide': '1', 'current': None, 'buoy': None},
             {'id': 'c-inlet', 'offshore': [40.0, -80.0], 'tide': '2', 'current': None, 'buoy': None}]
        # every field the same small grid: 1.0 (× the field's number) at (40, 280), 2.0 at (39.5, 280.5)
        names = [f[1:3] for f in inlets.FIELDS]
        msgs, idx, pos = [], [], 0
        for k, (var, lev) in enumerate(names):
            m = message([[1.0 + k, 0.5], [0.25, 2.0 + k]], 2, 2)
            idx.append(f'{k + 1}:{pos}:d=2026111412:{var}:{lev}:x fcst:')
            msgs.append(m); pos += len(m)
        blob, idx = b''.join(msgs), '\n'.join(idx) + '\n'
        seen = []

        def get(url, headers=None):
            seen.append(url)
            if url.endswith('.idx'):
                return idx.encode() if '/12/' in url else None
            if url.endswith('.grib2'):
                self.assertIn('Range', headers or {})
                a, b = headers['Range'][6:].split('-')
                return blob[int(a): (int(b) + 1) if b else None]
            if 'product=predictions' in url:
                return json.dumps({'predictions': [{'t': '2026-11-14 12:13', 'v': '0.5', 'type': 'L'}]}).encode()
            if 'product=currents_predictions' in url:
                return json.dumps({'current_predictions': {'cp': [{'Time': '2026-11-14 22:00', 'Velocity_Major': -2.0, 'meanFloodDir': 300, 'meanEbbDir': 120, 'Type': 'ebb'}]}}).encode()
            if 'realtime2' in url:
                return b'2026 11 14 20 26  MM   MM   MM   0.7     8   5.1  67     MM    MM  16.0    MM   MM   MM    MM\n'
            return None
        out = inlets.build(L, now, get=get, say=lambda *a: None)
        self.assertEqual(out['n'], 52)
        self.assertEqual(dt.datetime.fromtimestamp(out['t0'] / 1000, dt.timezone.utc).hour, 18)
        self.assertEqual(len(out['cells']), 2)   # (a and b share a cell)
        self.assertEqual(out['waves'], {'a-inlet': 0, 'b-inlet': 0, 'c-inlet': 1})
        hs = out['fields'].index('hs')
        self.assertEqual(out['cells'][0]['d'][hs][0], 200 + 100 * hs)   # (2.0 + k m, in cm)
        self.assertEqual(out['cells'][1]['d'][hs][-1], 100 * (1 + hs))
        self.assertEqual(out['scale']['tp'], 10)
        self.assertEqual(sorted(out['tides']), ['1', '2'])
        self.assertEqual(out['currents']['C1']['ev'][0][1:], ['e', -200])
        self.assertEqual(out['buoys']['b1']['obs'][0][1], 70)
        self.assertEqual(out['missing'], {})
        # (the model's file only ever read by byte range)
        grib_reads = [u for u in seen if u.endswith('.grib2')]
        self.assertTrue(grib_reads)
        json.dumps(out)   # (plain JSON)


if __name__ == '__main__':
    unittest.main()
