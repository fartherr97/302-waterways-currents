"""Tests for the water temperature step: the tiles' coding, the PNG, the satellite's answer read
from NetCDF3, the tiles with water, NDBC's latest readings and names, and the whole step run
against fake servers.   python3 -m unittest discover -s scripts -p 'test_*.py'"""
import io
import json
import os
import re
import struct
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watertemp as wt  # noqa: E402

HOUR = 3600 * 1000
FILL = -7.768


def nc_bytes(lat, lon, sst, t_s=1795338000.0):
    """an ERDDAP-style .nc answer (NetCDF3 classic, as the server writes it)"""
    from scipy.io import netcdf_file
    b = io.BytesIO()
    f = netcdf_file(b, 'w', version=1)
    f.createDimension('time', 1); f.createDimension('latitude', len(lat)); f.createDimension('longitude', len(lon))
    v = f.createVariable('time', 'f8', ('time',)); v[:] = [t_s]; v.units = 'seconds since 1970-01-01T00:00:00Z'
    v = f.createVariable('latitude', 'f4', ('latitude',)); v[:] = lat
    v = f.createVariable('longitude', 'f4', ('longitude',)); v[:] = lon
    v = f.createVariable('analysed_sst', 'f8', ('time', 'latitude', 'longitude'))
    v._FillValue = FILL
    v[:] = np.where(np.isfinite(sst), sst, FILL)[None]
    f.flush()
    data = b.getvalue()
    f.close()
    return data


def png_rows(png, rows=None):
    """a grey + alpha PNG → its (h, w, 2) pixels (Paeth rows undone; the first `rows` of them)"""
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    p, idat, w, h = 8, b'', 0, 0
    while p < len(png):
        n, kind = struct.unpack('>I4s', png[p:p + 8])
        body = png[p + 8:p + 8 + n]
        if kind == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
            assert (depth, ctype) == (8, 4)
        elif kind == b'IDAT':
            idat += body
        p += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(-1, 1 + w * 2)
    assert raw.shape[0] == h
    h = min(h, rows or h)
    out = np.zeros((h, w, 2), np.int16)
    for y in range(h):
        assert raw[y, 0] == 4
        row = raw[y, 1:].reshape(w, 2).astype(np.int16)
        for x in range(w):
            a = out[y, x - 1] if x else np.zeros(2, np.int16)
            b = out[y - 1, x] if y else np.zeros(2, np.int16)
            c = out[y - 1, x - 1] if x and y else np.zeros(2, np.int16)
            pp = a + b - c
            pa, pb, pc = np.abs(pp - a), np.abs(pp - b), np.abs(pp - c)
            pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
            out[y, x] = (row[x] + pred) & 255
    return out


class Coding(unittest.TestCase):
    def test_round_trip_within_half_a_step(self):
        c = np.array([[-1.9, 0.0, 15.37], [28.4, np.nan, 36.0]])
        g, a = wt.encode(c)
        back = wt.decode(g, a)
        self.assertTrue(np.isnan(back[1, 1]))
        self.assertEqual(a[1, 1], 0)
        ok = np.isfinite(c)
        self.assertTrue(np.all(np.abs(back[ok] - c[ok]) <= wt.STEP / 2 + 1e-9))

    def test_out_of_range_is_clamped(self):
        g, _ = wt.encode(np.array([-10.0, 50.0]))
        self.assertEqual(g.tolist(), [0, 255])

    def test_png_holds_the_pixels(self):
        rng = np.random.default_rng(1)
        g = rng.integers(0, 256, (7, 9)).astype(np.uint8)
        a = np.where(rng.random((7, 9)) > 0.3, 255, 0).astype(np.uint8)
        px = png_rows(wt.png_la(g, a))
        self.assertEqual(px[:, :, 0].tolist(), g.tolist())
        self.assertEqual(px[:, :, 1].tolist(), a.tolist())


class Satellite(unittest.TestCase):
    def test_reads_the_answer_fill_as_no_water(self):
        lat = np.array([38.0, 38.01, 38.02]); lon = np.array([-75.0, -74.99])
        sst = np.array([[15.2, np.nan], [15.3, 15.4], [np.nan, 15.6]])
        la, lo, v, t = wt.read_grid(nc_bytes(lat, lon, sst))
        self.assertAlmostEqual(la[0], 38.0, places=4)
        self.assertTrue(np.isnan(v[0, 1]) and np.isnan(v[2, 0]))
        self.assertAlmostEqual(v[1, 1], 15.4)
        self.assertEqual(t, 1795338000000)

    def test_north_first_and_padded(self):
        v = np.array([[1.0], [2.0]])   # south row first
        out = wt.north_first(None, v, 3)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertEqual(out[1:, 0].tolist(), [2.0, 1.0])

    def test_tiles_with_water(self):
        lat = np.array([24.5, 38.6, 47.0]); lon = np.array([-97.0, -75.1])
        v = np.array([[np.nan, np.nan], [np.nan, 15.0], [9.0, np.nan]])
        self.assertEqual(wt.tiles_with_water(lat, lon, v), [(0, 6), (5, 3)])

    def test_tile_box(self):
        self.assertEqual(wt.tile_box(5, 3), (35.0, 38.99, -78.0, -74.01))


class Stations(unittest.TestCase):
    NOW = 1795470000000   # 2026-11-23 20:20 UTC

    TEXT = '''#STN       LAT      LON  YYYY MM DD hh mm WDIR WSPD   GST WVHT  DPD APD MWD   PRES  PTDY  ATMP  WTMP  DEWP  VIS   TIDE
#text      deg      deg   yr mo day hr mn degT  m/s   m/s   m   sec sec degT   hPa   hPa  degC  degC  degC  nmi     ft
LWSD1    38.783  -75.119 2026 11 23 19 00  MM    MM    MM   MM  MM   MM  MM     MM    MM    MM  13.5    MM   MM     MM
44084    38.537  -75.044 2026 11 23 19 46  MM    MM    MM  0.9   9  MM 120     MM    MM    MM  14.4    MM   MM     MM
OLD01    38.000  -75.000 2026 11 22 01 00  MM    MM    MM   MM  MM   MM  MM     MM    MM    MM  14.0    MM   MM     MM
FAR01    12.000   65.000 2026 11 23 19 00  MM    MM    MM   MM  MM   MM  MM     MM    MM    MM  28.3    MM   MM     MM
NOT01    39.000  -75.000 2026 11 23 19 00  MM    MM    MM   MM  MM   MM  MM     MM    MM    MM    MM    MM   MM     MM
'''

    def test_keeps_fresh_readings_in_the_area(self):
        got = wt.parse_latest(self.TEXT, self.NOW)
        self.assertEqual([g[0] for g in got], ['LWSD1', '44084'])
        self.assertEqual(got[0][1:4], (38.783, -75.119, 13.5))

    def test_names(self):
        xml = '''<?xml version="1.0"?><stations>
          <station id="lwsd1" lat="38.78" lon="-75.12" name="8557380 - Lewes, DE"/>
          <station id="44084" lat="38.5" lon="-75.0" name="Bethany Beach, DE (263)"/>
          <station id="44009" lat="38.4" lon="-74.7" name="DELAWARE BAY 26 NM SOUTHEAST OF CAPE MAY, NJ"/></stations>'''
        n = wt.parse_names(xml)
        self.assertEqual(n['LWSD1'], 'Lewes, DE')
        self.assertEqual(n['44084'], 'Bethany Beach, DE')
        self.assertEqual(n['44009'], 'Delaware Bay 26 NM Southeast Of Cape May, NJ')
        self.assertEqual(wt.parse_names('<not xml'), {})


class Whole(unittest.TestCase):
    """the step against fake servers: the overview, the tiles with water, the stations"""

    def test_writes_the_site(self):
        def sea(la, lo):   # water east of 75.5°W and south of 40°N, 12–20 °C by latitude
            return np.where((lo[None, :] > -75.5) & (la[:, None] < 40), 20 - (la[:, None] - 36) * 2, np.nan)

        def fake(url, timeout=120):
            if 'latest_obs' in url:
                return Stations.TEXT.encode()
            if 'activestations' in url:
                return b'<stations><station id="lwsd1" name="8557380 - Lewes, DE"/></stations>'
            q = urllib.parse.unquote(url.split('?', 1)[1])
            (a0, st, a1), (o0, _, o1) = re.findall(r'\(([-\d.]+)\):(\d+):\(([-\d.]+)\)', q)
            step = wt.RES * int(st)
            la = np.round(np.arange(float(a0), float(a1) + step / 2, step), 2)
            lo = np.round(np.arange(float(o0), float(o1) + step / 2, step), 2)
            return nc_bytes(la, lo, sea(la, lo))

        real, wt.fetch = wt.fetch, fake
        # the clock at the fixture's time: its readings are fresh then, and stay so whenever this runs
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch.object(wt.time, 'time', return_value=Stations.NOW / 1000):
                self.assertEqual(wt.main(['--out', d, '-q']), 0)
                m = json.load(open(os.path.join(d, 'watertemp', 'manifest.json')))
                # water only east of 75.5°W, south of 40°N: the squares x 5–7 (78–66°W), y 0–4 (23–43°N)
                self.assertEqual(m['tiles'], sorted(f't{x}_{y}' for x in (5, 6, 7) for y in range(5)))
                self.assertEqual((m['o']['w'], m['o']['h']), (641, 561))
                self.assertEqual(m['stations'], 2)
                png = open(os.path.join(d, 'watertemp', 't5_3.png'), 'rb').read()
                self.assertEqual(struct.unpack('>II', png[16:24]), (400, 400))
                px = png_rows(png, rows=1)
                # the top row is 38.99°N: 20 − 2.99 × 2 ≈ 14.0 °C, east of 75.5°W only
                row = wt.decode(px[0, :, 0], px[0, :, 1])
                col = round((-75.0 - -78.0) / wt.RES)
                self.assertAlmostEqual(row[col], 14.02, delta=wt.STEP)
                self.assertTrue(np.isnan(row[round((-76.0 - -78.0) / wt.RES)]))
                s = json.load(open(os.path.join(d, 'watertemp', 'stations.json')))
                self.assertEqual(s['rows'][1][:2], ['LWSD1', 'Lewes, DE'])
        finally:
            wt.fetch = real


if __name__ == '__main__':
    unittest.main()
