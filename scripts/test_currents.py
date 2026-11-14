"""Tests for the currents tiles: python3 -m unittest discover -s scripts -p 'test_*.py'
Rotation, the ROMS face → centre averaging, the triangle mesh and its interpolation, land and the
open-boundary fade, overlapping models, the NetCDF3 reader, the PNG, the station index, times."""
import io
import os
import sys
import tempfile
import unittest
import zlib
import struct

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh import to_earth, roms_to_rho, structured_triangles, boundary_nodes, Mesh, Mapping  # noqa: E402
from cdf3 import CDF3  # noqa: E402
from tiles import Grid, encode_png, encode_byte, decode_byte, SCALE, OFFSET  # noqa: E402
from models import to_ms, file_url, MODELS  # noqa: E402
import stations  # noqa: E402


class Rotation(unittest.TestCase):
    def test_no_angle_is_unchanged(self):
        self.assertEqual(to_earth(1.0, 2.0, 0.0), (1.0, 2.0))

    def test_grid_turned_90_degrees(self):
        # the grid's xi axis points north (angle = 90° from east): along-xi flow is northward
        e, n = to_earth(1.0, 0.0, np.pi / 2)
        self.assertAlmostEqual(e, 0.0); self.assertAlmostEqual(n, 1.0)
        e, n = to_earth(0.0, 1.0, np.pi / 2)   # along eta (west of xi): westward
        self.assertAlmostEqual(e, -1.0); self.assertAlmostEqual(n, 0.0)

    def test_speed_is_kept(self):
        u, v = np.random.default_rng(1).normal(size=(2, 50))
        a = np.random.default_rng(2).uniform(-np.pi, np.pi, 50)
        e, n = to_earth(u, v, a)
        np.testing.assert_allclose(np.hypot(e, n), np.hypot(u, v))


class RomsFaces(unittest.TestCase):
    def test_faces_to_centres(self):
        u = np.array([[1.0, 3.0]])          # (eta 1, xi-1 2) → 3 centres
        v = np.zeros((0, 3))
        ur, _ = roms_to_rho(u, v)
        self.assertEqual(ur.shape, (1, 3))
        np.testing.assert_allclose(ur[0], [1.0, 2.0, 3.0])

    def test_land_faces_are_left_out(self):
        u = np.array([[1.0, 0.0]]); mu = np.array([[1.0, 0.0]])
        ur, _ = roms_to_rho(u, np.zeros((0, 3)), mu, np.zeros((0, 3)))
        np.testing.assert_allclose(ur[0], [1.0, 1.0, 0.0])

    def test_fill_values_are_zero(self):
        ur, vr = roms_to_rho(np.array([[1e37, 2.0]]), np.zeros((0, 3)))
        self.assertTrue(np.all(np.abs(ur) < 10))


def plane_mesh(n=6, wet=None):
    lon, lat = np.meshgrid(np.linspace(-76, -75, n), np.linspace(37, 38, n))
    tri = structured_triangles(np.ones(lon.shape, bool))
    return lon, lat, Mesh(lon, lat, tri, wet=wet)


class MeshInterpolation(unittest.TestCase):
    def test_triangles(self):
        tri = structured_triangles(np.ones((3, 4), bool))
        self.assertEqual(len(tri), 2 * 2 * 3)
        ok = np.ones((3, 4), bool); ok[1, 1] = False   # a node with no position takes its quads with it
        self.assertEqual(len(structured_triangles(ok)), 2 * (6 - 4))

    def test_linear_field_is_exact(self):
        lon, lat, m = plane_mesh()
        u = (lon * 2 + lat).ravel(); v = (lat - lon).ravel()
        mp = Mapping(m, -76, 38, 0.05, 20, 20)
        U, V, F, L = mp.sample(u, v)
        glon = -76 + (np.arange(20) + 0.5) * 0.05; glat = 38 - (np.arange(20) + 0.5) * 0.05
        GL, GA = np.meshgrid(glon, glat)
        np.testing.assert_allclose(U, GL * 2 + GA, atol=1e-9)
        np.testing.assert_allclose(V, GA - GL, atol=1e-9)
        self.assertFalse(L.any())

    def test_outside_is_nan(self):
        lon, lat, m = plane_mesh()
        U, V, F, L = Mapping(m, -77, 38, 0.1, 30, 5).sample(np.ones(lon.size), np.ones(lon.size))
        self.assertTrue(np.isnan(U[:, :5]).all())     # west of -76: no model
        self.assertFalse(np.isnan(U[:, 15]).any())

    def test_land_is_masked_not_blended(self):
        wet = np.ones((6, 6)); wet[:, 3:] = 0      # the east half is land
        lon, lat, m = plane_mesh(wet=wet)
        u = np.where(wet.ravel() > 0, 1.0, 99.0)   # (land holds junk: it must never show)
        U, V, F, L = Mapping(m, -76, 38, 0.02, 50, 50).sample(u, np.zeros(lon.size))
        water = np.isfinite(U)
        self.assertTrue(water[:, :20].all()); self.assertFalse(water[:, 40:].any())
        np.testing.assert_allclose(U[water], 1.0)
        self.assertTrue(L[:, 40:].all())

    def test_open_boundary_fade(self):
        lon, lat = np.meshgrid(np.linspace(-76, -75, 11), np.linspace(37, 38, 11))
        tri = structured_triangles(np.ones(lon.shape, bool))
        ring = np.zeros(lon.shape, bool); ring[:, -1] = True   # open sea along the east side
        m = Mesh(lon, lat, tri, open_b=np.nonzero(ring.ravel())[0], fade_km=20)
        U, V, F, L = Mapping(m, -76, 38, 0.05, 20, 20).sample(np.ones(lon.size), np.zeros(lon.size))
        self.assertAlmostEqual(F[10, 0], 1.0)             # far inside
        self.assertLess(F[10, -1], 0.2)                   # by the open boundary
        self.assertTrue(np.all(np.diff(F[10, 12:]) <= 1e-9))

    def test_land_cells_with_stray_positions_are_dropped(self):
        # a curvilinear grid's land cells can sit anywhere (the model never uses them): one flung
        # across the map must not paint water over land
        lon, lat = np.meshgrid(np.linspace(-76, -75, 6), np.linspace(37, 38, 6))
        wet = np.ones((6, 6)); wet[:, 4:] = 0
        lon = lon.copy(); lon[2, 4] = -70.0                # a land node by the water, flung far east
        tri = structured_triangles(np.ones(lon.shape, bool))
        m = Mesh(lon, lat, tri, wet=wet, prune=True)
        self.assertLess(m.bbox[2], -74.9)
        U, *_ = Mapping(m, -76, 38, 0.05, 140, 20).sample(np.ones(lon.size), np.zeros(lon.size))
        self.assertFalse(np.isfinite(U[:, 40:]).any())     # nothing east of -74
        self.assertTrue(np.isfinite(U[:, :10]).all())

    def test_boundary_nodes(self):
        tri = structured_triangles(np.ones((3, 3), bool))
        self.assertEqual(sorted(boundary_nodes(tri).tolist()), [0, 1, 2, 3, 5, 6, 7, 8])

    def test_folded_grid_takes_the_wettest(self):
        # two meshes laid over the same place, one land and one water: water shows
        lon, lat = np.meshgrid(np.linspace(-76, -75, 4), np.linspace(37, 38, 4))
        t = structured_triangles(np.ones(lon.shape, bool))
        L2 = np.concatenate([lon.ravel(), lon.ravel()]); A2 = np.concatenate([lat.ravel(), lat.ravel()])
        tri = np.concatenate([t, t + lon.size]); wet = np.concatenate([np.zeros(lon.size), np.ones(lon.size)])
        m = Mesh(L2, A2, tri, wet=wet)
        U, *_ = Mapping(m, -76, 38, 0.1, 10, 10).sample(np.ones(L2.size), np.zeros(L2.size))
        self.assertTrue(np.isfinite(U).all())


class Overlap(unittest.TestCase):
    def test_local_model_over_a_coarse_one_through_its_fade(self):
        lon, lat = np.meshgrid(np.linspace(-76, -75, 11), np.linspace(37, 38, 11))
        tri = structured_triangles(np.ones(lon.shape, bool))
        coarse = Mesh(lon, lat, tri)
        ring = np.zeros(lon.shape, bool); ring[:, 0] = True
        fine = Mesh(lon, lat, tri, open_b=np.nonzero(ring.ravel())[0], fade_km=30)
        g = Grid('t', -76, 38, 0.05, 20, 20)
        g.add(1, 'c', coarse); g.add(3, 'f', fine)
        n = lon.size
        rgba, any_ = g.composite({'c': (np.full(n, 1.0), np.zeros(n)), 'f': (np.full(n, -1.0), np.zeros(n))})
        u = decode_byte(rgba[..., 0])
        self.assertTrue(any_)
        self.assertGreater(u[10, 1], 0.3)      # by the fine model's open edge: mostly the coarse one
        self.assertAlmostEqual(u[10, 19], -1.0, places=1)   # inside it: the fine one
        self.assertTrue((rgba[..., 3] == 255).all())
        # a model without this hour is left out
        rgba2, _ = g.composite({'c': (np.full(n, 1.0), np.zeros(n))})
        np.testing.assert_allclose(decode_byte(rgba2[..., 0]), 1.0)


class Png(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, (7, 9, 4), dtype=np.uint8)
        png = encode_png(a)
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        w, h = struct.unpack('>II', png[16:24]); self.assertEqual((w, h), (9, 7))
        i = png.index(b'IDAT'); n = struct.unpack('>I', png[i - 4:i])[0]
        raw = np.frombuffer(zlib.decompress(png[i + 4:i + 4 + n]), np.uint8).reshape(7, 9 * 4 + 1)
        out = np.zeros((7, 36), np.int16)
        for y in range(7):   # undo Paeth
            f = raw[y, 1:].astype(np.int16)
            for x in range(36):
                a_ = out[y, x - 4] if x >= 4 else 0; b_ = out[y - 1, x] if y else 0; c_ = out[y - 1, x - 4] if x >= 4 and y else 0
                p = a_ + b_ - c_; pa, pb, pc = abs(p - a_), abs(p - b_), abs(p - c_)
                pred = a_ if pa <= pb and pa <= pc else b_ if pb <= pc else c_
                out[y, x] = (f[x] + pred) & 255
        np.testing.assert_array_equal(out.reshape(7, 9, 4), a)

    def test_bytes(self):
        self.assertEqual(encode_byte(0).item(), 128)
        self.assertEqual(encode_byte(1.0).item(), 128 + SCALE)
        self.assertEqual(encode_byte(10).item(), 255)
        self.assertAlmostEqual(decode_byte(encode_byte(-0.75)).item(), -0.75)


class NetCDF3(unittest.TestCase):
    def test_reads_a_record_slab(self):
        from scipy.io import netcdf_file
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.nc')
            f = netcdf_file(p, 'w')
            f.createDimension('time', None); f.createDimension('sigma', 3); f.createDimension('ny', 2); f.createDimension('nx', 4)
            t = f.createVariable('time', 'f4', ('time',)); t.units = 'days since 2008-01-01  0:00:00 00:00'
            lon = f.createVariable('lon', 'f4', ('ny', 'nx')); lon[:] = np.arange(8).reshape(2, 4)
            u = f.createVariable('u', 'f4', ('time', 'sigma', 'ny', 'nx'))
            v = f.createVariable('v', 'f4', ('time', 'sigma', 'ny', 'nx'))
            for r in range(3):
                t[r] = 6900 + r / 24
                u[r] = np.arange(24).reshape(3, 2, 4) + 100 * r
                v[r] = -np.arange(24).reshape(3, 2, 4) - 100 * r
            f.close()
            with open(p, 'rb') as fh:
                data = fh.read()

        class Local:
            def read_at(self, off, n): return data[off:off + n]
        c = CDF3(Local())
        self.assertEqual(c.numrecs, 3)
        np.testing.assert_array_equal(c.read('lon'), np.arange(8).reshape(2, 4))
        np.testing.assert_array_equal(c.read('u', (2, 0)), np.arange(8).reshape(2, 4) + 200)
        np.testing.assert_array_equal(c.read('v', (1, 2)), -(np.arange(16, 24).reshape(2, 4)) - 100)
        ms = to_ms(c.read('time'), c.vars['time'].attrs['units'])
        self.assertEqual(int(ms[1] - ms[0]), 3600_000)


class Times(unittest.TestCase):
    def test_units(self):
        self.assertEqual(int(to_ms([0], 'seconds since 2016-01-01 00:00:00')[0]), 1451606400000)
        self.assertEqual(int(to_ms([1], 'days since 2008-01-01  0:00:00 00:00')[0]), 1199232000000)

    def test_file_names(self):
        t = 1763121600000  # 2025-11-14 12Z
        self.assertTrue(file_url('cbofs', t, 'f001').endswith('/cbofs/netcdf/2025/11/14/cbofs.t12z.20251114.fields.f001.nc'))
        self.assertTrue(file_url('nyofs_fg', t, 'forecast').endswith('/nyofs/netcdf/2025/11/14/nyofs_fg.t12z.20251114.fields.forecast.nc'))
        self.assertTrue(file_url('ngofs2', t, 'n006').endswith('ngofs2.t12z.20251114.2ds.n006.nc'))
        self.assertIn('gomofs', MODELS)


class Stations(unittest.TestCase):
    def test_one_bin_each_nearest_the_surface_in_our_box(self):
        meta = [
            {'id': 'NYH1924', 'name': 'Hell Gate', 'lat': 40.7783, 'lng': -73.9383, 'currbin': 1, 'type': 'H', 'depth': 32.0},
            {'id': 'NYH1924', 'name': 'Hell Gate', 'lat': 40.7783, 'lng': -73.9383, 'currbin': 9, 'type': 'H', 'depth': 6.0},
            {'id': 'ACT1556', 'name': 'Sagamore Beach', 'lat': 41.8, 'lng': -70.5, 'currbin': 1, 'type': 'W', 'depth': None},
            {'id': 'SFB1201', 'name': 'Golden Gate', 'lat': 37.8, 'lng': -122.5, 'currbin': 1, 'type': 'H', 'depth': 10.0},
            {'id': 'ACT6241', 'name': 'Chesapeake City Bridge', 'lat': 39.52783, 'lng': -75.80717, 'currbin': 1, 'type': 'S', 'depth': None},
        ]
        out = stations.pick(meta)
        self.assertEqual(out['fields'], ['id', 'name', 'lat', 'lng', 'bin', 'type', 'depth'])
        self.assertEqual([r[0] for r in out['rows']], ['ACT6241', 'NYH1924'])
        self.assertEqual(out['rows'][1][4], 9)


if __name__ == '__main__':
    unittest.main()
