"""NOAA's Operational Forecast Systems: where each one's files are and how to read its surface
current. All on AWS Open Data (bucket noaa-nos-ofs-pds, public, no account needed).

Each model gives: mesh() → mesh.Mesh (once per run) and hours() → [(valid_ms, reader)] where
reader() → (u_east, v_north) at the mesh's nodes in m/s (the top layer: the current a boat feels).
Only the bytes of those two surface fields are fetched (remote.RangeFile).
"""
import datetime as dt
import re
import numpy as np

from remote import RangeFile, head
from cdf3 import CDF3
from mesh import Mesh, structured_triangles, roms_to_rho, to_earth, boundary_nodes

BUCKET = 'https://noaa-nos-ofs-pds.s3.amazonaws.com'
HOUR = 3600_000

# id: name (as the app says it), kind, the cycles it runs (UTC), product, forecast hours it has,
# priority where models overlap (higher wins: the finer, more local model), the open-boundary fade
MODELS = {
    'cbofs':  dict(name='Chesapeake Bay', kind='roms', cycles=(0, 6, 12, 18), product='fields', prio=2, fade_km=25),
    'dbofs':  dict(name='Delaware Bay', kind='roms', cycles=(0, 6, 12, 18), product='fields', prio=2, fade_km=25),
    'tbofs':  dict(name='Tampa Bay', kind='roms', cycles=(0, 6, 12, 18), product='fields', prio=3, fade_km=15),
    'gomofs': dict(name='Gulf of Maine', kind='roms', cycles=(0, 6, 12, 18), product='2ds', prio=1, fade_km=25, u='u_sur', v='v_sur'),
    'ngofs2': dict(name='Northern Gulf of Mexico', kind='fvcom', cycles=(3, 9, 15, 21), product='2ds', prio=1, fade_km=25, u='u_surface', v='v_surface', open_depth=15),
    'nyofs':  dict(name='New York Harbor', kind='pom', cycles=(5, 11, 17, 23), prio=3, fade_km=5),
    'nyofs_fg': dict(name='New York Harbor', kind='pom', dir='nyofs', cycles=(5, 11, 17, 23), prio=4, fade_km=2),
    'sjrofs': dict(name='St. Johns River', kind='pom', cycles=(5, 11, 17, 23), prio=3, fade_km=4),
    'leofs':  dict(name='Lake Erie', kind='fvcom', cycles=(0, 6, 12, 18), product='fields', prio=2, fade_km=0),
    'lmhofs': dict(name='Lakes Michigan and Huron', kind='fvcom', cycles=(0, 6, 12, 18), product='fields', prio=1, fade_km=0),
    'lsofs':  dict(name='Lake Superior', kind='fvcom', cycles=(0, 6, 12, 18), product='fields', prio=1, fade_km=0),
    'loofs':  dict(name='Lake Ontario', kind='fvcom', cycles=(0, 6, 12, 18), product='fields', prio=2, fade_km=0),
}
DEFAULT = ['cbofs', 'dbofs', 'nyofs', 'nyofs_fg', 'gomofs', 'tbofs', 'ngofs2', 'sjrofs', 'leofs', 'lmhofs', 'lsofs', 'loofs']


def file_url(mid, cycle_ms, tag, product=None):
    m = MODELS[mid]
    d = dt.datetime.fromtimestamp(cycle_ms / 1000, dt.timezone.utc)
    ymd = d.strftime('%Y%m%d')
    product = product or m.get('product', 'fields')
    return f"{BUCKET}/{m.get('dir', mid)}/netcdf/{d:%Y}/{d:%m}/{d:%d}/{mid}.t{d:%H}z.{ymd}.{product}.{tag}.nc"


def latest_cycle(mid, now_ms, last_hour=48, back_hours=36):
    """The newest cycle whose forecast is complete (its last file is up), or None."""
    m = MODELS[mid]
    t = (now_ms // HOUR) * HOUR
    for _ in range(back_hours):
        if dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).hour in m['cycles']:
            probe = file_url(mid, t, 'forecast') if m['kind'] == 'pom' else file_url(mid, t, f'f{last_hour:03d}')
            if head(probe):
                return t
        t -= HOUR
    return None


def _time_units(units):
    """'seconds since 2016-01-01 00:00:00' → (unit ms, epoch ms)."""
    u = units.decode() if isinstance(units, bytes) else str(units)
    mm = re.match(r'\s*(\w+)\s+since\s+(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?', u)
    if not mm:
        raise ValueError(f'time units {u!r}')
    unit = {'seconds': 1000, 'second': 1000, 'minutes': 60_000, 'hours': HOUR, 'days': 86_400_000}[mm.group(1).lower()]
    y, mo, d = int(mm.group(2)), int(mm.group(3)), int(mm.group(4))
    hh, mi, ss = int(mm.group(5) or 0), int(mm.group(6) or 0), int(mm.group(7) or 0)
    epoch = dt.datetime(y, mo, d, hh, mi, ss, tzinfo=dt.timezone.utc).timestamp() * 1000
    return unit, epoch


def slab(d, f, lead):
    """d[lead…] (the leading indices given, the rest whole) of an HDF5 dataset read straight from
    the file's bytes, for an uncompressed one: HDF5 itself reads whole chunks (a 20-layer chunk for
    one layer is 10× the bytes). Falls back to h5py for anything else."""
    try:
        if d.compression or d.id.get_create_plist().get_nfilters():
            return d[lead]
        shape, n = d.shape, len(lead)
        rest = shape[n:]
        dt = d.dtype
        if d.chunks is None:   # contiguous
            off = d.id.get_offset()
            if off is None:
                return d[lead]
            k = 0
            for i, x in enumerate(lead):
                k = k * shape[i] + x
            size = int(np.prod(rest))
            return np.frombuffer(f.read_at(off + k * size * dt.itemsize, size * dt.itemsize), dt).reshape(rest).astype(dt.newbyteorder('='))
        ch = d.chunks
        if any(c != 1 for c in ch[:n - 1]) or len(rest) not in (1, 2):
            return d[lead]
        out = np.empty(rest, dt.newbyteorder('='))
        got = np.zeros(rest, bool)
        cz = ch[n - 1]
        inner = ch[n:]
        for i in range(d.id.get_num_chunks()):
            info = d.id.get_chunk_info(i)
            o = info.chunk_offset
            if any(o[j] != lead[j] for j in range(n - 1)) or not (o[n - 1] <= lead[n - 1] < o[n - 1] + cz):
                continue
            per = int(np.prod(inner))
            start = info.byte_offset + (lead[n - 1] - o[n - 1]) * per * dt.itemsize
            block = np.frombuffer(f.read_at(start, per * dt.itemsize), dt).reshape(inner)
            sl = tuple(slice(o[n + j], min(o[n + j] + inner[j], rest[j])) for j in range(len(rest)))
            out[sl] = block[tuple(slice(0, s.stop - s.start) for s in sl)]
            got[sl] = True
        return out if got.all() else d[lead]
    except Exception:
        return d[lead]


def to_ms(values, units):
    unit, epoch = _time_units(units)
    return np.round((np.asarray(values, float) * unit + epoch) / 60_000) * 60_000


class Model:
    def __init__(self, mid, now_ms, hours=48, verbose=False):
        self.id, self.cfg = mid, MODELS[mid]
        self.hours_ahead, self.verbose = hours, verbose
        self.cycle = latest_cycle(mid, now_ms, last_hour=hours if self.cfg['kind'] != 'pom' else 48)
        self.fetched = 0
        self._mesh = None

    def log(self, *a):
        if self.verbose:
            print(f'[{self.id}]', *a, flush=True)

    # ── per-hour files (ROMS, FVCOM): n006 is the cycle's own hour, f001… the hours after ──
    def _hour_files(self):
        out = [(self.cycle, 'n006' if self.cfg['kind'] != 'fvcom' or self.cfg.get('product') != '2ds' else 'n006')]
        out += [(self.cycle + k * HOUR, f'f{k:03d}') for k in range(1, self.hours_ahead + 1)]
        return out

    def _h5(self, url):
        import h5py
        f = RangeFile(url)
        return f, h5py.File(f, 'r')

    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        k = self.cfg['kind']
        if k == 'pom':
            self._mesh = self._pom_mesh()
        else:
            f, h = self._h5(file_url(self.id, self.cycle, 'n006'))
            try:
                self._mesh = self._roms_mesh(h) if k == 'roms' else self._fvcom_mesh(h)
            finally:
                self.fetched += f.fetched; h.close()
        return self._mesh

    def _roms_mesh(self, h):
        lon, lat = h['lon_rho'][:], h['lat_rho'][:]
        mask = h['mask_rho'][:]
        self._angle = h['angle'][:]
        self._mask_u, self._mask_v = h['mask_u'][:], h['mask_v'][:]
        ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) > 0.1)
        tri = structured_triangles(ok)
        ny, nx = lon.shape
        # water that moves: a cell with at least one open face (a lone wet cell in a marsh has none,
        # and its current is always zero)
        mu, mv = self._mask_u > 0.5, self._mask_v > 0.5
        face = np.zeros((ny, nx), bool)
        face[:, :-1] |= mu; face[:, 1:] |= mu; face[:-1, :] |= mv; face[1:, :] |= mv
        mask = ((mask > 0.5) & face).astype(float)
        # open boundary: the grid's outer ring, where it's water
        ring = np.zeros((ny, nx), bool); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        open_b = np.nonzero((ring & (mask > 0.5)).ravel())[0]
        self.log('ROMS grid', lon.shape, 'water', int((mask > 0.5).sum()))
        return Mesh(lon, lat, tri, wet=mask, open_b=open_b, fade_km=self.cfg['fade_km'], prune=True)

    def _fvcom_mesh(self, h):
        lon, lat = h['lon'][:], h['lat'][:]
        tri = (h['nv'][:].T - 1).astype(np.int64)
        depth = h['h'][:]
        b = boundary_nodes(tri)
        od = self.cfg.get('open_depth')
        open_b = b[depth[b] >= od] if od else np.array([], np.int64)
        self._tri = tri
        self._counts = np.bincount(tri.ravel(), minlength=lon.size).astype(float)
        self.log('FVCOM mesh nodes', lon.size, 'elements', len(tri), 'open boundary nodes', len(open_b))
        return Mesh(lon, lat, tri, open_b=open_b, fade_km=self.cfg['fade_km'])

    def _pom_files(self):
        """POM/EFDC: one file for the forecast, its records hourly (and the nowcast's last records)."""
        url = file_url(self.id, self.cycle, 'forecast')
        f = RangeFile(url)
        c = CDF3(f)
        return f, c

    def _pom_mesh(self):
        f, c = self._pom_files()
        lon, lat, mask = c.read('lon'), c.read('lat'), c.read('mask')
        ok = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lat) > 0.1) & (np.abs(lon) > 0.1)
        tri = structured_triangles(ok)
        ny, nx = lon.shape
        ring = np.zeros((ny, nx), bool); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        open_b = np.nonzero((ring & (mask > 0.5) & ok).ravel())[0]
        self._cdf, self._cdf_f = c, f
        times = to_ms(c.read('time'), c.vars['time'].attrs['units'])
        self._pom_times = times
        self.log('POM/EFDC grid', lon.shape, 'records', c.numrecs, 'water', int((mask > 0.5).sum()))
        return Mesh(lon, lat, tri, wet=mask, open_b=open_b, fade_km=self.cfg['fade_km'], prune=True)

    def hours(self):
        """[(valid ms, reader)] for the run: reader() → (u_east, v_north) at the nodes."""
        if self.cycle is None:
            return []
        m = self.mesh()
        k = self.cfg['kind']
        if k == 'pom':
            c = self._cdf
            end = self.cycle + self.hours_ahead * HOUR
            out = []
            ts = np.asarray(self._pom_times, float)
            th = np.ceil((ts.min() - 300_000) / HOUR) * HOUR
            while th <= min(end, ts.max() + 300_000):
                # the record at the hour (float32 days: a few seconds off), else the two around it
                # (SJROFS writes at half past), blended
                near = np.nonzero(np.abs(ts - th) < 300_000)[0]
                if near.size:
                    out.append((int(th), (lambda r=int(near[0]): self._pom_read(r))))
                else:
                    a = np.nonzero(ts < th)[0]; b = np.nonzero(ts > th)[0]
                    if a.size and b.size and ts[b[0]] - ts[a[-1]] <= 2 * HOUR:
                        ra, rb = int(a[-1]), int(b[0]); w = (th - ts[ra]) / (ts[rb] - ts[ra])
                        out.append((int(th), (lambda ra=ra, rb=rb, w=w: self._pom_blend(ra, rb, w))))
                th += HOUR
            return [x for x in out if x[0] >= self.cycle - 1]
        return [(t, (lambda tag=tag, t=t: self._read(tag, t))) for t, tag in self._hour_files()]

    def _pom_blend(self, ra, rb, w):
        ua, va = self._pom_read(ra); ub, vb = self._pom_read(rb)
        return ua * (1 - w) + ub * w, va * (1 - w) + vb * w

    def _pom_read(self, r):
        c = self._cdf
        before = self._cdf_f.fetched
        u = c.read('u', (r, 0)).astype(float); v = c.read('v', (r, 0)).astype(float)
        self.fetched += self._cdf_f.fetched - before
        bad = ~np.isfinite(u) | ~np.isfinite(v) | (np.abs(u) > 50) | (np.abs(v) > 50)
        u[bad] = 0; v[bad] = 0
        return u.ravel(), v.ravel()

    def _read(self, tag, t):
        f, h = self._h5(file_url(self.id, self.cycle, tag))
        try:
            if self.cfg['kind'] == 'roms':
                if 'u' in self.cfg:   # (GoMOFS's 2ds files: the surface layer alone)
                    u, v = slab(h[self.cfg['u']], f, (0,)), slab(h[self.cfg['v']], f, (0,))
                else:
                    uu, vv = h['u'], h['v']
                    u, v = slab(uu, f, (0, uu.shape[1] - 1)), slab(vv, f, (0, vv.shape[1] - 1))   # (ROMS: the last s level is the surface)
                ur, vr = roms_to_rho(np.asarray(u, float), np.asarray(v, float), self._mask_u, self._mask_v)
                ue, vn = to_earth(ur, vr, self._angle)
                return ue.ravel(), vn.ravel()
            # FVCOM: the first sigma layer is the surface; elements → nodes
            if 'u' in self.cfg:
                ue, vn = slab(h[self.cfg['u']], f, (0,)), slab(h[self.cfg['v']], f, (0,))
            else:
                ue, vn = slab(h['u'], f, (0, 0)), slab(h['v'], f, (0, 0))
            ue = np.where(np.abs(ue) < 50, ue, 0).astype(float); vn = np.where(np.abs(vn) < 50, vn, 0).astype(float)
            tri = self._tri
            un = np.bincount(tri.ravel(), weights=np.repeat(ue, 3), minlength=self._counts.size) / np.maximum(self._counts, 1)
            vnn = np.bincount(tri.ravel(), weights=np.repeat(vn, 3), minlength=self._counts.size) / np.maximum(self._counts, 1)
            return un, vnn
        finally:
            self.fetched += f.fetched
            h.close()
