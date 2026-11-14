"""A small GRIB2 reader: one message at a time, as NOAA's wave model writes them.

Enough for the GFS-Wave (WAVEWATCH III) grids on AWS Open Data: a regular latitude/longitude
grid (grid template 3.0), a bitmap for land (section 6), and the values packed either simply
(data template 5.0) or as JPEG 2000 (5.40, what GFS-Wave uses), decoded by Pillow's OpenJPEG.
Nothing here downloads anything: inlets.py reads a message's bytes by range (the file's .idx
says where each field starts) and hands them here.

    f = decode(message_bytes)   # → Field: .values (rows north → south, NaN on land), .lat0, .lon0, .dlat, .dlon
    f.at(lat, lng)              # the nearest cell's value (NaN when it's land)
"""
import io
import struct

import numpy as np


def _s16(x):
    """GRIB2 writes signed integers as sign and magnitude"""
    return -(x & 0x7fff) if x & 0x8000 else x


def _s32(x):
    return -(x & 0x7fffffff) if x & 0x80000000 else x


class Field:
    def __init__(self, values, lat0, lon0, dlat, dlon, discipline, category, parameter):
        self.values = values            # (nj, ni) float32, row 0 = lat0 (the northern edge), NaN = no value
        self.lat0, self.lon0 = lat0, lon0
        self.dlat, self.dlon = dlat, dlon
        self.discipline, self.category, self.parameter = discipline, category, parameter

    @property
    def shape(self):
        return self.values.shape

    def index(self, lat, lng):
        """the (row, column) of the cell nearest a point, or None outside the grid"""
        lon = lng % 360
        j = int(round((self.lat0 - lat) / self.dlat))
        i = int(round((lon - self.lon0) / self.dlon))
        nj, ni = self.values.shape
        return (j, i) if 0 <= j < nj and 0 <= i < ni else None

    def center(self, j, i):
        lon = self.lon0 + i * self.dlon
        return self.lat0 - j * self.dlat, (lon + 180) % 360 - 180

    def at(self, lat, lng):
        ji = self.index(lat, lng)
        return float('nan') if ji is None else float(self.values[ji])


def _sections(buf):
    b = memoryview(buf)
    if bytes(b[0:4]) != b'GRIB' or b[7] != 2:
        raise ValueError('not a GRIB2 message')
    total = struct.unpack('>Q', b[8:16])[0]
    if total > len(b):
        raise ValueError(f'truncated GRIB2 message ({len(b)} of {total} bytes)')
    out, p = {}, 16
    while p < total - 4:
        if bytes(b[p:p + 4]) == b'7777':
            break
        n = struct.unpack('>I', b[p:p + 4])[0]
        out[b[p + 4]] = bytes(b[p:p + n])
        p += n
    return b[6], out


def _unpack_simple(data, nbits, n):
    if nbits == 0:
        return np.zeros(n, np.float64)
    bits = np.unpackbits(np.frombuffer(data, np.uint8))[: n * nbits].reshape(n, nbits)
    w = (1 << np.arange(nbits - 1, -1, -1, dtype=np.int64))
    return bits.astype(np.int64) @ w


def narrow(x, mode, prec):
    """Pillow widens a JPEG 2000 component to its mode's bits (8 for L, 16 for I;16), shifting
    it up: back to the codestream's own precision (its SIZ marker's Ssiz + 1)"""
    mode_bits = 8 if mode in ('L', 'P') else 16 if mode.startswith('I;16') else 0
    return x >> (mode_bits - prec) if mode_bits and prec < mode_bits else x


def _unpack_jpeg2000(data, nbits, n):
    from PIL import Image   # (Pillow's wheels carry OpenJPEG)
    if nbits == 0 or not data:
        return np.zeros(n, np.float64)
    im = Image.open(io.BytesIO(data))
    siz = data.find(b'\xff\x51')
    prec = (data[siz + 40] & 0x7f) + 1 if siz >= 0 else nbits
    x = narrow(np.asarray(im).astype(np.int64).ravel(), im.mode, prec)
    if x.size < n:
        raise ValueError(f'JPEG 2000 field has {x.size} values, expected {n}')
    return x[:n]


def decode(buf):
    discipline, s = _sections(buf)
    s3, s4, s5, s7 = s.get(3), s.get(4), s.get(5), s.get(7)
    if not (s3 and s4 and s5 and s7):
        raise ValueError('incomplete GRIB2 message')
    tmpl3 = struct.unpack('>H', s3[12:14])[0]
    if tmpl3 != 0:
        raise ValueError(f'grid template 3.{tmpl3} not supported (only latitude/longitude)')
    npts = struct.unpack('>I', s3[6:10])[0]
    t = 14
    ni, nj = struct.unpack('>II', s3[t + 16:t + 24])
    la1, lo1 = _s32(struct.unpack('>I', s3[t + 32:t + 36])[0]), struct.unpack('>I', s3[t + 36:t + 40])[0]
    la2, lo2 = _s32(struct.unpack('>I', s3[t + 41:t + 45])[0]), struct.unpack('>I', s3[t + 45:t + 49])[0]
    di, dj = struct.unpack('>II', s3[t + 49:t + 57])
    scan = s3[t + 57]
    if scan & 0x80 or scan & 0x20:
        raise ValueError(f'scanning mode {scan:#x} not supported')
    category, parameter = s4[9], s4[10]
    nvals = struct.unpack('>I', s5[5:9])[0]
    tmpl5 = struct.unpack('>H', s5[9:11])[0]
    ref = struct.unpack('>f', s5[11:15])[0]
    E = _s16(struct.unpack('>H', s5[15:17])[0])
    D = _s16(struct.unpack('>H', s5[17:19])[0])
    nbits = s5[19]
    data = s7[5:]
    if tmpl5 == 0:
        x = _unpack_simple(data, nbits, nvals)
    elif tmpl5 == 40:
        x = _unpack_jpeg2000(data, nbits, nvals)
    else:
        raise ValueError(f'data template 5.{tmpl5} not supported')
    vals = (ref + x * 2.0 ** E) / 10.0 ** D
    s6 = s.get(6)
    full = np.full(npts, np.nan, np.float32)
    if s6 is not None and s6[5] == 0:
        mask = np.unpackbits(np.frombuffer(s6[6:], np.uint8))[:npts].astype(bool)
        if mask.sum() != nvals:
            raise ValueError('bitmap and values disagree')
        full[mask] = vals
    else:
        full[:] = vals[:npts]
    g = full.reshape(nj, ni)
    # scanning: +i west → east; j north → south unless bit 2 (0x40) says south → north
    north_first = not (scan & 0x40)
    if not north_first:
        g = g[::-1]
    lat_n = max(la1, la2) / 1e6
    return Field(g, lat_n, lo1 / 1e6, dj / 1e6, di / 1e6, discipline, category, parameter)


def parse_idx(text):
    """a .idx → [{ n, start, end (exclusive, None for the last), var, level, fcst }]"""
    rows = []
    for line in text.strip().splitlines():
        p = line.split(':')
        if len(p) < 6:
            continue
        rows.append({'n': int(p[0]), 'start': int(p[1]), 'var': p[3], 'level': p[4], 'fcst': p[5]})
    for a, b in zip(rows, rows[1:]):
        a['end'] = b['start']
    if rows:
        rows[-1]['end'] = None
    return rows
