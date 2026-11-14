"""The currents as the app reads them: the same tiles as the wind (RGBA PNG; R, G = east, north as
byte = m/s × SCALE + OFFSET; A = 255 where a model has water), in three levels so a view of a
canal and a view of the coast each get a sensible amount of data:
  f  0.005° cells (about 500 m), 1° tiles, only where some model is finer than about 1.6 km
  m  0.02° cells (about 2 km), 4° tiles, everywhere a model covers
  o  one 0.05° overview of the whole coverage, for wide views
B holds the fade in from a model's open boundary (0 there → 255 inside), so the app fades the
field out where the model stops at sea; at the shore it stops at the land mask.
Where models overlap, the more local one is laid over the other through its own fade.
"""
import json
import struct
import zlib
import numpy as np

from mesh import Mapping

SCALE, OFFSET = 40.0, 128.0          # 1/40 m/s steps (0.05 kn), ±3.2 m/s (6.2 kn)
WEST, SOUTH, EAST, NORTH = -98.0, 24.0, -66.0, 50.0   # Texas to Maine, the Gulf and the Great Lakes
LEVELS = [
    dict(id='f', res=0.005, tile=1.0, edge=1.5, fine_m=1600.0),
    dict(id='m', res=0.02, tile=4.0, edge=1.2),
]
OV = dict(id='o', res=0.05, edge=1.0)
for L in LEVELS:
    L['px'] = int(round(L['tile'] / L['res']))
    L['cols'] = int(np.ceil((EAST - WEST) / L['tile']))
    L['rows'] = int(np.ceil((NORTH - SOUTH) / L['tile']))
OV['w'] = int(round((EAST - WEST) / OV['res']))
OV['h'] = int(round((NORTH - SOUTH) / OV['res']))


def tile_box(L, tx, ty):
    west = WEST + tx * L['tile']
    south = SOUTH + ty * L['tile']
    return west, south, west + L['tile'], south + L['tile']


def tiles_over(L, bbox):
    w, s, e, n = bbox
    tx0 = max(0, int(np.floor((w - WEST) / L['tile']))); tx1 = min(L['cols'] - 1, int(np.floor((e - WEST) / L['tile'])))
    ty0 = max(0, int(np.floor((s - SOUTH) / L['tile']))); ty1 = min(L['rows'] - 1, int(np.floor((n - SOUTH) / L['tile'])))
    return [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]


def encode_byte(ms):
    return np.clip(np.round(np.asarray(ms) * SCALE + OFFSET), 0, 255).astype(np.uint8)


def decode_byte(b):
    return (np.asarray(b, float) - OFFSET) / SCALE


class Grid:
    """One output image (a tile or the overview) and every model's mapping onto it."""

    def __init__(self, key, west, north, res, w, h):
        self.key, self.west, self.north, self.res, self.w, self.h = key, west, north, res, w, h
        self.maps = []   # [(prio, model id, Mapping)]

    def add(self, prio, mid, mesh):
        mp = Mapping(mesh, self.west, self.north, self.res, self.w, self.h)
        if mp.cells.size:
            self.maps.append((prio, mid, mp))
            self.maps.sort(key=lambda x: x[0])
        return mp

    def composite(self, fields):
        """fields: model id → (u_node, v_node) for this hour (models without the hour are left out).
        → RGBA bytes (h, w, 4) and whether any cell has water."""
        U = np.zeros((self.h, self.w)); V = np.zeros((self.h, self.w)); F = np.zeros((self.h, self.w))
        valid = np.zeros((self.h, self.w), bool)
        for prio, mid, mp in self.maps:
            if mid not in fields:
                continue
            u, v, f, land = mp.sample(*fields[mid])
            has = np.isfinite(u)
            # a more local model's land hides a coarser one's water there
            valid &= ~land
            f = np.where(has, f, 0)
            a = np.where(valid, f, 1.0)   # (over nothing, a model shows at full strength, faded by B)
            U = np.where(has, U * (1 - a) + np.nan_to_num(u) * a, U)
            V = np.where(has, V * (1 - a) + np.nan_to_num(v) * a, V)
            F = np.where(has, np.where(valid, 1 - (1 - F) * (1 - f), f), F)
            valid |= has
        rgba = np.zeros((self.h, self.w, 4), np.uint8)
        rgba[..., 0] = np.where(valid, encode_byte(U), int(OFFSET))
        rgba[..., 1] = np.where(valid, encode_byte(V), int(OFFSET))
        rgba[..., 2] = np.where(valid, np.round(np.clip(F, 0, 1) * 255), 0).astype(np.uint8)
        rgba[..., 3] = np.where(valid, 255, 0)
        return rgba, bool(valid.any())


# ── PNG (RGBA, 8-bit), Paeth-filtered rows: smooth fields compress well ──
def _chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def encode_png(rgba):
    h, w, _ = rgba.shape
    px = rgba.astype(np.int16)
    a = np.zeros_like(px); a[:, 1:] = px[:, :-1]           # left
    b = np.zeros_like(px); b[1:] = px[:-1]                 # up
    c = np.zeros_like(px); c[1:, 1:] = px[:-1, :-1]        # up-left
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    filt = ((px - pred) & 255).astype(np.uint8).reshape(h, w * 4)
    raw = np.concatenate([np.full((h, 1), 4, np.uint8), filt], 1).tobytes()
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + _chunk(b'IHDR', ihdr) + _chunk(b'IDAT', zlib.compress(raw, 9)) + _chunk(b'IEND', b'')


def manifest(hours, models, updated):
    """What the app reads first (currents/manifest.json). The top-level res/tile/west/… are the fine
    level's (the wind's manifest shape); `levels` and `ov` say the rest."""
    f = LEVELS[0]
    return {
        'kind': 'currents', 'updated': updated, 'scale': SCALE, 'offset': OFFSET, 'fade': 'tile',
        'res': f['res'], 'tile': f['tile'], 'west': WEST, 'south': SOUTH, 'cols': f['cols'], 'rows': f['rows'],
        'ovRes': OV['res'],
        'levels': [{k: L[k] for k in ('id', 'res', 'tile', 'cols', 'rows', 'edge')} for L in LEVELS],
        'ov': {'res': OV['res'], 'w': OV['w'], 'h': OV['h'], 'edge': OV['edge']},
        'models': models,
        'hours': hours,
    }


def dumps(obj):
    return json.dumps(obj, separators=(',', ':'))
