"""A minimal NetCDF3 (classic and 64-bit offset) reader over any object with read_at(off, n).

Some forecast systems (NYOFS, SJROFS) still write NetCDF3: one file holds all the forecast hours
as records. We parse the header and read only the slabs we need (the surface layer of u and v at
one record), never the whole file. Big-endian throughout, as the format says.
"""
import struct
import numpy as np

NC_DIMENSION, NC_VARIABLE, NC_ATTRIBUTE = 0x0A, 0x0B, 0x0C
TYPES = {1: ('i1', 1), 2: ('S1', 1), 3: ('>i2', 2), 4: ('>i4', 4), 5: ('>f4', 4), 6: ('>f8', 8)}


class Var:
    def __init__(self, name, dims, shape, attrs, nctype, vsize, begin, record):
        self.name, self.dims, self.shape, self.attrs = name, dims, shape, attrs
        self.dtype, self.itemsize = TYPES[nctype]
        self.vsize, self.begin, self.record = vsize, begin, record


class CDF3:
    def __init__(self, f):
        self.f = f
        head = f.read_at(0, 1 << 20)
        if head[:3] != b'CDF' or head[3] not in (1, 2):
            raise ValueError('not a NetCDF3 file')
        self.offset64 = head[3] == 2
        self._b, self._p = head, 4
        self.numrecs = self._u32()
        self.dims = []      # [(name, length)] (length 0: the record dimension)
        self.attrs = {}
        self.vars = {}
        self._dims()
        self.attrs = self._attrs()
        self._vars()
        recvars = [v for v in self.vars.values() if v.record]
        self.recsize = sum(v.vsize for v in recvars)
        if len(recvars) == 1:  # (one record variable: no padding between records)
            v = recvars[0]
            self.recsize = int(np.prod([s for s in v.shape[1:]] or [1])) * v.itemsize

    # ── header ──
    def _need(self, n):
        if self._p + n > len(self._b):
            self._b = self._b + self.f.read_at(len(self._b), max(n, 1 << 20))

    def _u32(self):
        self._need(4)
        v = struct.unpack('>I', self._b[self._p:self._p + 4])[0]
        self._p += 4
        return v

    def _u64(self):
        self._need(8)
        v = struct.unpack('>Q', self._b[self._p:self._p + 8])[0]
        self._p += 8
        return v

    def _name(self):
        n = self._u32()
        self._need(n + 4)
        s = self._b[self._p:self._p + n].decode('utf-8', 'replace')
        self._p += (n + 3) & ~3
        return s

    def _dims(self):
        tag, n = self._u32(), self._u32()
        if tag not in (0, NC_DIMENSION):
            raise ValueError('bad dimension list')
        for _ in range(n):
            self.dims.append((self._name(), self._u32()))

    def _attrs(self):
        tag, n = self._u32(), self._u32()
        out = {}
        if tag not in (0, NC_ATTRIBUTE):
            raise ValueError('bad attribute list')
        for _ in range(n):
            name = self._name()
            t, cnt = self._u32(), self._u32()
            dt, sz = TYPES[t]
            nbytes = cnt * sz
            self._need(nbytes + 4)
            raw = self._b[self._p:self._p + nbytes]
            self._p += (nbytes + 3) & ~3
            out[name] = raw.decode('utf-8', 'replace').rstrip('\x00') if t == 2 else np.frombuffer(raw, dt)
        return out

    def _vars(self):
        tag, n = self._u32(), self._u32()
        if tag not in (0, NC_VARIABLE):
            raise ValueError('bad variable list')
        for _ in range(n):
            name = self._name()
            nd = self._u32()
            ids = [self._u32() for _ in range(nd)]
            attrs = self._attrs()
            t, vsize = self._u32(), self._u32()
            begin = self._u64() if self.offset64 else self._u32()
            dims = [self.dims[i][0] for i in ids]
            shape = [self.dims[i][1] for i in ids]
            record = bool(ids) and self.dims[ids[0]][1] == 0
            if record:
                shape[0] = self.numrecs
            self.vars[name] = Var(name, dims, tuple(shape), attrs, t, vsize, begin, record)

    # ── data ──
    def read(self, name, index=()):
        """The variable, or with index = (i0, i1, ...) the slab var[i0, i1, ..., :, :] (leading
        indices only, the rest whole), read in one range."""
        v = self.vars[name]
        shape = list(v.shape)
        if v.record:
            if not index:
                return np.stack([self.read(name, (r,)) for r in range(self.numrecs)])
            rec, rest = index[0], index[1:]
            base = v.begin + rec * self.recsize
            shape = shape[1:]
        else:
            rest, base = index, v.begin
        inner = shape[len(rest):]
        stride = int(np.prod(inner or [1]))
        off = 0
        for k, i in enumerate(rest):
            off = off * shape[k] + i
        for k in range(len(rest), len(shape)):
            off *= shape[k]
        raw = self.f.read_at(base + off * v.itemsize, stride * v.itemsize)
        return np.frombuffer(raw, v.dtype).reshape(inner or ()).astype(v.dtype.replace('>', '='), copy=True)
