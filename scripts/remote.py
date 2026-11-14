"""Reading NOAA's model files where they sit (AWS Open Data), a few byte ranges at a time.

The forecast systems' files run from 12 MB to 700 MB an hour, and we need two surface fields
from each: so nothing is downloaded whole. `RangeFile` is a read-only, seekable file object over
HTTP Range requests (h5py opens NetCDF4/HDF5 files from it; cdf3.py reads NetCDF3 ones), with a
small block cache so the format's many small header reads cost a handful of requests.
"""
import io
import time
import urllib.error
import urllib.request

UA = {'User-Agent': 'currents-tiles/1.0'}


def _open(req, timeout, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416) or i == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
        time.sleep(1.5 * (i + 1))


def head(url, timeout=60):
    """Content length of a remote file, or None when it isn't there."""
    try:
        r = _open(urllib.request.Request(url, method='HEAD', headers=UA), timeout, tries=2)
        return int(r.headers['Content-Length'])
    except urllib.error.HTTPError:
        return None


class RangeFile(io.RawIOBase):
    """A remote file read by byte ranges. `fetched` counts the bytes actually transferred."""

    def __init__(self, url, block=1 << 18, size=None, max_blocks=64):
        super().__init__()
        self.url, self.block, self.pos, self.fetched = url, block, 0, 0
        self.size = size if size is not None else head(url)
        if self.size is None:
            raise FileNotFoundError(url)
        self._cache, self._max = {}, max_blocks

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _range(self, a, b):
        req = urllib.request.Request(self.url, headers={**UA, 'Range': f'bytes={a}-{b}'})
        data = _open(req, 180).read()
        self.fetched += len(data)
        return data

    def _blk(self, i):
        d = self._cache.pop(i, None)
        if d is None:
            a = i * self.block
            d = self._range(a, min(self.size, a + self.block) - 1)
        self._cache[i] = d  # (most recent last)
        while len(self._cache) > self._max:
            self._cache.pop(next(iter(self._cache)))
        return d

    def read_at(self, off, n):
        """n bytes at off: a large read goes straight through as one request, a small one via the cache."""
        if n <= 0:
            return b''
        if n >= 4 * self.block:
            return self._range(off, min(self.size, off + n) - 1)
        out, p = bytearray(), off
        while len(out) < n and p < self.size:
            i = p // self.block
            d = self._blk(i)
            o = p - i * self.block
            take = d[o:o + n - len(out)]
            if not take:
                break
            out += take
            p += len(take)
        return bytes(out)

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.read_at(self.pos, n)
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)
