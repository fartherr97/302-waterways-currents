"""Model grids → our plain latitude/longitude grid.

Every forecast system is turned into one shape: a triangle mesh of nodes, each with a position,
whether it's water, and (per hour) the current at it as east/north components in m/s.
  - ROMS (CBOFS, DBOFS, TBOFS, GoMOFS): a curvilinear grid; u and v sit on the cell faces and run
    along the grid's own axes, so they're averaged to the cell centres and rotated by `angle`
    (the angle between the grid's xi axis and east). Each grid quad becomes two triangles.
  - POM / EFDC (NYOFS, SJROFS): curvilinear too, with east/north velocities at the nodes.
  - FVCOM (NGOFS2, the Great Lakes): an unstructured triangle mesh; u and v are at the triangles'
    centres, averaged to the nodes.
Then every cell centre of the output grid is placed in the triangle holding it (once per model,
reused for every hour) and the current interpolated with the triangle's barycentric weights.
Cells in no triangle are outside the model; cells whose triangle is mostly land are land.
The open boundary (where a model meets the sea it doesn't model) gets a fade: 0 there, rising to 1
`fade_km` inside, so the field never stops at a hard line in open water.
"""
import numpy as np
from scipy.spatial import cKDTree

M_PER_DEG = 111_320.0
WATER = 0.4   # a cell is water when its triangle's wet nodes weigh this much (0.5 is the midline)


def to_earth(u_grid, v_grid, angle):
    """Grid-relative (along xi, eta) → east, north. ROMS: angle = angle between xi and east (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return u_grid * c - v_grid * s, u_grid * s + v_grid * c


def roms_to_rho(u, v, mask_u=None, mask_v=None):
    """ROMS u (eta, xi-1) and v (eta-1, xi) on the faces → both at the cell centres (eta, xi),
    averaging only the faces that are water (land faces hold zeros or fill)."""
    u = np.where(np.isfinite(u) & (np.abs(u) < 100), u, 0.0)
    v = np.where(np.isfinite(v) & (np.abs(v) < 100), v, 0.0)
    mu = np.ones_like(u) if mask_u is None else (mask_u > 0.5).astype(float)
    mv = np.ones_like(v) if mask_v is None else (mask_v > 0.5).astype(float)
    ny, nx1 = u.shape
    nx = nx1 + 1
    us, uw = np.zeros((ny, nx)), np.zeros((ny, nx))
    us[:, :-1] += u * mu; uw[:, :-1] += mu
    us[:, 1:] += u * mu; uw[:, 1:] += mu
    vs, vw = np.zeros((ny, nx)), np.zeros((ny, nx))
    vs[:-1, :] += v * mv; vw[:-1, :] += mv
    vs[1:, :] += v * mv; vw[1:, :] += mv
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(uw > 0, us / np.maximum(uw, 1e-9), 0.0), np.where(vw > 0, vs / np.maximum(vw, 1e-9), 0.0)


def structured_triangles(valid):
    """A (ny, nx) grid's quads as two triangles each (node index = j·nx + i), skipping quads with a
    corner that has no position."""
    ny, nx = valid.shape
    j, i = np.mgrid[0:ny - 1, 0:nx - 1]
    n0 = (j * nx + i).ravel()
    n1, n2 = n0 + 1, n0 + nx
    n3 = n2 + 1
    v = valid.ravel()
    ok = v[n0] & v[n1] & v[n2] & v[n3]
    a = np.stack([n0, n1, n3], 1)[ok]
    b = np.stack([n0, n3, n2], 1)[ok]
    return np.concatenate([a, b]).astype(np.int32)


def boundary_nodes(tri):
    """Nodes on the mesh's outer edge (edges used by one triangle only)."""
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(1)
    key = e[:, 0].astype(np.int64) * (1 << 31) + e[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    edge = uniq[cnt == 1]
    return np.unique(np.concatenate([edge >> 31, edge & ((1 << 31) - 1)])).astype(np.int64)


class Mesh:
    """lon, lat, wet: per node; tri: (T, 3) node indices; open_b: indices of open-boundary nodes."""

    def __init__(self, lon, lat, tri, wet=None, open_b=None, fade_km=10.0, prune=False):
        self.lon = np.asarray(lon, float).ravel()
        self.lon = np.where(self.lon > 180, self.lon - 360, self.lon)
        self.lat = np.asarray(lat, float).ravel()
        self.tri = np.asarray(tri, np.int64)
        self.wet = np.ones(self.lon.size) if wet is None else (np.asarray(wet, float).ravel() > 0.5).astype(float)
        used = np.zeros(self.lon.size, bool); used[self.tri.ravel()] = True
        self.lat0 = float(np.mean(self.lat[used])) if used.any() else 0.0
        self.kx = M_PER_DEG * np.cos(np.radians(self.lat0))
        self.x, self.y = self.lon * self.kx, self.lat * M_PER_DEG
        P = np.stack([self.x, self.y], 1)
        if prune and self.tri.size:
            # A curvilinear grid's land cells are never used by the model, so their positions may be
            # anything (folded behind the shore, stretched across a peninsula). Keep the triangles
            # with two or more wet corners, and only where the land corner is no further off than
            # the wet edge is long: the coast falls between the water and the land beside it.
            w3 = self.wet[self.tri] > 0.5
            t = self.tri
            e = np.stack([np.hypot(*(P[t[:, 0]] - P[t[:, 1]]).T), np.hypot(*(P[t[:, 1]] - P[t[:, 2]]).T), np.hypot(*(P[t[:, 2]] - P[t[:, 0]]).T)], 1)
            pair = np.stack([w3[:, 0] & w3[:, 1], w3[:, 1] & w3[:, 2], w3[:, 2] & w3[:, 0]], 1)
            wet_edge = np.where(pair, e, 0).max(1)
            self.tri = t[(w3.sum(1) >= 2) & (e.max(1) <= 2.5 * wet_edge)]
        A, B, C = P[self.tri[:, 0]], P[self.tri[:, 1]], P[self.tri[:, 2]]
        self.A, self.v0, self.v1 = A, C - A, B - A
        self.d00 = (self.v0 ** 2).sum(1); self.d01 = (self.v0 * self.v1).sum(1); self.d11 = (self.v1 ** 2).sum(1)
        self.den = self.d00 * self.d11 - self.d01 ** 2
        area = 0.5 * np.abs(self.den) ** 0.5
        self.size = np.sqrt(2 * area)                         # a triangle's size, metres
        # each triangle's box in degrees, to find the ones over a grid
        tl = self.lon[self.tri]; ta = self.lat[self.tri]
        self.tbox = np.stack([tl.min(1), ta.min(1), tl.max(1), ta.max(1)], 1)
        # (the box of the water: a curvilinear grid's land corners don't count)
        wet_nodes = np.zeros(self.lon.size, bool); wet_nodes[self.tri[self.wet[self.tri].max(1) > 0.5].ravel()] = True
        lo, la = self.lon[wet_nodes], self.lat[wet_nodes]
        used = wet_nodes
        self.bbox = (float(lo.min()), float(la.min()), float(lo.max()), float(la.max())) if used.any() else (0, 0, 0, 0)
        # the fade in from the open boundary
        self.node_fade = np.ones(self.lon.size)
        if open_b is not None and len(open_b) and fade_km > 0:
            ob = np.stack([self.x[open_b], self.y[open_b]], 1)
            d, _ = cKDTree(ob).query(np.stack([self.x, self.y], 1), k=1)
            self.node_fade = np.clip(d / (fade_km * 1000.0), 0, 1)

    def rasterize(self, west, north, res, w, h, max_pairs=4_000_000):
        """Every cell centre of a grid (rows north → south) that lies in a triangle: the flat cell
        indices, their triangles and barycentric weights (n, 3). Each triangle is laid on the cells
        under its box (exact, whatever the triangles' shapes)."""
        b = self.tbox
        south, east = north - h * res, west + w * res
        sel = np.nonzero((b[:, 2] >= west) & (b[:, 0] <= east) & (b[:, 3] >= south) & (b[:, 1] <= north))[0]
        cells, tris, ws = [], [], []
        if not sel.size:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3))
        # cell index range under each triangle's box (centres at west + (c + 0.5)·res)
        c0 = np.maximum(0, np.ceil((b[sel, 0] - west) / res - 0.5)).astype(np.int64)
        c1 = np.minimum(w - 1, np.floor((b[sel, 2] - west) / res - 0.5)).astype(np.int64)
        r0 = np.maximum(0, np.ceil((north - b[sel, 3]) / res - 0.5)).astype(np.int64)
        r1 = np.minimum(h - 1, np.floor((north - b[sel, 1]) / res - 0.5)).astype(np.int64)
        nc, nr = c1 - c0 + 1, r1 - r0 + 1
        keep = (nc > 0) & (nr > 0)
        sel, c0, r0, nc, nr = sel[keep], c0[keep], r0[keep], nc[keep], nr[keep]
        cnt = nc * nr
        # in batches, so a big grid under big triangles never makes a huge array
        cum = np.cumsum(cnt)
        start = 0
        while start < len(sel):
            base = cum[start - 1] if start else 0
            stop = int(np.searchsorted(cum, base + max_pairs, side='right'))
            stop = max(stop, start + 1)
            s_, c0_, r0_, nc_, cnt_ = sel[start:stop], c0[start:stop], r0[start:stop], nc[start:stop], cnt[start:stop]
            rep = np.repeat(np.arange(len(s_)), cnt_)
            off = np.arange(rep.size) - np.repeat(np.cumsum(cnt_) - cnt_, cnt_)
            col = c0_[rep] + off % nc_[rep]
            row = r0_[rep] + off // nc_[rep]
            t = s_[rep]
            px = (west + (col + 0.5) * res) * self.kx
            py = (north - (row + 0.5) * res) * M_PER_DEG
            v2x, v2y = px - self.A[t, 0], py - self.A[t, 1]
            d20 = v2x * self.v0[t, 0] + v2y * self.v0[t, 1]
            d21 = v2x * self.v1[t, 0] + v2y * self.v1[t, 1]
            den = self.den[t]
            with np.errstate(invalid='ignore', divide='ignore'):
                wc = (self.d11[t] * d20 - self.d01[t] * d21) / den   # weight of C
                wb = (self.d00[t] * d21 - self.d01[t] * d20) / den   # weight of B
            wa = 1 - wc - wb
            ins = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9) & (np.abs(den) > 0)
            cells.append((row * w + col)[ins]); tris.append(t[ins])
            ws.append(np.stack([wa[ins], wb[ins], wc[ins]], 1))
            start = stop
        cells = np.concatenate(cells); tris = np.concatenate(tris); ws = np.concatenate(ws)
        # a centre on a shared edge, or under two triangles where a curvilinear grid folds over
        # itself behind the land: once, from the wettest
        wetw = (ws * self.wet[self.tri[tris]]).sum(1)
        order = np.lexsort((-wetw, cells))
        cells, tris, ws = cells[order], tris[order], ws[order]
        cells, first = np.unique(cells, return_index=True)
        return cells, tris[first], ws[first]


class Mapping:
    """Where a grid's cells fall in a mesh: computed once, used for every hour."""

    def __init__(self, mesh, west, north, res, w, h):
        self.mesh = mesh
        self.shape = (h, w)
        cells, t, wts = mesh.rasterize(west, north, res, w, h)
        self.cells = cells.astype(np.int32)           # flat indices of cells inside the mesh
        self.nodes = mesh.tri[t].astype(np.int32)     # (m, 3)
        self.w = wts.astype(np.float32)
        wet = mesh.wet[self.nodes]
        self.wsum = (self.w * wet).sum(1)
        self.water = self.wsum >= WATER
        self.fade = (self.w * mesh.node_fade[self.nodes]).sum(1)

    def sample(self, u_node, v_node):
        """u, v (east, north m/s) on the grid; NaN outside the model; and a land mask (inside, not water)."""
        wet = self.mesh.wet[self.nodes]
        ww = self.w * wet
        den = np.maximum(self.wsum, 1e-9)
        uu = (ww * np.nan_to_num(u_node)[self.nodes]).sum(1) / den
        vv = (ww * np.nan_to_num(v_node)[self.nodes]).sum(1) / den
        U = np.full(int(np.prod(self.shape)), np.nan); V = U.copy()
        F = np.zeros(U.size); L = np.zeros(U.size, bool)
        c = self.cells
        U[c] = np.where(self.water, uu, np.nan); V[c] = np.where(self.water, vv, np.nan)
        F[c] = np.where(self.water, self.fade, 0); L[c] = ~self.water
        return U.reshape(self.shape), V.reshape(self.shape), F.reshape(self.shape), L.reshape(self.shape)


def grid_centres(west, north, res, w, h):
    """Cell-centre lon/lat of a grid whose rows run north → south."""
    lon = west + (np.arange(w) + 0.5) * res
    lat = north - (np.arange(h) + 0.5) * res
    return np.meshgrid(lon, lat)
