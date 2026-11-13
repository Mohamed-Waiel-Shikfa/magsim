"""Eddy-current loop network in a conducting magnet (task 4).

The magnet is cut into `n_z` axial slabs and, in each slab, `n_r` nested loops ("bands") around the
axis that the coil field threads. Each band is a closed current path with an assumed uniform current
distribution over its cross-section; the network is linear (R, L, M) and goes into SPICE as floating
R-L loops coupled with `K` elements (`magsim.circuit.netlist`).

* cylinder / ring: annular bands r in [r1, r2], z in [z1, z2]; azimuthal current with uniform
  density, R_k = rho 2 pi r_mid / (dr dz) (exact for that distribution).
* cuboid: bands between the similar rectangles of scale s1, s2 (s in [0, 1], s = 1 the outer
  surface), stream function psi = max(2|x|/A, 2|y|/B). With uniform current per unit psi the density
  is uniform within each of the four trapezoids and continuous across the diagonals, and the
  dissipated power gives R_k = 4 rho s_mid (A/B + B/A) / (ds dz) (the old
  `magnet-flip-sim/src/emag.py:307-345` value, here derived from P = int rho J^2 dV). This assumes
  the eddy currents follow similar rectangles (exact only for a square in the thin-band limit).

Inductances use the filament machinery of `magsim.coil` (Maxwell / Neumann closed forms; each band is
split into near-square sub-filaments with Grover's rectangle GMD as self term [grover1946]).
`field_per_amp` integrates exact uniformly-polarised cylinder / cuboid fields (magpylib, a side-wall
current sheet = a uniformly polarised body [ortner2020]) over the band thickness with Gauss-Legendre
quadrature split where the observer lies on a sheet's plane, so points inside a band are handled exactly up
to quadrature error. Validated in `tests/test_eddy.py` against the exact Bessel solution for a long
cylinder in a uniform AC axial field (gate C4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from magpylib.func import cuboid_field, cylinder_field
from numpy.polynomial.legendre import leggauss
from scipy.spatial.transform import Rotation

from magsim.coil import (
    FilamentSet,
    _as_rotation,
    _polygon_self,
    gmd_rectangle,
    maxwell_mutual,
    outline_polygon,
    outline_segments,
    segment_mutual,
)
from magsim.constants import MU0

_CHUNK = 200_000

__all__ = ["EddyLoops", "eddy_loops", "harmonic_currents"]


def _parse_pose(pose):
    """pose: None | (position, rotation) | object with .position and .rotation."""
    if pose is None:
        return np.zeros(3), Rotation.identity()
    if hasattr(pose, "position") and hasattr(pose, "rotation"):
        return np.asarray(pose.position, float).reshape(3), _as_rotation(pose.rotation)
    pos, rot = pose
    return np.asarray(pos, float).reshape(3), _as_rotation(rot)


def _local_to_body(axis: int) -> np.ndarray:
    """Rotation matrix whose columns are the body axes used as local x, y, z (cyclic: right-handed)."""
    e = np.eye(3)
    return np.column_stack([e[(axis + 1) % 3], e[(axis + 2) % 3], e[axis]])


@dataclass
class EddyLoops:
    """Linear eddy-loop network of one magnet.

    R : (n,) loop resistances [ohm]; M : (n,n) loop inductance matrix [H] (positive definite).
    Loop k = iz * n_r + ir (slab iz from local -z, band ir from the axis outwards); positive current
    circulates counter-clockwise about the local +z axis (global direction `axis_global`).
    """

    shape: str
    kind: str  # "annular" | "rect"
    n_r: int
    n_z: int
    resistivity: float
    R: np.ndarray
    M: np.ndarray
    lo: np.ndarray  # (n,) inner radius [m] (annular) or inner scale (rect)
    hi: np.ndarray  # (n,) outer radius / scale
    z1: np.ndarray  # (n,) local axial range [m]
    z2: np.ndarray
    in_plane: tuple  # (A, B) for rect [m]; () for annular
    position: np.ndarray
    rot_total: Rotation  # local -> global
    _fil: FilamentSet = field(repr=False, default=None)
    _area: np.ndarray = field(repr=False, default=None)

    @property
    def n(self) -> int:
        return len(self.R)

    @property
    def axis_global(self) -> np.ndarray:
        return self.rot_total.apply([0.0, 0.0, 1.0])

    def filaments(self, owner_offset=0) -> FilamentSet:
        """Global-frame sub-filaments; owner = loop index (+ owner_offset)."""
        fs = self._fil
        return FilamentSet(fs.loops, fs.seg_gmd, fs.weight, fs.owner + owner_offset, fs.normal,
                           fs.circle_center, fs.circle_radius)

    def mutual_with(self, coil) -> np.ndarray:
        """(n,) mutual inductance [H] between each loop and a coil (unit coil current)."""
        return coil.mutual(self)

    def full_inductance(self, coil) -> np.ndarray:
        """(n+1, n+1) inductance matrix [H]: index 0 = coil, 1..n = loops."""
        m = self.mutual_with(coil)
        L = np.empty((self.n + 1, self.n + 1))
        L[0, 0] = coil.inductance()
        L[0, 1:] = L[1:, 0] = m
        L[1:, 1:] = self.M
        return L

    def area_vectors(self) -> np.ndarray:
        """(n,3) current-weighted mean loop area vector [m^2]: flux per loop in a uniform field B is B . a_k."""
        return self._area[:, None] * self.axis_global[None, :]

    # ---- field -------------------------------------------------------------------------------------
    def field_per_amp(self, points, n_q=4) -> np.ndarray:
        """H [A/m per A] of each loop at global `points` (m,3) -> (m, n, 3), global frame.

        Band current = integral over the band thickness of uniform current sheets (side walls of
        uniformly polarised cylinders / cuboids, magpylib exact fields [ortner2020]); Gauss-Legendre
        with `n_q` nodes per piece, split at the observer's own radius (annular) or at the two scales whose
        side faces pass through it (rect), where the integrand is not smooth.
        """
        pts = self.rot_total.inv().apply(np.atleast_2d(np.asarray(points, float)) - self.position)
        n_pts = len(pts)
        x, w = leggauss(n_q)
        out = np.zeros((n_pts, self.n, 3))
        per_pt = self.n * 3 * n_q
        step = max(1, _CHUNK // per_pt)
        for k0 in range(0, n_pts, step):
            p = pts[k0 : k0 + step]
            m = len(p)
            if self.kind == "annular":
                own1 = own2 = np.hypot(p[:, 0], p[:, 1])
            else:
                A, B = self.in_plane
                sx, sy = 2 * np.abs(p[:, 0]) / A, 2 * np.abs(p[:, 1]) / B
                own1, own2 = np.minimum(sx, sy), np.maximum(sx, sy)  # face planes through the point
            lo, hi = self.lo[None, :], self.hi[None, :]
            c1 = np.clip(own1[:, None], lo, hi)  # split points (m, n)
            c2 = np.clip(own2[:, None], lo, hi)
            nodes, wts = [], []
            for a, b in ((lo, c1), (c1, c2), (c2, hi)):
                length = b - a
                nd = a[..., None] + 0.5 * (x + 1.0) * length[..., None]  # (m, n, n_q)
                wt = 0.5 * w * length[..., None] / (hi - lo)[..., None]
                mid = 0.5 * (lo + hi)[..., None]
                nd = np.where(length[..., None] > 0, nd, np.broadcast_to(mid, nd.shape))
                nodes.append(nd)
                wts.append(np.broadcast_to(wt, nd.shape))
            nodes = np.concatenate(nodes, axis=2)  # (m, n, 3 n_q)
            wts = np.concatenate(wts, axis=2)
            dz = (self.z2 - self.z1)[None, :, None]
            zc = (0.5 * (self.z1 + self.z2))[None, :, None]
            shp = nodes.shape
            obs = np.broadcast_to(p[:, None, None, :], shp + (3,)).reshape(-1, 3)
            pos = np.zeros(shp + (3,))
            pos[..., 2] = zc
            pol = np.zeros(shp + (3,))
            pol[..., 2] = MU0 / dz  # sheet current 1 A spread over dz -> M = 1/dz, J = MU0 M
            if self.kind == "annular":
                dims = np.stack([2 * nodes, np.broadcast_to(dz, shp)], axis=-1)
                B = cylinder_field("B", obs, dims.reshape(-1, 2), pol.reshape(-1, 3), positions=pos.reshape(-1, 3),
                                   squeeze=False)
            else:
                A, Bd = self.in_plane
                dims = np.stack([A * nodes, Bd * nodes, np.broadcast_to(dz, shp)], axis=-1)
                B = cuboid_field("B", obs, dims.reshape(-1, 3), pol.reshape(-1, 3), positions=pos.reshape(-1, 3),
                                 squeeze=False)
            H = B.reshape(shp + (3,)) / MU0
            out[k0 : k0 + m] = np.einsum("pnq,pnqk->pnk", wts, H)
        return self.rot_total.apply(out.reshape(-1, 3)).reshape(out.shape)


def harmonic_currents(R, M, coupling, omega, source_current=1.0):
    """Complex loop currents [A] of `(R + j omega M) i = -j omega coupling * source_current`.

    coupling : (n,) mutual inductance [H] (or flux per source amp [Wb/A]) of each loop with the source.
    """
    Z = np.diag(np.asarray(R, float)) + 1j * omega * np.asarray(M, float)
    return np.linalg.solve(Z, -1j * omega * np.asarray(coupling, float) * source_current)


def _auto_sub(radial, axial, cap=64):
    """Near-square sub-cells for a band of radial thickness `radial` and height `axial`."""
    if radial >= axial:
        return min(cap, max(1, int(round(radial / axial)))), 1
    return 1, min(cap, max(1, int(round(axial / radial))))


def _band_matrix(table, n_r, n_sr, n_z, n_sz):
    """Band inductance matrix [H] from the sub-filament table T[i, j, delta].

    T[i, j, delta] = mutual of sub-filament outlines i, j (radial index, n_r*n_sr of them) at axial
    separation delta * dz_sub, with the GMD self term at T[i, i, 0]. Sub-filaments sit on a uniform
    axial grid, so the sum over two slabs' sub-filaments is a Toeplitz sum
    S(D) = sum_{m=-(n_sz-1)}^{n_sz-1} (n_sz - |m|) T[|m + D|], D = slab offset * n_sz.
    Equal to the plain double sum over all sub-filaments, at O(n_rad^2 n_z n_sz) cost.
    """
    Tb = table.reshape(n_r, n_sr, n_r, n_sr, -1).sum(axis=(1, 3))  # (n_r, n_r, nzt)
    m = np.arange(-(n_sz - 1), n_sz)
    wgt = n_sz - np.abs(m)
    w = 1.0 / (n_sr * n_sz)
    S = np.stack([Tb[..., np.abs(m + D * n_sz)] @ wgt for D in range(n_z)])  # (n_z, n_r, n_r)
    p = np.arange(n_z)
    blocks = S[np.abs(p[:, None] - p[None, :])]  # (n_z, n_z, n_r, n_r)
    M = w * w * blocks.transpose(0, 2, 1, 3).reshape(n_z * n_r, n_z * n_r)
    return 0.5 * (M + M.T)


def _local_inductance(kind, edges, n_sr, n_z, n_sz, dz, A_B):
    """Band inductance matrix [H] in the local frame (pose-independent): sub-filament table + Toeplitz sum.

    Sub-filaments: n_sr per band radially (cell centres), n_sz per slab axially, near-square cells,
    GMD self terms (Grover rectangle); annular -> Maxwell, rect -> Neumann segment sums with the
    rectangle's mirror symmetry (M(A,B) = 4 sum over A's first quadrant).
    """
    n_r = len(edges) - 1
    span = np.diff(edges)
    sub = (edges[:-1, None] + (np.arange(n_sr)[None, :] + 0.5) * span[:, None] / n_sr).ravel()
    dsub = np.repeat(span / n_sr, n_sr)
    dzs = dz / n_sz
    delta = np.arange(n_z * n_sz) * dzs
    n_rad = len(sub)
    if kind == "annular":
        with np.errstate(divide="ignore", invalid="ignore"):
            T = maxwell_mutual(sub[:, None, None], sub[None, :, None], delta[None, None, :])
        T[np.arange(n_rad), np.arange(n_rad), 0] = maxwell_mutual(sub, sub, gmd_rectangle(dsub, dzs))
    else:
        A, B = A_B
        zhat = np.array([0.0, 0.0, 1.0])
        T = np.empty((n_rad, n_rad, len(delta)))
        quads = [outline_segments(0.5 * A * s, 0.5 * B * s, 0.0, 1, quadrant=True) for s in sub]
        fulls = [outline_segments(0.5 * A * s, 0.5 * B * s, 0.0, 1) for s in sub]
        shift = delta[None, None, :, None] * zhat
        for i in range(n_rad):
            qa1, qa2 = quads[i]
            for j in range(n_rad):
                fb1, fb2 = fulls[j]
                m = segment_mutual(qa1[:, None, None], qa2[:, None, None],
                                   fb1[None, :, None] + shift, fb2[None, :, None] + shift)
                T[i, j] = 4.0 * m.sum(axis=(0, 1))
            P1, P2 = fulls[i]
            x_run = np.abs(P2[:, 0] - P1[:, 0]) > np.abs(P2[:, 1] - P1[:, 1])
            g = np.where(x_run, gmd_rectangle(0.5 * B * dsub[i], dzs), gmd_rectangle(0.5 * A * dsub[i], dzs))
            T[i, i, 0] = _polygon_self(P1, g, zhat)
    return _band_matrix(T, n_r, n_sr, n_z, n_sz)


def eddy_loops(shape, dims, pose=None, n_r=4, n_z=2, resistivity=1.5e-6, axis=2, n_sub=None) -> EddyLoops:
    """Build the eddy-loop network of a conducting magnet.

    shape : "cuboid" (dims = (a, b, c) edge lengths along body x, y, z [m]),
            "cylinder" (dims = (diameter, height) [m], axis = body z),
            "ring" (dims = (outer diameter, inner diameter, height) [m], axis = body z)
    pose : None | (position [m], scipy Rotation) | object with .position/.rotation (body -> global)
    n_r, n_z : bands per slab and slabs along the axis
    resistivity : electrical resistivity [ohm m] (from the grade table; 1.5e-6 = NdFeB typical,
        `magnet_pulse_sim.py` source [M], used only as the default)
    axis : cuboid body axis (0, 1, 2) threaded by the coil field (loops lie in the plane normal to it)
    n_sub : (n_radial, n_axial) sub-filaments per band for the inductances; default near-square cells
    """
    position, rot = _parse_pose(pose)
    if n_r < 1 or n_z < 1 or resistivity <= 0:
        raise ValueError("need n_r >= 1, n_z >= 1 and resistivity > 0")
    if shape in ("cylinder", "ring"):
        if shape == "cylinder":
            d, h = dims
            r_in, r_out = 0.0, 0.5 * d
        else:
            d_out, d_in, h = dims
            r_in, r_out = 0.5 * d_in, 0.5 * d_out
            if not 0 < r_in < r_out:
                raise ValueError("ring needs 0 < inner diameter < outer diameter")
        kind, R_lb, A_B = "annular", np.eye(3), ()
        edges = np.linspace(r_in, r_out, n_r + 1)
        radial = (r_out - r_in) / n_r
    elif shape == "cuboid":
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        R_lb = _local_to_body(axis)
        A, B = float(dims[(axis + 1) % 3]), float(dims[(axis + 2) % 3])
        h = float(dims[axis])
        kind, A_B = "rect", (A, B)
        edges = np.linspace(0.0, 1.0, n_r + 1)
        radial = 0.25 * (A + B) / n_r  # mean band thickness of the x- and y-running sides
    else:
        raise ValueError(f"unknown shape {shape!r}")

    rot_total = rot * Rotation.from_matrix(R_lb)
    Rm = rot_total.as_matrix()
    nrm = Rm @ np.array([0.0, 0.0, 1.0])
    zed = np.linspace(-0.5 * h, 0.5 * h, n_z + 1)
    dz = h / n_z
    n_sr, n_sz = _auto_sub(radial, dz) if n_sub is None else (int(n_sub[0]), int(n_sub[1]))

    lo = np.tile(edges[:-1], n_z)
    hi = np.tile(edges[1:], n_z)
    z1 = np.repeat(zed[:-1], n_r)
    z2 = np.repeat(zed[1:], n_r)
    mid = 0.5 * (lo + hi)
    span = hi - lo
    if kind == "annular":
        Rk = resistivity * 2.0 * np.pi * mid / (span * dz)
    else:
        Rk = 4.0 * resistivity * mid * (A / B + B / A) / (span * dz)

    loops, gmds, owner, cen, rad, area = [], [], [], [], [], np.zeros(len(lo))
    w = 1.0 / (n_sr * n_sz)
    for k in range(len(lo)):
        for i in range(n_sr):
            s = lo[k] + (i + 0.5) * span[k] / n_sr
            for j in range(n_sz):
                zz = z1[k] + (j + 0.5) * dz / n_sz
                if kind == "annular":
                    v = outline_polygon(0.0, 0.0, s, 32)
                    g = np.full(len(v), float(gmd_rectangle(span[k] / n_sr, dz / n_sz)))
                    rad.append(s)
                    area[k] += w * np.pi * s * s
                else:
                    v = outline_polygon(0.5 * A * s, 0.5 * B * s, 0.0, 1)
                    seg = np.roll(v, -1, axis=0) - v
                    x_run = np.abs(seg[:, 0]) > np.abs(seg[:, 1])
                    gx = float(gmd_rectangle(0.5 * B * span[k] / n_sr, dz / n_sz))  # x-running sides
                    gy = float(gmd_rectangle(0.5 * A * span[k] / n_sr, dz / n_sz))  # y-running sides
                    g = np.where(x_run, gx, gy)
                    area[k] += w * A * B * s * s
                vb = np.hstack([v, np.full((len(v), 1), zz)])
                loops.append(vb @ Rm.T + position)
                gmds.append(g)
                owner.append(k)
                cen.append(Rm @ np.array([0.0, 0.0, zz]) + position)
    n_f = len(loops)
    fs = FilamentSet(
        loops=loops,
        seg_gmd=gmds,
        weight=np.full(n_f, w),
        owner=np.array(owner),
        normal=np.tile(nrm, (n_f, 1)),
        circle_center=np.array(cen) if kind == "annular" else None,
        circle_radius=np.array(rad) if kind == "annular" else None,
    )
    M = _local_inductance(kind, edges, n_sr, n_z, n_sz, dz, A_B)
    np.linalg.cholesky(M)  # raises LinAlgError if the network is not passive
    return EddyLoops(shape=shape, kind=kind, n_r=n_r, n_z=n_z, resistivity=float(resistivity), R=Rk, M=M,
                     lo=lo, hi=hi, z1=z1, z2=z2, in_plane=A_B, position=position, rot_total=rot_total,
                     _fil=fs, _area=area)
