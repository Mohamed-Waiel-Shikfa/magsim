"""Bodies, poses and meshing into uniformly polarised cells (task 3).

Conventions (docs/SPEC.md): SI units (m); global frame right-handed Cartesian. A body's pose is the
position of its centroid plus a ``scipy.spatial.transform.Rotation`` mapping body -> global
coordinates. Body frame: every shape is centred at the origin; cylinder / ring axis = body z. The
easy axis is given in the body frame (default +z).

Meshing produces structured grids so that neighbouring cells share whole faces (needed for the
net surface charge ``(J_minus - J_plus).n`` on internal faces, ``magsim.forces``):

- ``Cuboid`` -> nx x ny x nz cuboid cells (optionally geometrically graded toward the faces).
- ``Cylinder`` / ``Ring`` -> nr x nphi x nz cells of exact geometry. With ``nphi == 1`` every radial
  layer is a full annulus (cell kind ``ANNULUS``; its field is the exact superposition
  cylinder(r2) - cylinder(r1) of magpylib's fast full-cylinder kernel [caciagli2018]). With
  ``nphi > 1`` the radial layers are cylinder segments (``SEGMENT``, magpylib's exact but slow
  segment kernel [slanovc2022], ~100 us per evaluation). A full cylinder always keeps a central
  full-cylinder core cell (``CYLINDER``) so no cell has r_in = 0 and the axis is handled exactly.

Each cell carries: kind, magpylib source parameters (position of the cuboid centre / of the axis
point at mid-height, dimensions), centroid, volume, owning body, and Gauss points with weights
normalised to 1 per cell (q^3 tensor Gauss-Legendre; full rings use q radial x q axial x
max(2q, 4) equally spaced azimuthal points, exact for the rotational symmetry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

__all__ = [
    "Pose",
    "Cuboid",
    "Cylinder",
    "Ring",
    "Magnet",
    "MeshSpec",
    "BodyMesh",
    "BodyFaces",
    "CUBOID",
    "CYLINDER",
    "ANNULUS",
    "SEGMENT",
    "RECT",
    "ANNSEC",
    "CYLPATCH",
    "perp_axes",
    "check_overlap",
]

# cell kinds
CUBOID, CYLINDER, ANNULUS, SEGMENT = 0, 1, 2, 3
# face kinds
RECT, ANNSEC, CYLPATCH = 0, 1, 2


# ---------------------------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Pose:
    """Rigid placement of a body: centroid ``position`` [m] and body->global ``rotation``."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: Rotation = field(default_factory=Rotation.identity)

    def __post_init__(self):
        p = np.asarray(self.position, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError("pose position must be finite")
        object.__setattr__(self, "position", p)
        if self.rotation is None:
            object.__setattr__(self, "rotation", Rotation.identity())
        if not isinstance(self.rotation, Rotation) or self.rotation.single is False:
            raise TypeError("rotation must be a single scipy Rotation")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Pose:
        """Build from ``{position: [x,y,z], rotation: {euler_xyz_deg: [a,b,c]} | {rotvec_deg: [..]}}``.

        ``euler_xyz_deg`` is scipy's extrinsic ``"xyz"`` sequence (rotate about global x, then global
        y, then global z), in degrees. ``rotvec_deg`` is a rotation vector whose length is the angle
        in degrees. A missing rotation is the identity.
        """
        if d is None:
            return cls()
        if isinstance(d, Pose):
            return d
        pos = d.get("position", (0.0, 0.0, 0.0))
        rot = d.get("rotation")
        if rot is None:
            R = Rotation.identity()
        elif isinstance(rot, Rotation):
            R = rot
        elif "euler_xyz_deg" in rot:
            R = Rotation.from_euler("xyz", rot["euler_xyz_deg"], degrees=True)
        elif "rotvec_deg" in rot:
            R = Rotation.from_rotvec(np.deg2rad(np.asarray(rot["rotvec_deg"], dtype=float)))
        else:
            raise ValueError(f"rotation must give euler_xyz_deg or rotvec_deg, got {rot!r}")
        return cls(np.asarray(pos, dtype=float), R)

    @property
    def matrix(self) -> np.ndarray:
        """3x3 rotation matrix (body -> global)."""
        return self.rotation.as_matrix()

    def apply(self, p_body) -> np.ndarray:
        """Body-frame points (..., 3) -> global."""
        return np.asarray(p_body, dtype=float) @ self.matrix.T + self.position

    def to_body(self, p_global) -> np.ndarray:
        """Global points (..., 3) -> body frame."""
        return (np.asarray(p_global, dtype=float) - self.position) @ self.matrix

    def rotated(self, rot: Rotation, center=(0.0, 0.0, 0.0)) -> Pose:
        """This pose after a rigid rotation ``rot`` of space about ``center`` (global)."""
        c = np.asarray(center, dtype=float)
        return Pose(rot.apply(self.position - c) + c, rot * self.rotation)

    def translated(self, d) -> Pose:
        return Pose(self.position + np.asarray(d, dtype=float), self.rotation)


def perp_axes(e) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``e`` (unit, (..., 3)) to a right-handed frame (e1, e2, e)."""
    e = np.asarray(e, dtype=float)
    ref = np.where(np.abs(e[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(ref, e)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(e, e1)
    return e1, e2


# ---------------------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cuboid:
    """Rectangular block with full side lengths ``dims`` = (dx, dy, dz) [m], centred at the origin."""

    dims: tuple[float, float, float]

    def __post_init__(self):
        d = tuple(float(x) for x in np.asarray(self.dims, dtype=float).reshape(3))
        if min(d) <= 0:
            raise ValueError(f"cuboid dims must be positive, got {d}")
        object.__setattr__(self, "dims", d)

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

    @property
    def size(self) -> float:
        return float(max(self.dims))

    @property
    def bounding_radius(self) -> float:
        return 0.5 * float(np.linalg.norm(self.dims))

    def sdf(self, p) -> np.ndarray:
        """Signed distance [m] of body-frame points (negative inside)."""
        q = np.abs(np.asarray(p, dtype=float)) - 0.5 * np.asarray(self.dims)
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return out + np.minimum(q.max(axis=-1), 0.0)


@dataclass(frozen=True)
class Cylinder:
    """Solid cylinder (``diameter``, ``height`` [m]), axis = body z, centred at the origin."""

    diameter: float
    height: float

    def __post_init__(self):
        if self.diameter <= 0 or self.height <= 0:
            raise ValueError("cylinder diameter and height must be positive")

    @property
    def r_in(self) -> float:
        return 0.0

    @property
    def r_out(self) -> float:
        return 0.5 * float(self.diameter)

    @property
    def volume(self) -> float:
        return float(np.pi * self.r_out**2 * self.height)

    @property
    def size(self) -> float:
        return float(max(self.diameter, self.height))

    @property
    def bounding_radius(self) -> float:
        return float(np.hypot(self.r_out, 0.5 * self.height))

    def sdf(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        rho = np.hypot(p[..., 0], p[..., 1])
        d = np.stack([rho - self.r_out, np.abs(p[..., 2]) - 0.5 * self.height], axis=-1)
        return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


@dataclass(frozen=True)
class Ring:
    """Hollow cylinder (``d_inner`` < ``d_outer``, ``height`` [m]), axis = body z, centred at origin."""

    d_inner: float
    d_outer: float
    height: float

    def __post_init__(self):
        if not (0 < self.d_inner < self.d_outer) or self.height <= 0:
            raise ValueError("ring needs 0 < d_inner < d_outer and height > 0")

    @property
    def r_in(self) -> float:
        return 0.5 * float(self.d_inner)

    @property
    def r_out(self) -> float:
        return 0.5 * float(self.d_outer)

    @property
    def volume(self) -> float:
        return float(np.pi * (self.r_out**2 - self.r_in**2) * self.height)

    @property
    def size(self) -> float:
        return float(max(self.d_outer, self.height))

    @property
    def bounding_radius(self) -> float:
        return float(np.hypot(self.r_out, 0.5 * self.height))

    def sdf(self, p) -> np.ndarray:
        """Exact SDF of the revolved rectangle [r_in, r_out] x [-h/2, h/2] in the (rho, z) half-plane."""
        p = np.asarray(p, dtype=float)
        rho = np.hypot(p[..., 0], p[..., 1])
        dr = np.maximum(self.r_in - rho, rho - self.r_out)
        d = np.stack([dr, np.abs(p[..., 2]) - 0.5 * self.height], axis=-1)
        return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


# ---------------------------------------------------------------------------------------------
# Mesh specification and body meshes
# ---------------------------------------------------------------------------------------------
_DEFAULT_N = {Cuboid: (4, 4, 4), Cylinder: (3, 1, 6), Ring: (2, 1, 6)}


@dataclass(frozen=True)
class MeshSpec:
    """Structured mesh request.

    n : (nx, ny, nz) for cuboids, (nr, nphi, nz) for cylinders / rings (nr counts the core layer).
    grading : per-axis geometric ratio between adjacent cells (>= 1). Cuboid axes and the cylinder
        z axis are graded symmetrically toward both faces; the cylinder r axis toward the outer
        surface (full cylinder) or both surfaces (ring); phi is always uniform. 1 = uniform.
    q : Gauss points per direction for cell averaging (1 = centroid-like, 2 default).
    edges : optional explicit normalised edges (increasing, 0 ... 1) per axis, overriding grading.
    """

    n: tuple[int, int, int]
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0)
    q: int = 2
    edges: tuple[Any, Any, Any] = (None, None, None)

    @classmethod
    def parse(cls, mesh, shape) -> MeshSpec:
        if isinstance(mesh, MeshSpec):
            spec = mesh
        elif mesh is None:
            spec = cls(_DEFAULT_N[type(shape)])
        elif isinstance(mesh, dict):
            g = mesh.get("grading", (1.0, 1.0, 1.0))
            if np.isscalar(g):
                g = (g, g, g)
            spec = cls(
                tuple(int(k) for k in mesh.get("n", _DEFAULT_N[type(shape)])),
                tuple(float(x) for x in g),
                int(mesh.get("q", 2)),
                tuple(mesh.get("edges", (None, None, None))),
            )
        else:
            spec = cls(tuple(int(k) for k in mesh))
        if len(spec.n) != 3 or min(spec.n) < 1:
            raise ValueError(f"mesh n must be three positive integers, got {spec.n}")
        if spec.q < 1:
            raise ValueError("q must be >= 1")
        if min(spec.grading) < 1.0:
            raise ValueError("grading ratios must be >= 1")
        return spec


def _graded_edges(n, ratio, mode, explicit=None) -> np.ndarray:
    """Normalised edges in [0, 1] with geometric grading.

    mode "both": cells shrink by ``ratio`` per step toward both ends; "hi": toward 1 only.
    """
    if explicit is not None:
        e = np.asarray(explicit, dtype=float)
        if e.ndim != 1 or e.size != n + 1 or abs(e[0]) > 1e-15 or abs(e[-1] - 1) > 1e-12 or np.any(np.diff(e) <= 0):
            raise ValueError(f"explicit edges must be {n + 1} increasing values from 0 to 1")
        e = e.copy()
        e[0], e[-1] = 0.0, 1.0
        return e
    k = np.arange(n)
    if mode == "both":
        sizes = float(ratio) ** np.minimum(k, n - 1 - k)
    else:
        sizes = float(ratio) ** (n - 1 - k)
    e = np.concatenate([[0.0], np.cumsum(sizes)])
    e /= e[-1]
    e[-1] = 1.0
    return e


def _gl(q):
    x, w = np.polynomial.legendre.leggauss(q)
    return 0.5 * (x + 1.0), 0.5 * w  # nodes / weights on [0, 1]


@dataclass
class BodyFaces:
    """Charge-carrying faces of a structured body mesh, in the body frame.

    Each face carries net charge sigma(p) = (J[minus] - J[plus]) . n(p) [T], where ``minus`` / ``plus``
    are the cells on the -n / +n side (-1 = outside the body, J = 0). Geometry by ``kind``:
    RECT p = o + s e1 + t e2, n = e3 (constant); ANNSEC p = o + s (cos t e1 + sin t e2), n = e3,
    jacobian s; CYLPATCH p = o + R (cos s e1 + sin s e2) + t e3, n = cos s e1 + sin s e2, jacobian R.
    Parameter ranges ``rng`` = (s0, s1, t0, t1).
    """

    kind: np.ndarray
    o: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    e3: np.ndarray
    rng: np.ndarray
    R: np.ndarray
    minus: np.ndarray
    plus: np.ndarray
    axis_id: np.ndarray  # RECT faces of cuboid meshes: body axis of the normal (0, 1, 2); else -1

    def __len__(self):
        return int(self.kind.shape[0])


@dataclass
class BodyMesh:
    """Cells of one body in its own frame (see module docstring for the cell data)."""

    kind: np.ndarray  # (n,) cell kind
    pos: np.ndarray  # (n, 3) magpylib source position (cuboid centre / axis point at mid-height)
    dims: np.ndarray  # (n, 5) cuboid (dx, dy, dz, 0, 0); cylinder types (r1, r2, h, phi1, phi2) [m, rad]
    centroid: np.ndarray  # (n, 3)
    volume: np.ndarray  # (n,)
    bsphere_c: np.ndarray  # (n, 3) bounding-sphere centre
    bsphere_r: np.ndarray  # (n,) bounding-sphere radius
    gp: np.ndarray  # (m, 3) Gauss points
    gw: np.ndarray  # (m,) weights, sum 1 per cell
    gcell: np.ndarray  # (m,) owning cell
    gphi0: np.ndarray  # (m,) True for the first azimuthal Gauss point of full rings (all points otherwise)
    nphi_g: np.ndarray  # (n,) azimuthal Gauss points per cell (full rings), 1 otherwise
    faces: BodyFaces
    grid: dict

    @property
    def n_cells(self) -> int:
        return int(self.kind.shape[0])


def _mesh_cuboid(shape: Cuboid, spec: MeshSpec) -> BodyMesh:
    d = np.asarray(shape.dims)
    E = [(_graded_edges(spec.n[a], spec.grading[a], "both", spec.edges[a]) - 0.5) * d[a] for a in range(3)]
    xe, ye, ze = E
    nx, ny, nz = spec.n
    lo = np.stack(np.meshgrid(xe[:-1], ye[:-1], ze[:-1], indexing="ij"), -1).reshape(-1, 3)
    hi = np.stack(np.meshgrid(xe[1:], ye[1:], ze[1:], indexing="ij"), -1).reshape(-1, 3)
    n = lo.shape[0]
    ctr = 0.5 * (lo + hi)
    size = hi - lo
    dims = np.zeros((n, 5))
    dims[:, :3] = size
    x, w = _gl(spec.q)
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    gwt = np.einsum("i,j,k->ijk", w, w, w).ravel()
    gp = (lo[:, None, :] + g[None, :, :] * size[:, None, :]).reshape(-1, 3)
    gw = np.tile(gwt, n)
    gcell = np.repeat(np.arange(n), g.shape[0])
    idx = np.arange(n).reshape(nx, ny, nz)
    faces = _faces_cuboid(E, idx)
    return BodyMesh(
        kind=np.full(n, CUBOID),
        pos=ctr,
        dims=dims,
        centroid=ctr.copy(),
        volume=size.prod(axis=1),
        bsphere_c=ctr.copy(),
        bsphere_r=0.5 * np.linalg.norm(size, axis=1),
        gp=gp,
        gw=gw,
        gcell=gcell,
        gphi0=np.ones(gw.size, dtype=bool),
        nphi_g=np.ones(n, dtype=int),
        faces=faces,
        grid={"type": "cuboid", "edges": E, "idx": idx},
    )


def _faces_cuboid(E, idx) -> BodyFaces:
    out = {k: [] for k in ("kind", "o", "e1", "e2", "e3", "rng", "R", "minus", "plus", "axis_id")}
    I3 = np.eye(3)
    shape = idx.shape
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3  # right-handed (e_b, e_c, e_a)
        na = shape[a]
        for m in range(na + 1):
            # cells on the minus (m-1) and plus (m) side of plane m along axis a
            sl_minus = np.take(idx, m - 1, axis=a) if m > 0 else None
            sl_plus = np.take(idx, m, axis=a) if m < na else None
            # remaining axes after removing a, in increasing order
            rem = [ax for ax in range(3) if ax != a]
            g0, g1 = np.meshgrid(np.arange(shape[rem[0]]), np.arange(shape[rem[1]]), indexing="ij")
            g0, g1 = g0.ravel(), g1.ravel()
            nf = g0.size
            mi = sl_minus[g0, g1] if sl_minus is not None else np.full(nf, -1)
            pl = sl_plus[g0, g1] if sl_plus is not None else np.full(nf, -1)
            coord = {rem[0]: g0, rem[1]: g1}
            lo_b = E[b][coord[b]]
            hi_b = E[b][coord[b] + 1]
            lo_c = E[c][coord[c]]
            hi_c = E[c][coord[c] + 1]
            o = np.zeros((nf, 3))
            o[:, a] = E[a][m]
            out["kind"].append(np.full(nf, RECT))
            out["o"].append(o)
            out["e1"].append(np.tile(I3[b], (nf, 1)))
            out["e2"].append(np.tile(I3[c], (nf, 1)))
            out["e3"].append(np.tile(I3[a], (nf, 1)))
            out["rng"].append(np.stack([lo_b, hi_b, lo_c, hi_c], axis=1))
            out["R"].append(np.zeros(nf))
            out["minus"].append(mi)
            out["plus"].append(pl)
            out["axis_id"].append(np.full(nf, a))
    return BodyFaces(**{k: np.concatenate(v) for k, v in out.items()})


def _mesh_cylindrical(shape, spec: MeshSpec) -> BodyMesh:
    nr, nphi, nz = spec.n
    r_in, r_out, h = shape.r_in, shape.r_out, float(shape.height)
    has_core = r_in == 0.0
    re = r_in + (r_out - r_in) * _graded_edges(nr, spec.grading[0], "hi" if has_core else "both", spec.edges[0])
    ze = (_graded_edges(nz, spec.grading[2], "both", spec.edges[2]) - 0.5) * h
    if spec.edges[1] is not None:
        raise ValueError("explicit phi edges are not supported (phi is always uniform)")
    pe = np.linspace(0.0, 2.0 * np.pi, nphi + 1)
    q = spec.q
    xr, wr = _gl(q)
    xz, wz = _gl(q)
    nphi_full = max(2 * q, 4)
    # index map idx[ir, j, iz]; the core (ir = 0 of a full cylinder) is one cell per layer
    cells = []  # (kind, ir, j, iz)
    idx = np.full((nr, nphi, nz), -1, dtype=int)
    for iz in range(nz):
        for ir in range(nr):
            if ir == 0 and has_core:
                idx[0, :, iz] = len(cells)
                cells.append((CYLINDER, 0, 0, iz))
            elif nphi == 1:
                idx[ir, 0, iz] = len(cells)
                cells.append((ANNULUS, ir, 0, iz))
            else:
                for j in range(nphi):
                    idx[ir, j, iz] = len(cells)
                    cells.append((SEGMENT, ir, j, iz))
    n = len(cells)
    kind = np.array([c[0] for c in cells])
    ir = np.array([c[1] for c in cells])
    jj = np.array([c[2] for c in cells])
    iz = np.array([c[3] for c in cells])
    r1, r2 = re[ir], re[ir + 1]
    z1, z2 = ze[iz], ze[iz + 1]
    full = kind != SEGMENT
    p1 = np.where(full, 0.0, pe[jj])
    p2 = np.where(full, 2.0 * np.pi, pe[jj + 1])
    hh = z2 - z1
    zm = 0.5 * (z1 + z2)
    pos = np.stack([np.zeros(n), np.zeros(n), zm], axis=1)
    dims = np.stack([r1, r2, hh, p1, p2], axis=1)
    dphi = p2 - p1
    volume = 0.5 * dphi * (r2**2 - r1**2) * hh
    rbar = (2.0 / 3.0) * (r2**3 - r1**3) / (r2**2 - r1**2)
    pm = 0.5 * (p1 + p2)
    sinc = np.where(full, 0.0, np.sin(0.5 * dphi) / (0.5 * dphi))
    centroid = np.stack([rbar * sinc * np.cos(pm), rbar * sinc * np.sin(pm), zm], axis=1)
    # bounding spheres: full rings about the axis point; segments about the centroid (sampled hull)
    bc = np.where(full[:, None], pos, centroid)
    ang = p1[:, None] + dphi[:, None] * np.linspace(0, 1, 17)[None, :]
    ext = []
    for rr in (r1, r2):
        for zz in (z1, z2):
            pts = np.stack([rr[:, None] * np.cos(ang), rr[:, None] * np.sin(ang), np.broadcast_to(zz[:, None], ang.shape)], -1)
            ext.append(np.linalg.norm(pts - bc[:, None, :], axis=-1).max(axis=1))
    br = np.max(np.stack(ext), axis=0) * 1.01
    br = np.where(full, np.hypot(r2, 0.5 * hh), br)
    # Gauss points
    gps, gws, gcs, g0s = [], [], [], []
    nphi_g = np.where(full, nphi_full, 1)
    for c in range(n):
        rr = r1[c] + (r2[c] - r1[c]) * xr
        wrr = wr * rr
        zz = z1[c] + hh[c] * xz
        if kind[c] == SEGMENT:
            pp = p1[c] + dphi[c] * xr
            wp = wr
        else:
            pp = (np.arange(nphi_full) + 0.5) * 2.0 * np.pi / nphi_full
            wp = np.full(nphi_full, 1.0 / nphi_full)
        R_, P_, Z_ = np.meshgrid(rr, pp, zz, indexing="ij")
        first = np.zeros(R_.shape, dtype=bool)
        first[:, 0, :] = True
        g0s.append(first.ravel() if kind[c] != SEGMENT else np.ones(R_.size, dtype=bool))
        W = np.einsum("i,j,k->ijk", wrr, wp, wz).ravel()
        gps.append(np.stack([(R_ * np.cos(P_)).ravel(), (R_ * np.sin(P_)).ravel(), Z_.ravel()], 1))
        gws.append(W / W.sum())
        gcs.append(np.full(W.size, c))
    faces = _faces_cyl(re, pe, ze, idx, has_core, nphi)
    return BodyMesh(
        kind=kind,
        pos=pos,
        dims=dims,
        centroid=centroid,
        volume=volume,
        bsphere_c=bc,
        bsphere_r=br,
        gp=np.concatenate(gps),
        gw=np.concatenate(gws),
        gcell=np.concatenate(gcs),
        gphi0=np.concatenate(g0s),
        nphi_g=nphi_g,
        faces=faces,
        grid={"type": "cylinder", "r_edges": re, "phi_edges": pe, "z_edges": ze, "idx": idx, "has_core": has_core},
    )


def _faces_cyl(re, pe, ze, idx, has_core, nphi) -> BodyFaces:
    nr, _, nz = idx.shape
    ex, ey, ez = np.eye(3)
    F = {k: [] for k in ("kind", "o", "e1", "e2", "e3", "rng", "R", "minus", "plus", "axis_id")}

    def add(kind, o, e1, e2, e3, rng, R, mi, pl):
        F["kind"].append(kind)
        F["o"].append(o)
        F["e1"].append(e1)
        F["e2"].append(e2)
        F["e3"].append(e3)
        F["rng"].append(rng)
        F["R"].append(R)
        F["minus"].append(mi)
        F["plus"].append(pl)
        F["axis_id"].append(-1)

    def cell(i, j, k):
        if k < 0 or k >= nz or i < 0 or i >= nr:
            return -1
        return int(idx[i, j, k])

    # z faces (normal +z): core disc + annular sectors / annuli
    for m in range(nz + 1):
        o = np.array([0.0, 0.0, ze[m]])
        for i in range(nr):
            if (i == 0 and has_core) or nphi == 1:
                add(ANNSEC, o, ex, ey, ez, (re[i], re[i + 1], 0.0, 2 * np.pi), 0.0, cell(i, 0, m - 1), cell(i, 0, m))
            else:
                for j in range(nphi):
                    add(ANNSEC, o, ex, ey, ez, (re[i], re[i + 1], pe[j], pe[j + 1]), 0.0, cell(i, j, m - 1), cell(i, j, m))
    # r faces (normal +r): at each radial edge except r = 0
    for i in range(nr + 1):
        if i == 0 and has_core:
            continue
        for m in range(nz):
            o = np.zeros(3)
            # split by the finer phi partition of the two sides
            inner_full = i == 0 or (i == 1 and has_core) or nphi == 1
            outer_full = i == nr or nphi == 1
            if inner_full and outer_full:
                segs = [(0, 0.0, 2 * np.pi)]
            else:
                segs = [(j, pe[j], pe[j + 1]) for j in range(nphi)]
            for j, a0, a1 in segs:
                mi = cell(i - 1, j if not inner_full else 0, m) if i > 0 else -1
                pl = cell(i, j if not outer_full else 0, m) if i < nr else -1
                add(CYLPATCH, o, ex, ey, ez, (a0, a1, ze[m], ze[m + 1]), re[i], mi, pl)
    # phi faces (normal +phi_hat at phi_j): only between segments
    if nphi > 1:
        for i in range(1 if has_core else 0, nr):
            for j in range(nphi):
                c, s = np.cos(pe[j]), np.sin(pe[j])
                rhat = np.array([c, s, 0.0])
                phat = np.array([-s, c, 0.0])
                for m in range(nz):
                    add(RECT, np.zeros(3), rhat, ez, phat, (re[i], re[i + 1], ze[m], ze[m + 1]), 0.0,
                        cell(i, (j - 1) % nphi, m), cell(i, j, m))
    out = {}
    for k, v in F.items():
        out[k] = np.array(v, dtype=float if k in ("o", "e1", "e2", "e3", "rng", "R") else int)
    return BodyFaces(**out)


def mesh_shape(shape, mesh) -> BodyMesh:
    """Mesh ``shape`` according to ``mesh`` (tuple, dict or ``MeshSpec``), in the body frame."""
    spec = MeshSpec.parse(mesh, shape)
    if isinstance(shape, Cuboid):
        return _mesh_cuboid(shape, spec)
    if isinstance(shape, (Cylinder, Ring)):
        return _mesh_cylindrical(shape, spec)
    raise TypeError(f"unsupported shape {type(shape).__name__}")


# ---------------------------------------------------------------------------------------------
# Magnet
# ---------------------------------------------------------------------------------------------
class Magnet:
    """A permanent-magnet body (interface contract in docs/SPEC.md).

    name : unique label; shape : Cuboid | Cylinder | Ring; pose : Pose or YAML-like dict;
    material : object implementing ``magsim.materials.base.Material``; easy_axis : body-frame
    direction (normalised); mesh : see ``MeshSpec``; initial : passed to ``material.init_state``;
    temperature [K].
    """

    def __init__(self, name, shape, pose=None, material=None, easy_axis=(0.0, 0.0, 1.0), mesh=None,
                 initial="+sat", temperature=293.15):
        if material is None:
            raise ValueError("a material is required")
        self.name = str(name)
        self.shape = shape
        self.pose = Pose.from_dict(pose) if not isinstance(pose, Pose) else pose
        self.material = material
        e = np.asarray(easy_axis, dtype=float).reshape(3)
        if np.linalg.norm(e) == 0:
            raise ValueError("easy_axis must be non-zero")
        self.easy_axis = e / np.linalg.norm(e)
        self.mesh_spec = MeshSpec.parse(mesh, shape)
        self.initial = initial
        if not np.all(np.isfinite(np.asarray(temperature, dtype=float))) or np.any(np.asarray(temperature) <= 0):
            raise ValueError("temperature must be positive [K]")
        self.temperature = temperature
        self.body_mesh = mesh_shape(shape, self.mesh_spec)

    @property
    def n_cells(self) -> int:
        return self.body_mesh.n_cells

    def __repr__(self):
        return f"Magnet({self.name!r}, {self.shape}, cells={self.n_cells})"


# ---------------------------------------------------------------------------------------------
# Overlap check
# ---------------------------------------------------------------------------------------------
def _sat_cuboids(ma: Magnet, mb: Magnet, tol: float) -> bool:
    """Separating-axis test for two oriented boxes. True if interiors overlap by more than tol."""
    Ra, Rb = ma.pose.matrix, mb.pose.matrix
    ha, hb = 0.5 * np.asarray(ma.shape.dims), 0.5 * np.asarray(mb.shape.dims)
    t = mb.pose.position - ma.pose.position
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[:, i], Rb[:, j])
            if np.linalg.norm(c) > 1e-9:
                axes.append(c / np.linalg.norm(c))
    for L in axes:
        ra = np.sum(ha * np.abs(Ra.T @ L))
        rb = np.sum(hb * np.abs(Rb.T @ L))
        if abs(t @ L) >= ra + rb - tol:
            return False
    return True


def _surface_samples(shape, n=10) -> np.ndarray:
    """Body-frame sample points covering the volume, surfaces, edges and corners of ``shape``."""
    s = np.linspace(-0.5, 0.5, n)
    if isinstance(shape, Cuboid):
        g = np.stack(np.meshgrid(s, s, s, indexing="ij"), -1).reshape(-1, 3)
        return g * np.asarray(shape.dims)
    rr = np.linspace(shape.r_in, shape.r_out, n)
    pp = np.linspace(0, 2 * np.pi, 4 * n, endpoint=False)
    zz = s * shape.height
    R_, P_, Z_ = np.meshgrid(rr, pp, zz, indexing="ij")
    return np.stack([(R_ * np.cos(P_)).ravel(), (R_ * np.sin(P_)).ravel(), Z_.ravel()], 1)


def _param_point(shape, x):
    """Map an unconstrained 3-vector x to a point of ``shape`` (body frame) via smooth clamping."""
    t = 0.5 * np.tanh(x)  # in (-0.5, 0.5)
    if isinstance(shape, Cuboid):
        return 2.0 * t * 0.5 * np.asarray(shape.dims) * 1.0
    r = shape.r_in + (t[0] + 0.5) * (shape.r_out - shape.r_in)
    phi = 2.0 * np.pi * x[1]
    return np.array([r * np.cos(phi), r * np.sin(phi), t[2] * shape.height])


def _overlap_depth(ma: Magnet, mb: Magnet) -> float:
    """Approximate max penetration of body a into body b: -min_{p in a} sdf_b(p) (>0 = overlap)."""
    pa = ma.pose.apply(_surface_samples(ma.shape))
    d = mb.shape.sdf(mb.pose.to_body(pa))
    best = np.argsort(d)[:3]
    worst = float(d.min())

    def f(x):
        return float(mb.shape.sdf(mb.pose.to_body(ma.pose.apply(_param_point(ma.shape, x)))))

    for k in best:
        p = _surface_samples(ma.shape)[k]
        if isinstance(ma.shape, Cuboid):
            x0 = np.arctanh(np.clip(2.0 * p / np.asarray(ma.shape.dims), -0.999999, 0.999999))
        else:
            rr = np.hypot(p[0], p[1])
            x0 = np.array([
                np.arctanh(np.clip(2 * ((rr - ma.shape.r_in) / (ma.shape.r_out - ma.shape.r_in)) - 1, -0.999999, 0.999999)),
                np.arctan2(p[1], p[0]) / (2 * np.pi),
                np.arctanh(np.clip(2 * p[2] / ma.shape.height, -0.999999, 0.999999)),
            ])
        res = minimize(f, x0, method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 400})
        worst = min(worst, float(res.fun))
    return -worst


def check_overlap(magnets, rtol=1e-9) -> None:
    """Raise ValueError if any two bodies interpenetrate (touching is allowed).

    Cuboid pairs: exact separating-axis test. Pairs involving cylinders / rings: dense sampling of each
    body against the other's exact signed-distance function, refined by local minimisation; detects
    penetrations deeper than ``rtol`` x size reliably once they exceed ~1e-3 of the body size
    (shallower ones are found when the sampling or the refinement lands on them).
    """
    ms = list(magnets)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            tol = rtol * max(a.shape.size, b.shape.size)
            dist = np.linalg.norm(a.pose.position - b.pose.position)
            if dist >= a.shape.bounding_radius + b.shape.bounding_radius - tol:
                continue
            if isinstance(a.shape, Cuboid) and isinstance(b.shape, Cuboid):
                bad = _sat_cuboids(a, b, tol)
            else:
                bad = max(_overlap_depth(a, b), _overlap_depth(b, a)) > tol
            if bad:
                raise ValueError(f"magnets {a.name!r} and {b.name!r} overlap")
