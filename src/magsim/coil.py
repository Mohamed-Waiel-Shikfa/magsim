"""Coil geometry, inductance, resistance, heating and field per amp (task 4).

Frames and signs
----------------
A coil is built in its body frame: winding axis = body z, winding centred on the origin. Positive
current circulates counter-clockwise seen from +z, so the field inside the bore points along +z.
The pose maps body to global coordinates: ``x_global = rotation.apply(x_body) + position``.

Filament model [grover1946, ruehli1972]
---------------------------------------
Every turn is a closed planar loop in a plane z = const. The helical pitch, layer cross-overs and the
foil spiral transition are ignored; `geometry_report()` bounds that error (path length
sqrt(1 + (p/P)^2) - 1 ~ (p/P)^2/2 for pitch p and turn perimeter P; the net axial current of one layer
is a straight conductor of the winding length, ~nH, and alternate layers cancel it). A conductor
cross-section is split into sub-cells, each a filament carrying an equal share of the turn current
(uniform current density, DC). Mutual inductances:

* coaxial circles: Maxwell's elliptic-integral formula (`maxwell_mutual`);
* straight segments: closed-form Neumann double integral (`segment_mutual`), parallel and oblique;
* self terms: the loop's mutual with a copy of itself displaced along its axis by the geometric mean
  distance (GMD) of the sub-cell cross-section (round wire r e^(-1/4), rectangle per Grover,
  `gmd_rectangle`). For straight segments this is exactly the GMD partial self-inductance.

Rounded corners are inscribed polygons of `n_arc` chords per quarter arc. Round wire uses one filament
per turn: for uniform current the GMD between two round wires is their centre distance, so this is
exact apart from O((r/R)^2) curvature terms. Rectangular conductors (foil) use near-square sub-cells,
for which the centre-distance approximation of the cell-pair GMD is 4th-order accurate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from magpylib.func import circle_field, cylinder_field, polyline_field
from numpy.polynomial.legendre import leggauss
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.spatial.transform import Rotation

from magsim.constants import ALPHA_CU, CP_CU, DENSITY_CU, MU0, RHO_CU_20

T_REF = 293.15  # reference temperature of RHO_CU_20 and ALPHA_CU [K]
_CHUNK = 200_000  # max (point x source) instances per vectorised block (memory bound)
_KEY_RES = 1e-12  # [m] length resolution used to detect identical shapes / separations

__all__ = [
    "FilamentSet",
    "FoilCoil",
    "RoundWireCoil",
    "circuit_inductance",
    "circuit_mutual",
    "copper_resistivity",
    "dowell_factor",
    "gmd_rectangle",
    "gmd_round",
    "lorenz_sheet_inductance",
    "loop_axis_field",
    "maxwell_mutual",
    "nagaoka",
    "reference_coil_r1",
    "reference_coil_r2",
    "reference_coil_r3",
    "segment_mutual",
    "skin_depth",
    "strip_sheet_field",
]


# =====================================================================================================
# Material and closed-form kernels
# =====================================================================================================
def copper_resistivity(T):
    """Copper resistivity rho_Cu(T) = RHO_CU_20 (1 + ALPHA_CU (T - 293.15)) [ohm m], T in K.

    Linear about 20 degC (IACS / IEC 60028 constants in `magsim.constants`); within 0.5 % of the
    tabulated data of [matula1979] between 250 K and 400 K.
    """
    return RHO_CU_20 * (1.0 + ALPHA_CU * (np.asarray(T, dtype=float) - T_REF))


def maxwell_mutual(r1, r2, dz):
    """Mutual inductance [H] of two coaxial circular filaments, radii r1, r2 [m], axial offset dz [m].

    Maxwell's formula M = mu0 sqrt(r1 r2) [(2/k - k) K(k) - (2/k) E(k)],
    k^2 = 4 r1 r2 / ((r1 + r2)^2 + dz^2) [grover1946, ch. 13]. K is evaluated with `ellipkm1` on
    1 - k^2 computed without cancellation (close filaments, GMD self terms); for k^2 < 1e-3 the
    bracket uses its series (pi k^3/16)(1 + 3k^2/4 + 75k^4/128 + 245k^6/512) to avoid cancellation.
    Broadcasts over its arguments.
    """
    r1, r2, dz = np.broadcast_arrays(np.asarray(r1, float), np.asarray(r2, float), np.asarray(dz, float))
    shape = r1.shape
    r1, r2, dz = r1.ravel(), r2.ravel(), np.abs(dz.ravel())
    den = (r1 + r2) ** 2 + dz**2
    m = 4.0 * r1 * r2 / den
    p = ((r1 - r2) ** 2 + dz**2) / den  # = 1 - m, no cancellation
    k = np.sqrt(m)
    out = np.zeros_like(m)
    big = m >= 1e-3
    kb, mb = k[big], m[big]
    out[big] = (2.0 / kb - kb) * ellipkm1(p[big]) - (2.0 / kb) * ellipe(mb)
    ks, ms = k[~big], m[~big]
    out[~big] = np.pi * ks**3 / 16.0 * (1.0 + 0.75 * ms + 75.0 / 128.0 * ms**2 + 245.0 / 512.0 * ms**3)
    return (MU0 * np.sqrt(r1 * r2) * out).reshape(shape)


def segment_mutual(P1, P2, Q1, Q2):
    """Mutual (partial) inductance [H] of straight filaments P1->P2 and Q1->Q2 [m], broadcast.

    Neumann integral M = mu0/(4 pi) (u.v) int int ds dt / |P(s) - Q(t)| in closed form
    [grover1946, ch. 7; ruehli1972]. Oblique pairs use the double antiderivative in coordinates
    (sigma, tau) measured from the feet of the common perpendicular (length d, c = cos(eps)):
        F = sigma ln(tau - sigma c + r) + tau ln(sigma - tau c + r)
            - (d / sin eps) atan[(sigma tau sin^2 eps + d^2 c) / (d r sin eps)],
    M = mu0/(4 pi) c [F(s2,t2) - F(s1,t2) - F(s2,t1) + F(s1,t1)]. The log arguments are rationalised
    when they would cancel. Pairs with sin(eps) < 2e-5 are treated as parallel with
    G(x) = x asinh(x/d) - sqrt(x^2 + d^2) (relative error < 1e-5 for such near-parallel pairs; the oblique form loses < 1e-6 there).
    Zero-length segments give 0. Coincident overlapping filaments diverge (use a GMD offset).
    """
    arrs = np.broadcast_arrays(*(np.asarray(x, float) for x in (P1, P2, Q1, Q2)))
    shape = arrs[0].shape[:-1]
    P1, P2, Q1, Q2 = (a.reshape(-1, 3) for a in arrs)
    dp, dq = P2 - P1, Q2 - Q1
    lp = np.linalg.norm(dp, axis=1)
    lq = np.linalg.norm(dq, axis=1)
    ok = (lp > 0) & (lq > 0)
    u = dp / np.where(lp > 0, lp, 1.0)[:, None]
    v = dq / np.where(lq > 0, lq, 1.0)[:, None]
    c = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
    cr = np.cross(u, v)
    sn = np.linalg.norm(cr, axis=1)
    out = np.zeros(len(P1))

    par = ok & (sn < 2e-5)
    if np.any(par):
        p1, q1, q2, uu, ll = P1[par], Q1[par], Q2[par], u[par], lp[par]
        a1 = np.sum((q1 - p1) * uu, axis=1)
        a2 = np.sum((q2 - p1) * uu, axis=1)
        lo, hi = np.minimum(a1, a2), np.maximum(a1, a2)
        mid = 0.5 * (q1 + q2) - p1
        perp = mid - np.sum(mid * uu, axis=1)[:, None] * uu
        d = np.maximum(np.linalg.norm(perp, axis=1), 1e-15)

        def G(x):
            return x * np.arcsinh(x / d) - np.sqrt(x * x + d * d)

        out[par] = c[par] * (G(ll - lo) - G(ll - hi) - G(-lo) + G(-hi))

    gen = ok & ~par
    if np.any(gen):
        p1, p2, q1, q2 = P1[gen], P2[gen], Q1[gen], Q2[gen]
        uu, vv, cc, ss, cg = u[gen], v[gen], c[gen], sn[gen], cr[gen]
        w0 = p1 - q1
        a = np.sum(w0 * uu, axis=1)
        b = np.sum(w0 * vv, axis=1)
        sn2 = ss * ss
        s0 = (cc * b - a) / sn2
        t0 = (b - cc * a) / sn2
        d = np.abs(np.sum(w0 * cg, axis=1)) / ss

        def F(P, Q, sig, tau):
            D = Q - P  # from the point on P to the point on Q
            r = np.linalg.norm(D, axis=1)
            x1 = np.sum(D * vv, axis=1)  # = tau - sigma c
            x2 = -np.sum(D * uu, axis=1)  # = sigma - tau c
            qq1 = np.sum(np.cross(D, vv) ** 2, axis=1)  # = r^2 - x1^2
            qq2 = np.sum(np.cross(D, uu) ** 2, axis=1)  # = r^2 - x2^2
            with np.errstate(divide="ignore", invalid="ignore"):
                L1 = np.where(x1 >= 0, np.log(x1 + r), np.log(np.maximum(qq1, 1e-300)) - np.log(r - x1))
                L2 = np.where(x2 >= 0, np.log(x2 + r), np.log(np.maximum(qq2, 1e-300)) - np.log(r - x2))
            # sigma ln(.) and tau ln(.) -> 0 where the two points (nearly) touch: there sigma, tau = O(r)
            touch = r < 1e-12 * scale
            with np.errstate(invalid="ignore"):
                t1 = np.where(touch | (sig == 0), 0.0, sig * L1)
                t2 = np.where(touch | (tau == 0), 0.0, tau * L2)
            t3 = (d / ss) * np.arctan2(sig * tau * sn2 + d * d * cc, d * r * ss)
            return t1 + t2 - t3

        scale = np.maximum(lp[gen], lq[gen])

        s1, s2 = -s0, lp[gen] - s0
        tt1, tt2 = -t0, lq[gen] - t0
        val = F(p2, q2, s2, tt2) - F(p1, q2, s1, tt2) - F(p2, q1, s2, tt1) + F(p1, q1, s1, tt1)
        out[gen] = cc * val
    return (MU0 / (4.0 * np.pi) * out).reshape(shape)


def gmd_round(r):
    """Self GMD [m] of a round conductor of radius r [m] with uniform current: r e^(-1/4) [grover1946]."""
    return np.asarray(r, float) * np.exp(-0.25)


def gmd_rectangle(a, b):
    """Self GMD [m] of a rectangle a x b [m] with uniform current, exact (Rosa/Grover [grover1946]).

    ln g = ln sqrt(a^2+b^2) - (a^2/12b^2) ln(1+b^2/a^2) - (b^2/12a^2) ln(1+a^2/b^2)
           + (2a/3b) atan(b/a) + (2b/3a) atan(a/b) - 25/12.
    Limits: square 0.44705 a; thin strip (b -> 0) a e^(-3/2).
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    a, b = np.maximum(a, b), np.minimum(a, b)  # formula is symmetric; keep a >= b
    with np.errstate(divide="ignore", invalid="ignore"):
        x = b / a
        lg = (
            0.5 * np.log(a * a + b * b)
            - np.log1p(x * x) / (12.0 * x * x)
            - (x * x / 12.0) * np.log1p(1.0 / (x * x))
            + (2.0 / (3.0 * x)) * np.arctan(x)
            + (2.0 * x / 3.0) * np.arctan(1.0 / x)
            - 25.0 / 12.0
        )
    lg = np.where(b > 0, lg, np.log(a) - 1.5)
    return np.exp(lg)


def nagaoka(radius, length):
    """Nagaoka coefficient of a uniform cylindrical current sheet (Lorenz formula) [grover1946, ch. 15].

    K_N = 4/(3 pi k') [ (k'^2/k^2)(K(k) - E(k)) + E(k) - k ], k^2 = 4a^2/(4a^2 + l^2),
    k' = l / sqrt(4a^2 + l^2).
    """
    a, l = float(radius), float(length)
    m = 4.0 * a * a / (4.0 * a * a + l * l)
    k, kp = np.sqrt(m), l / np.sqrt(4.0 * a * a + l * l)
    K, E = ellipk(m), ellipe(m)
    return 4.0 / (3.0 * np.pi * kp) * ((kp * kp / m) * (K - E) + E - k)


def lorenz_sheet_inductance(radius, length, n_turns):
    """Exact inductance [H] of a uniform cylindrical current sheet (single-layer solenoid limit).

    L = mu0 N^2 pi a^2 / l * K_N (Lorenz/Nagaoka) [grover1946, ch. 15].
    """
    return MU0 * n_turns**2 * np.pi * radius**2 / length * nagaoka(radius, length)


def loop_axis_field(radius, z):
    """On-axis field H_z [A/m per A] of a circular filament: R^2 / (2 (R^2 + z^2)^(3/2))."""
    R, z = float(radius), np.asarray(z, float)
    return R * R / (2.0 * (R * R + z * z) ** 1.5)


def skin_depth(f, rho, mu_r=1.0):
    """Skin depth [m] delta = sqrt(rho / (pi f mu0 mu_r)) at frequency f [Hz]."""
    return np.sqrt(rho / (np.pi * f * MU0 * mu_r))


def dowell_factor(delta_ratio, n_layers):
    """Dowell AC/DC resistance factor of an m-layer winding [dowell1966].

    F_R = D [ (sinh 2D + sin 2D)/(cosh 2D - cos 2D) + 2(m^2-1)/3 (sinh D - sin D)/(cosh D + cos D) ],
    D = layer thickness / skin depth (round wire: equivalent foil, see `RoundWireCoil.skin_report`).
    Low-frequency limit 1 + (5m^2 - 1) D^4 / 45. (The old `ac_resistance.py:138` used half the
    proximity term; this is the standard form.)
    """
    scalar = np.ndim(delta_ratio) == 0
    D = np.atleast_1d(np.asarray(delta_ratio, float))
    m = float(n_layers)
    out = np.empty_like(D)
    lo, hi = D < 1e-2, D > 20.0
    mid = ~lo & ~hi
    Dm = D[mid]
    s1 = (np.sinh(2 * Dm) + np.sin(2 * Dm)) / (np.cosh(2 * Dm) - np.cos(2 * Dm))
    s2 = (np.sinh(Dm) - np.sin(Dm)) / (np.cosh(Dm) + np.cos(Dm))
    out[mid] = Dm * (s1 + 2.0 * (m * m - 1.0) / 3.0 * s2)
    out[lo] = 1.0 + (5.0 * m * m - 1.0) * D[lo] ** 4 / 45.0
    out[hi] = D[hi] * (1.0 + 2.0 * (m * m - 1.0) / 3.0)
    return float(out[0]) if scalar else out


def strip_sheet_field(points, P1, P2, z_lo, z_hi, current=1.0):
    """H [A/m] at `points` (n,3) of straight current sheets ("strips") in the body frame.

    Strip i spans the segment P1[i] -> P2[i] (only x, y are used) and z in [z_lo[i], z_hi[i]];
    `current[i]` [A] flows along P1 -> P2, uniformly spread over the strip height. Exact integral of
    the finite straight-wire Biot-Savart field over the height (old validated
    `magnet_pulse_sim.py:221-259`, generalised to any in-plane direction):
        H_z  = I/(4 pi w) sum (+-) atan(u Z / (d R)),  H_n = I/(4 pi w) sum (+-) asinh(u / sqrt(d^2+Z^2)),
    with u the along-strip and d the signed in-plane normal distance, Z = z - z'. Singular on the
    strip itself (one-sided value returned for points exactly in its plane).
    """
    pts = np.atleast_2d(np.asarray(points, float))
    P1 = np.atleast_2d(np.asarray(P1, float))[:, :2]
    P2 = np.atleast_2d(np.asarray(P2, float))[:, :2]
    n_s = len(P1)
    z_lo = np.broadcast_to(np.asarray(z_lo, float), (n_s,))
    z_hi = np.broadcast_to(np.asarray(z_hi, float), (n_s,))
    cur = np.broadcast_to(np.asarray(current, float), (n_s,))
    t = P2 - P1
    L = np.linalg.norm(t, axis=1)
    keep = L > 0
    P1, L, t, z_lo, z_hi, cur = P1[keep], L[keep], t[keep], z_lo[keep], z_hi[keep], cur[keep]
    tang = t / L[:, None]
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # z_hat x t_hat
    coef = cur / (4.0 * np.pi * (z_hi - z_lo))
    scale = max(float(np.max(np.abs(pts))) if pts.size else 1.0, float(np.max(L)) if L.size else 1.0)
    tiny = 1e-13 * scale
    H = np.zeros((len(pts), 3))
    step = max(1, _CHUNK // max(len(pts), 1))
    for k in range(0, len(P1), step):
        sl = slice(k, k + step)
        r = pts[:, None, :2] - P1[None, sl]
        dist = np.sum(r * normal[None, sl], axis=2)
        dist = np.where(np.abs(dist) < tiny, tiny, dist)
        u1 = np.sum(r * tang[None, sl], axis=2)
        u2 = u1 - L[None, sl]
        Zh = pts[:, None, 2] - z_lo[None, sl]
        Zl = pts[:, None, 2] - z_hi[None, sl]
        axial = np.zeros_like(dist)
        trans = np.zeros_like(dist)
        for uu, su in ((u1, 1.0), (u2, -1.0)):
            for ZZ, sz in ((Zh, 1.0), (Zl, -1.0)):
                rad = np.sqrt(uu * uu + dist * dist + ZZ * ZZ)
                axial += su * sz * np.arctan(uu * ZZ / (dist * rad))
                trans += su * sz * np.arcsinh(uu / np.maximum(np.hypot(dist, ZZ), tiny))
        H[:, 2] += axial @ coef[sl]
        H[:, :2] += (trans * coef[None, sl]) @ normal[sl]
    return H


# =====================================================================================================
# Outlines (closed planar loops in the body xy plane)
# =====================================================================================================
def _outline_quadrant(hx, hy, rho, n_arc):
    """First-quadrant vertices (counter-clockwise) of a rounded rectangle.

    Straight half-lengths hx, hy, corner radius rho (rho = 0: sharp rectangle with half-sides hx, hy;
    hx = hy = 0: circle). Path (hx+rho, 0) -> (hx+rho, hy) -> arc about (hx, hy) -> (hx, hy+rho)
    -> (0, hy+rho). Arc vertices lie on the arc (inscribed chords).
    """
    v = [(hx + rho, 0.0), (hx + rho, hy)]
    if rho > 0:
        th = np.linspace(0.0, np.pi / 2, n_arc + 1)[1:]
        v += [(hx + rho * np.cos(a), hy + rho * np.sin(a)) for a in th]
    v.append((0.0, hy + rho))
    return np.array(v)


def _dedupe(v):
    keep = np.ones(len(v), bool)
    keep[1:] = np.linalg.norm(np.diff(v, axis=0), axis=1) > 1e-15
    return v[keep]


def outline_polygon(hx, hy, rho, n_arc):
    """Closed polygon (m,2) of a rounded rectangle, counter-clockwise, side midpoints as vertices."""
    q = _outline_quadrant(hx, hy, rho, n_arc)
    q2 = (q * [-1, 1])[::-1]
    q3 = q * [-1, -1]
    q4 = (q * [1, -1])[::-1]
    v = _dedupe(np.vstack([q, q2[1:], q3[1:], q4[1:]]))
    if np.linalg.norm(v[-1] - v[0]) < 1e-15:
        v = v[:-1]
    return v


def outline_segments(hx, hy, rho, n_arc, quadrant=False):
    """Segments (P1, P2) as (m,3) arrays at z = 0; `quadrant=True` returns only the x>=0, y>=0 part."""
    if quadrant:
        v = _dedupe(_outline_quadrant(hx, hy, rho, n_arc))
        a, b = v[:-1], v[1:]
    else:
        v = outline_polygon(hx, hy, rho, n_arc)
        a, b = v, np.roll(v, -1, axis=0)
    z = np.zeros((len(a), 1))
    return np.hstack([a, z]), np.hstack([b, z])


def outline_perimeter(hx, hy, rho):
    """Exact perimeter [m] of the rounded rectangle (not of its polygon): 4 hx + 4 hy + 2 pi rho."""
    return 4.0 * hx + 4.0 * hy + 2.0 * np.pi * rho


def outline_area(hx, hy, rho):
    """Exact enclosed area [m^2]."""
    return 4.0 * hx * hy + 4.0 * rho * (hx + hy) + np.pi * rho * rho


# =====================================================================================================
# Filament sets (global frame) and circuit matrices
# =====================================================================================================
@dataclass
class FilamentSet:
    """Closed filament loops in the global frame and the circuits they belong to.

    loops[i]   : (m_i, 3) polygon vertices [m], implicitly closed, counter-clockwise about normal[i]
    seg_gmd[i] : (m_i,) self GMD [m] of each segment's cross-section (used only for self terms)
    weight[i]  : loop current per unit circuit current (e.g. 1/n_sub for sub-filaments)
    owner[i]   : circuit index of the loop
    normal     : (n,3) unit loop normals
    circle_center/circle_radius : exact circles (all loops) for the coaxial Maxwell fast path, or None
    """

    loops: list
    seg_gmd: list
    weight: np.ndarray
    owner: np.ndarray
    normal: np.ndarray
    circle_center: np.ndarray | None = None
    circle_radius: np.ndarray | None = None

    @property
    def n_loops(self) -> int:
        return len(self.loops)

    @property
    def n_circuits(self) -> int:
        return int(np.max(self.owner)) + 1 if len(self.owner) else 0

    def segments(self):
        """(P1, P2, loop_index) of all straight segments."""
        P1 = np.vstack(self.loops)
        P2 = np.vstack([np.roll(v, -1, axis=0) for v in self.loops])
        idx = np.repeat(np.arange(self.n_loops), [len(v) for v in self.loops])
        return P1, P2, idx

    def weight_matrix(self):
        """W (n_circuits, n_loops) with W[owner_i, i] = weight_i."""
        W = np.zeros((self.n_circuits, self.n_loops))
        W[self.owner, np.arange(self.n_loops)] = self.weight
        return W


def _coaxial(A: FilamentSet, B: FilamentSet, tol=1e-12):
    """If every loop of A and B is a circle on one common axis, return (zA, zB, sA, sB) else None.

    z = axial position along the common axis e; s = +-1 orientation of each loop's normal vs e.
    """
    if A.circle_center is None or B.circle_center is None:
        return None
    C = np.vstack([A.circle_center, B.circle_center])
    N = np.vstack([A.normal, B.normal])
    e = N[0]
    if np.any(np.linalg.norm(np.cross(N, e), axis=1) > tol):
        return None
    rel = C - C[0]
    if np.any(np.linalg.norm(np.cross(rel, e), axis=1) > tol * max(1.0, float(np.max(np.abs(C))))):
        return None
    z = rel @ e
    s = np.sign(N @ e)
    nA = A.n_loops
    return z[:nA], z[nA:], s[:nA], s[nA:]


def loop_mutual_matrix(A: FilamentSet, B: FilamentSet) -> np.ndarray:
    """Filament mutual inductances [H] between every loop of A and every loop of B (no self terms).

    Coaxial circles use `maxwell_mutual`; anything else sums `segment_mutual` over segment pairs.
    """
    co = _coaxial(A, B)
    if co is not None:
        zA, zB, sA, sB = co
        M = maxwell_mutual(A.circle_radius[:, None], B.circle_radius[None, :], zA[:, None] - zB[None, :])
        return M * sA[:, None] * sB[None, :]
    P1a, P2a, ia = A.segments()
    P1b, P2b, ib = B.segments()
    startb = np.r_[0, np.cumsum([len(v) for v in B.loops])[:-1]]
    out = np.zeros((A.n_loops, B.n_loops))
    rows = max(1, _CHUNK // max(len(P1b), 1))
    for k in range(0, len(P1a), rows):
        sl = slice(k, k + rows)
        m = segment_mutual(P1a[sl, None], P2a[sl, None], P1b[None], P2b[None])
        m = np.add.reduceat(m, startb, axis=1)
        np.add.at(out, ia[sl], m)
    return out


def _polygon_self(P, gmd, normal):
    """Self inductance [H] of one polygon loop: segment pairs with the second copy displaced along the
    loop normal by sqrt(g_a g_b) (GMD regularisation, exact partial self terms for straight sides)."""
    P1, P2 = P, np.roll(P, -1, axis=0)
    shift = np.sqrt(gmd[:, None] * gmd[None, :])[..., None] * normal
    return float(np.sum(segment_mutual(P1[:, None], P2[:, None], P1[None] + shift, P2[None] + shift)))


def loop_inductance_matrix(A: FilamentSet) -> np.ndarray:
    """Full loop inductance matrix [H] of a filament set: mutuals plus GMD self terms on the diagonal."""
    M = loop_mutual_matrix(A, A)
    co = _coaxial(A, A)
    if co is not None:
        g = np.array([gg[0] for gg in A.seg_gmd])
        diag = maxwell_mutual(A.circle_radius, A.circle_radius, g)
    else:
        diag = np.array([_polygon_self(P, g, n) for P, g, n in zip(A.loops, A.seg_gmd, A.normal)])
    np.fill_diagonal(M, diag)
    return M


def circuit_mutual(A: FilamentSet, B: FilamentSet) -> np.ndarray:
    """Circuit-level mutual inductance matrix [H] (A.n_circuits, B.n_circuits) between distinct sets."""
    return A.weight_matrix() @ loop_mutual_matrix(A, B) @ B.weight_matrix().T


def circuit_inductance(A: FilamentSet) -> np.ndarray:
    """Circuit-level inductance matrix [H] (n_circuits, n_circuits) of one set, self terms included."""
    W = A.weight_matrix()
    L = W @ loop_inductance_matrix(A) @ W.T
    return 0.5 * (L + L.T)


# =====================================================================================================
# Coils
# =====================================================================================================
def _rrect(hx, hy, rho):
    """Outline tuple; a rounded rectangle without straight parts is an exact circle (Maxwell path)."""
    if hx <= 1e-15 and hy <= 1e-15:
        return ("circle", rho)
    return ("rrect", hx, hy, rho)


def _as_rotation(rotation):
    if rotation is None:
        return Rotation.identity()
    if isinstance(rotation, Rotation):
        return rotation
    return Rotation.from_matrix(np.asarray(rotation, float))


class _Coil:
    """Shared machinery. Subclasses define the outline family, filaments and conductor data."""

    n_arc = 16  # chords per quarter arc for inductance
    n_arc_field = 32  # chords per quarter arc for field evaluation
    n_poly_circle = 128  # polygon sides when a circular turn meets a non-coaxial partner

    def __init__(self, position=(0.0, 0.0, 0.0), rotation=None):
        self.position = np.asarray(position, float).reshape(3)
        self.rotation = _as_rotation(rotation)
        self._L = None

    # ---- subclass interface -----------------------------------------------------------------------
    def _outline(self, d):
        """('circle', R) or ('rrect', hx, hy, rho) for a filament at radial offset d from the bore."""
        raise NotImplementedError

    def _filaments_body(self):
        """(d, z, weight, gmd) arrays: radial offset from bore [m], axial position [m], current share, GMD [m]."""
        raise NotImplementedError

    def _turn_offsets(self):
        """(n_turns,) radial offset [m] of each turn's conductor centre from the bore."""
        raise NotImplementedError

    conductor_area: float = 0.0
    lead_length: float = 0.0

    # ---- geometry helpers ---------------------------------------------------------------------------
    @property
    def is_circular(self) -> bool:
        return self._outline(0.0)[0] == "circle"

    def _perimeter(self, d):
        o = self._outline(d)
        return 2.0 * np.pi * o[1] if o[0] == "circle" else outline_perimeter(*o[1:])

    def to_body(self, points):
        return self.rotation.inv().apply(np.atleast_2d(np.asarray(points, float)) - self.position)

    def conductor_length(self) -> float:
        """Total conductor length [m]: closed-turn perimeters at conductor centres + leads."""
        return float(sum(self._perimeter(d) for d in self._turn_offsets())) + self.lead_length

    # ---- electrical / thermal -----------------------------------------------------------------------
    def resistance(self, T=T_REF):
        """DC resistance [ohm] at temperature T [K]: rho_Cu(T) * length / area (leads included)."""
        return copper_resistivity(T) * self.conductor_length() / self.conductor_area

    def mass(self) -> float:
        """Copper mass [kg] (leads included), density DENSITY_CU."""
        return DENSITY_CU * self.conductor_length() * self.conductor_area

    def heat_capacity(self) -> float:
        """Adiabatic heat capacity of the copper [J/K] = mass * CP_CU (insulation and former ignored)."""
        return self.mass() * CP_CU

    def lumped(self, T0=T_REF) -> dict:
        """Coil parameters for `circuit.netlist.build_deck`: L [H], R0 [ohm] at T_ref, alpha [1/K], T_ref,
        T0 [K] initial temperature, C_th [J/K]."""
        return {
            "L": self.inductance(),
            "R0": float(self.resistance(T_REF)),
            "alpha": ALPHA_CU,
            "T_ref": T_REF,
            "T0": float(T0),
            "C_th": self.heat_capacity(),
        }

    # ---- inductance ---------------------------------------------------------------------------------
    def inductance(self) -> float:
        """Self inductance [H] of the winding (filament sum; see module docstring). Cached."""
        if self._L is None:
            d, z, w, g = self._filaments_body()
            if self.is_circular:
                R = np.array([self._outline(x)[1] for x in d])
                M = maxwell_mutual(R[:, None], R[None, :], z[:, None] - z[None, :])
                np.fill_diagonal(M, maxwell_mutual(R, R, g))
                self._L = float(w @ M @ w)
            else:
                self._L = self._polygon_inductance(d, z, w, g)
        return self._L

    def _polygon_inductance(self, d, z, w, g):
        """Filament sum for coaxial rounded-rectangle loops using T[shape_a, shape_b, |dz|] caching and
        the two mirror symmetries of the outline (M(A,B) = 4 sum_{a in quadrant(A)} sum_{b in B} m_ab)."""
        dk = np.round(d / _KEY_RES).astype(np.int64)
        shapes, sid = np.unique(dk, return_inverse=True)
        quad = [outline_segments(*self._outline(s * _KEY_RES)[1:], self.n_arc, quadrant=True) for s in shapes]
        full = [outline_segments(*self._outline(s * _KEY_RES)[1:], self.n_arc) for s in shapes]
        i, j = np.triu_indices(len(d))
        dz = np.abs(z[i] - z[j])
        same = i == j
        dz[same] = g[i[same]]
        if np.any((dz < _KEY_RES) & (sid[i] == sid[j])):
            raise ValueError("two filaments coincide (same outline and axial position)")
        a, b = np.minimum(sid[i], sid[j]), np.maximum(sid[i], sid[j])
        keys = np.stack([a, b, np.round(dz / _KEY_RES).astype(np.int64)], axis=1)
        ukeys, inv = np.unique(keys, axis=0, return_inverse=True)
        T = np.zeros(len(ukeys))
        zhat = np.array([0.0, 0.0, 1.0])
        for (sa, sb) in np.unique(ukeys[:, :2], axis=0):
            sel = np.where((ukeys[:, 0] == sa) & (ukeys[:, 1] == sb))[0]
            dzs = ukeys[sel, 2] * _KEY_RES
            qa1, qa2 = quad[sa]
            fb1, fb2 = full[sb]
            shift = dzs[None, None, :, None] * zhat
            m = segment_mutual(
                qa1[:, None, None], qa2[:, None, None], fb1[None, :, None] + shift, fb2[None, :, None] + shift
            )
            T[sel] = 4.0 * m.sum(axis=(0, 1))
        pair = T[inv.ravel()]
        fac = np.where(same, 1.0, 2.0)
        return float(np.sum(fac * w[i] * w[j] * pair))

    # ---- filaments in the global frame ------------------------------------------------------------
    def filaments(self, owner=0, n_arc=None) -> FilamentSet:
        """Global-frame filament set of this coil (one circuit, index `owner`)."""
        d, z, w, g = self._filaments_body()
        R = self.rotation.as_matrix()
        nrm = R @ np.array([0.0, 0.0, 1.0])
        loops, gm, cen, rad = [], [], [], []
        n_arc = self.n_arc if n_arc is None else n_arc
        for dd, zz, gg in zip(d, z, g):
            o = self._outline(dd)
            if o[0] == "circle":
                v = outline_polygon(0.0, 0.0, o[1], self.n_poly_circle // 4)
                rad.append(o[1])
            else:
                v = outline_polygon(*o[1:], n_arc)
            vb = np.hstack([v, np.full((len(v), 1), zz)])
            loops.append(vb @ R.T + self.position)
            gm.append(np.full(len(v), gg))
            cen.append(R @ np.array([0.0, 0.0, zz]) + self.position)
        circ = self.is_circular
        return FilamentSet(
            loops=loops,
            seg_gmd=gm,
            weight=np.asarray(w, float),
            owner=np.full(len(d), owner, int),
            normal=np.tile(nrm, (len(d), 1)),
            circle_center=np.array(cen) if circ else None,
            circle_radius=np.array(rad) if circ else None,
        )

    def mutual(self, other):
        """Mutual inductance [H] with another coil (float) or with each circuit of a FilamentSet / object
        with a `.filaments()` method (e.g. `EddyLoops`), returned as (n_circuits,)."""
        fs = other if isinstance(other, FilamentSet) else other.filaments()
        M = circuit_mutual(self.filaments(), fs)[0]
        return float(M[0]) if isinstance(other, _Coil) else M

    # ---- field ---------------------------------------------------------------------------------------
    def field_per_amp(self, points) -> np.ndarray:
        """H [A/m per A of coil current] at global `points` (n,3), returned in the global frame."""
        pb = self.to_body(points)
        Hb = self._field_body(pb)
        return self.rotation.apply(Hb) if len(Hb) else Hb

    def _field_body(self, pb):
        raise NotImplementedError


def _magpy_sum(fun, pts, n_src, field_kind="H", **src):
    """Sum a magpylib.func field ('H' or 'B') over n_src sources at every point, in memory-bounded chunks."""
    H = np.zeros((len(pts), 3))
    step = max(1, _CHUNK // max(n_src, 1))
    for k in range(0, len(pts), step):
        p = pts[k : k + step]
        obs = np.repeat(p, n_src, axis=0)
        kw = {key: np.tile(val, (len(p),) + (1,) * (np.ndim(val) - 1)) for key, val in src.items()}
        h = fun(field=field_kind, observers=obs, squeeze=False, **kw)
        H[k : k + step] = h.reshape(len(p), n_src, 3).sum(axis=1)
    return H


class RoundWireCoil(_Coil):
    """Layer-wound coil of round enamelled wire on a circular or rectangular former.

    Parameters (SI)
    ---------------
    former : "circular" (bore_radius) or "rectangular" (bore_a x bore_b along body x, y, corner_radius)
    winding_length : axial length of a layer [m]; default turns_per_layer * d_ins. Turns are spaced at
        pitch winding_length / turns_per_layer, centred on z = 0, aligned layer to layer.
    d_bare, d_ins : copper and insulated wire diameters [m]
    turns_per_layer, n_layers : winding counts
    former_build : radial distance from the bore surface to the inner surface of layer 1 [m]
    layer_insulation : extra radial gap between layers [m]
    lead_length : total length of both leads [m] (resistance and mass only)
    position, rotation : pose (body z = winding axis)

    Layer k (0-based) wire centres sit at offset d_k = former_build + d_ins/2 + k (d_ins + layer_insulation)
    from the bore; on a rectangular former a turn is the exact offset curve (corner radius
    corner_radius + d_k).
    """

    def __init__(
        self,
        former="rectangular",
        *,
        bore_radius=None,
        bore_a=None,
        bore_b=None,
        corner_radius=0.0,
        winding_length=None,
        d_bare,
        d_ins,
        turns_per_layer,
        n_layers,
        former_build=0.0,
        layer_insulation=0.0,
        lead_length=0.0,
        position=(0.0, 0.0, 0.0),
        rotation=None,
    ):
        super().__init__(position, rotation)
        if former not in ("circular", "rectangular"):
            raise ValueError(f"former must be 'circular' or 'rectangular', got {former!r}")
        self.former = former
        if former == "circular":
            if bore_radius is None or bore_radius <= 0:
                raise ValueError("circular former needs bore_radius > 0")
        else:
            if bore_a is None or bore_b is None or bore_a <= 0 or bore_b <= 0:
                raise ValueError("rectangular former needs bore_a, bore_b > 0")
            if not 0.0 <= corner_radius <= 0.5 * min(bore_a, bore_b):
                raise ValueError("corner_radius must be in [0, min(bore_a, bore_b)/2]")
        if not 0 < d_bare <= d_ins:
            raise ValueError("need 0 < d_bare <= d_ins")
        if turns_per_layer < 1 or n_layers < 1:
            raise ValueError("need turns_per_layer >= 1 and n_layers >= 1")
        self.bore_radius, self.bore_a, self.bore_b = bore_radius, bore_a, bore_b
        self.corner_radius = float(corner_radius)
        self.d_bare, self.d_ins = float(d_bare), float(d_ins)
        self.turns_per_layer, self.n_layers = int(turns_per_layer), int(n_layers)
        self.winding_length = float(turns_per_layer * d_ins if winding_length is None else winding_length)
        if self.winding_length < turns_per_layer * d_ins * (1 - 1e-9):
            raise ValueError("winding_length < turns_per_layer * d_ins: turns would overlap")
        self.former_build, self.layer_insulation = float(former_build), float(layer_insulation)
        self.lead_length = float(lead_length)
        self.conductor_area = np.pi * self.d_bare**2 / 4.0

    @property
    def n_turns(self) -> int:
        return self.turns_per_layer * self.n_layers

    @property
    def pitch(self) -> float:
        """Axial turn pitch [m]."""
        return self.winding_length / self.turns_per_layer

    def _outline(self, d):
        if self.former == "circular":
            return ("circle", self.bore_radius + d)
        rc = self.corner_radius
        return _rrect(0.5 * self.bore_a - rc, 0.5 * self.bore_b - rc, rc + d)

    def _layer_offsets(self):
        return self.former_build + 0.5 * self.d_ins + np.arange(self.n_layers) * (self.d_ins + self.layer_insulation)

    def _turn_offsets(self):
        return np.repeat(self._layer_offsets(), self.turns_per_layer)

    def _turn_z(self):
        return -0.5 * self.winding_length + (np.arange(self.turns_per_layer) + 0.5) * self.pitch

    def _filaments_body(self):
        d = self._turn_offsets()
        z = np.tile(self._turn_z(), self.n_layers)
        n = len(d)
        return d, z, np.ones(n), np.full(n, float(gmd_round(0.5 * self.d_bare)))

    def outer_dimensions(self):
        """Outer extent of the winding: ('circular', outer radius) or (outer a, outer b) [m]."""
        t = self._layer_offsets()[-1] + 0.5 * self.d_ins
        if self.former == "circular":
            return self.bore_radius + t
        return self.bore_a + 2 * t, self.bore_b + 2 * t

    def geometry_report(self) -> dict:
        """Lengths and the bounds of the closed-ring approximation."""
        P = np.array([self._perimeter(d) for d in self._layer_offsets()])
        helix = np.sqrt(1.0 + (self.pitch / P) ** 2) - 1.0
        return {
            "n_turns": self.n_turns,
            "pitch_m": self.pitch,
            "conductor_length_m": self.conductor_length(),
            "turn_perimeter_inner_m": float(P[0]),
            "turn_perimeter_outer_m": float(P[-1]),
            "helix_length_rel_error_max": float(helix.max()),
            # one layer's net axial current ~ a straight wire of the winding length (partial L, Grover)
            "axial_current_partial_L_H": float(
                MU0 / (2 * np.pi) * self.winding_length * (np.log(2 * self.winding_length / (0.5 * self.d_bare)) - 0.75)
            ),
        }

    def _field_body(self, pb):
        d, z, w, _ = self._filaments_body()
        if self.is_circular:
            R = np.array([self._outline(x)[1] for x in d])
            pos = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
            return _magpy_sum(circle_field, pb, len(d), diameters=2 * R, currents=w, positions=pos)
        starts, ends, cur = [], [], []
        for dd, zz, ww in zip(d, z, w):
            a, b = outline_segments(*self._outline(dd)[1:], self.n_arc_field)
            a[:, 2] = b[:, 2] = zz
            starts.append(a)
            ends.append(b)
            cur.append(np.full(len(a), ww))
        S, E, C = np.vstack(starts), np.vstack(ends), np.concatenate(cur)
        return _magpy_sum(polyline_field, pb, len(S), segments_start=S, segments_end=E, currents=C)

    def skin_report(self, f, T=T_REF) -> dict:
        """Skin depth and Dowell factor at frequency f [Hz] [dowell1966].

        Round wire -> equivalent square foil of side h = d sqrt(pi)/2 with porosity eta = h / pitch:
        D = (h/delta) sqrt(eta); m = n_layers. Warns when F_R > 1.05.
        """
        rho = float(copper_resistivity(T))
        delta = float(skin_depth(f, rho))
        h = self.d_bare * np.sqrt(np.pi) / 2.0
        eta = h / self.pitch
        D = h / delta * np.sqrt(eta)
        F = float(dowell_factor(D, self.n_layers))
        if F > 1.05:
            warnings.warn(f"Dowell factor {F:.3f} > 1.05 at {f:.4g} Hz: DC resistance underestimates loss", stacklevel=2)
        return {"f_Hz": f, "skin_depth_m": delta, "d_over_delta": self.d_bare / delta, "dowell_D": D,
                "n_layers": self.n_layers, "dowell_FR": F}


class FoilCoil(_Coil):
    """Spiral foil coil on a rectangular former (one turn per layer, concentric turns).

    Parameters (SI)
    ---------------
    bore_a, bore_b : bore dimensions along body x, y [m]; corner_radius [m] of the bore
    strip_width : foil width along the axis [m]; foil_thickness [m]; insulation_thickness [m] between turns
    turns : number of turns; former_build : radial gap bore -> first foil surface [m]
    corners : "offset" (turn k is the exact offset curve, corner radius corner_radius + d_k) or
        "sharp" (every turn a sharp rectangle of half-sides bore/2 + d_k, the old
        `magnet_pulse_sim.py` convention; needs corner_radius = 0)
    lead_length : total lead length [m] (resistance and mass only)
    n_sub : (n_t, n_w) sub-cells across thickness and width for the inductance; default near-square cells
    """

    n_t_field = 4  # Gauss points across the foil thickness for field evaluation

    def __init__(
        self,
        *,
        bore_a,
        bore_b,
        corner_radius=0.0,
        strip_width,
        foil_thickness,
        insulation_thickness,
        turns,
        former_build=0.0,
        corners="offset",
        lead_length=0.0,
        n_sub=None,
        position=(0.0, 0.0, 0.0),
        rotation=None,
    ):
        super().__init__(position, rotation)
        if corners not in ("offset", "sharp"):
            raise ValueError("corners must be 'offset' or 'sharp'")
        if corners == "sharp" and corner_radius != 0:
            raise ValueError("corners='sharp' needs corner_radius = 0")
        if not 0.0 <= corner_radius <= 0.5 * min(bore_a, bore_b):
            raise ValueError("corner_radius must be in [0, min(bore_a, bore_b)/2]")
        if min(bore_a, bore_b, strip_width, foil_thickness) <= 0 or insulation_thickness < 0 or turns < 1:
            raise ValueError("invalid foil coil dimensions")
        self.bore_a, self.bore_b, self.corner_radius = float(bore_a), float(bore_b), float(corner_radius)
        self.strip_width, self.foil_thickness = float(strip_width), float(foil_thickness)
        self.insulation_thickness, self.turns = float(insulation_thickness), int(turns)
        self.former_build, self.corners, self.lead_length = float(former_build), corners, float(lead_length)
        if n_sub is None:
            n_sub = (1, int(min(64, max(1, round(strip_width / foil_thickness)))))
        self.n_sub = (int(n_sub[0]), int(n_sub[1]))
        self.conductor_area = self.strip_width * self.foil_thickness

    @property
    def n_turns(self) -> int:
        return self.turns

    def _outline(self, d):
        if self.corners == "sharp":
            return ("rrect", 0.5 * self.bore_a + d, 0.5 * self.bore_b + d, 0.0)
        rc = self.corner_radius
        return _rrect(0.5 * self.bore_a - rc, 0.5 * self.bore_b - rc, rc + d)

    def _turn_offsets(self):
        t = self.foil_thickness
        return self.former_build + 0.5 * t + np.arange(self.turns) * (t + self.insulation_thickness)

    def _filaments_body(self):
        nt, nw = self.n_sub
        t, w = self.foil_thickness, self.strip_width
        dt = (np.arange(nt) + 0.5) / nt * t - 0.5 * t
        zw = (np.arange(nw) + 0.5) / nw * w - 0.5 * w
        D = (self._turn_offsets()[:, None, None] + dt[None, :, None] + 0 * zw[None, None, :]).ravel()
        Z = np.broadcast_to(zw[None, None, :], (self.turns, nt, nw)).ravel().copy()
        n = len(D)
        g = float(gmd_rectangle(t / nt, w / nw))
        return D, Z, np.full(n, 1.0 / (nt * nw)), np.full(n, g)

    def geometry_report(self) -> dict:
        P = np.array([self._perimeter(d) for d in self._turn_offsets()])
        pitch = self.foil_thickness + self.insulation_thickness
        return {
            "n_turns": self.turns,
            "conductor_length_m": self.conductor_length(),
            "turn_perimeter_inner_m": float(P[0]),
            "turn_perimeter_outer_m": float(P[-1]),
            # spiral vs concentric rings: radial pitch per turn, relative length error ~ (pitch/P)^2 / 2
            "spiral_length_rel_error_max": float(0.5 * (pitch / P[0]) ** 2),
        }

    def _field_body(self, pb):
        x, wq = leggauss(self.n_t_field)
        xi, wq = 0.5 * x * self.foil_thickness, 0.5 * wq
        if self.is_circular:  # exact circular current sheets = side walls of polarised cylinders
            R = np.array([self._outline(d + off)[1] for d in self._turn_offsets() for off in xi])
            cur = np.tile(wq, self.turns)
            w = self.strip_width
            pol = np.zeros((len(R), 3))
            pol[:, 2] = MU0 * cur / w
            B = _magpy_sum(cylinder_field, pb, len(R), dimensions=np.column_stack([2 * R, np.full(len(R), w)]),
                           polarizations=pol, field_kind="B")
            return B / MU0
        P1s, P2s, cur = [], [], []
        for d in self._turn_offsets():
            for off, ww in zip(xi, wq):
                a, b = outline_segments(*self._outline(d + off)[1:], self.n_arc_field)
                P1s.append(a)
                P2s.append(b)
                cur.append(np.full(len(a), ww))
        return strip_sheet_field(
            pb, np.vstack(P1s), np.vstack(P2s), -0.5 * self.strip_width, 0.5 * self.strip_width, np.concatenate(cur)
        )

    def skin_report(self, f, T=T_REF) -> dict:
        """Skin depth and Dowell factor at f [Hz] for the foil: D = t/delta, m = turns [dowell1966]."""
        rho = float(copper_resistivity(T))
        delta = float(skin_depth(f, rho))
        D = self.foil_thickness / delta
        F = float(dowell_factor(D, self.turns))
        if F > 1.05:
            warnings.warn(f"Dowell factor {F:.3f} > 1.05 at {f:.4g} Hz: DC resistance underestimates loss", stacklevel=2)
        return {"f_Hz": f, "skin_depth_m": delta, "t_over_delta": D, "dowell_D": D, "n_layers": self.turns,
                "dowell_FR": F}


# =====================================================================================================
# Reference coils (SPEC "Reference cases")
# =====================================================================================================
def reference_coil_r1(corners="sharp", **kw) -> FoilCoil:
    """R1: 22-turn Cu foil 0.2 x 5 mm, 0.05 mm kapton between turns, 20 x 20 mm bore, no leads.

    Source: `$PARENT/simulations_old/Switch_sim/magnet-simulation/magnet_pulse_sim.py:157-191`
    (turns 22, copper_thickness 0.2e-3, strip_width 5e-3, kapton_thickness 0.05e-3, inner_clearance 0,
    magnet_side 20e-3). # ASSUMPTION: corners="sharp" reproduces the old validated model; the real
    corner radii are unknown (corners="offset" bounds the alternative). External wiring (old 5 mOhm)
    is a circuit element, not part of the coil.
    """
    p = dict(bore_a=20e-3, bore_b=20e-3, corner_radius=0.0, strip_width=5e-3, foil_thickness=0.2e-3,
             insulation_thickness=0.05e-3, turns=22, former_build=0.0, corners=corners, lead_length=0.0)
    p.update(kw)
    return FoilCoil(**p)


def reference_coil_r2(n_layers=20, **kw) -> RoundWireCoil:
    """R2: 1.0 mm Cu wire (1.07 mm OD, IEC 60317 grade 2), 5 turns/layer, `n_layers` layers, wound
    directly on the 20 x 10 mm magnet (sharp bore, offset-curve turns), 2 x 30 mm leads.

    Source: `$PARENT/simulations_old/Switch_sim/magnet-flip-sim/src/params.py` (WireSpec, MagnetSpec,
    TURNS_PER_LAYER) and `geometry.py` (layer stand-off (k - 1/2) d_ins, offset-curve perimeter).
    """
    p = dict(former="rectangular", bore_a=20e-3, bore_b=10e-3, corner_radius=0.0, d_bare=1.0e-3, d_ins=1.07e-3,
             turns_per_layer=5, n_layers=n_layers, winding_length=5 * 1.07e-3, former_build=0.0,
             lead_length=2 * 0.030)
    p.update(kw)
    return RoundWireCoil(**p)


def reference_coil_r3(**kw) -> RoundWireCoil:
    """R3 (ILLUSTRATIVE design, not a built coil): solenoid on the D4.75 x 12.5 mm LNG37 rod.

    # ASSUMPTION: 0.25 mm wall former tube with 0.05 mm radial clearance (bore radius 2.425 mm),
    # 0.50 mm Cu wire, 0.55 mm OD (IEC 60317 grade 2 nominal), 21 turns/layer over 11.55 mm, 6 layers
    # (126 turns), 2 x 50 mm leads. Chosen so ~20 A gives ~200 kA/m (about 4 x Hc of Alnico 5/LNG37).
    """
    p = dict(former="circular", bore_radius=4.75e-3 / 2 + 0.05e-3, d_bare=0.50e-3, d_ins=0.55e-3,
             turns_per_layer=21, n_layers=6, winding_length=21 * 0.55e-3, former_build=0.25e-3,
             lead_length=2 * 0.050)
    p.update(kw)
    return RoundWireCoil(**p)
