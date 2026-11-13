"""Closed-form magnetostatic references (task 3).

Everything here is an independent reference for the MoM / force code, tested against brute-force
quadrature or against magpylib in ``tests/test_analytic.py``. SI units throughout: lengths in m,
polarisation J = MU0*M in T, fields H in A/m, forces in N.

Contents
--------
- ``rect_pair_phi`` / ``rect_pair_integral``: corner-sum antiderivatives for two parallel uniformly
  charged rectangles (the Akoun-Yonnet integrand family [akoun1984, allag2009]), with the correct
  one-sided limits at zero normal separation.
- ``akoun_yonnet_force``: full force vector between two cuboids polarised along parallel z axes,
  arbitrary offset [akoun1984]. The zero-gap branch uses the one-sided limit
  ``uv*arctan(uv/(r w)) -> (pi/2)|uv| sign`` (the old ``magnet_force.py:214`` set it to 0, which drops
  the contact pressure term).
- Lipschitz-Hankel kernels ``force_kernel_G`` and ``field_kernel_Phi`` for coaxial equal-radius discs
  [vokoun2009], their stable difference, ``coaxial_cylinder_force`` and the ported 1-D slab benchmark
  ``CoaxialRodPair`` (from ``$PARENT/simulations_old/Force_compute/python/magnet_force.py:44-418``).
- Magnetometric demagnetising factors: ``aharoni_prism_demag`` [aharoni1998] and
  ``cylinder_demag_factor`` (Chen-Brug-Goldfarb magnetometric factor [chen1991], via Phi).
- ``dipole_force`` between two point dipoles.
"""

from __future__ import annotations

import numpy as np

from magsim.constants import MU0

__all__ = [
    "rect_pair_phi",
    "rect_pair_integral",
    "akoun_yonnet_force",
    "force_kernel_G",
    "field_kernel_Phi",
    "field_kernel_Phi_diff",
    "coaxial_disc_force",
    "coaxial_cylinder_force",
    "cylinder_demag_factor",
    "aharoni_prism_demag",
    "dipole_force",
    "CoaxialRodPair",
]


# ---------------------------------------------------------------------------------------------
# Parallel charged rectangles (Akoun-Yonnet family)
# ---------------------------------------------------------------------------------------------
def _log_r_minus(a, r, rest2):
    """ln(r - a) evaluated stably; r = sqrt(a^2 + rest2). Returns 0 where r - a == 0 exactly.

    For a > 0 the difference r - a cancels; use r - a = rest2 / (r + a). Where rest2 == 0 and a >= 0
    the logarithm is -inf, but every term that multiplies it in ``rect_pair_phi`` vanishes there
    (it carries a factor of the other two coordinates), so 0 is the correct limit of the product.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(a > 0.0, rest2 / (r + a), r - a)
        out = np.where(d > 0.0, np.log(np.where(d > 0.0, d, 1.0)), 0.0)
    return out


def rect_pair_phi(u, v, w, side=1.0, w_tol=0.0):
    """Corner antiderivative Phi(u, v, w) (3-vector, last axis) for parallel charged rectangles.

    Phi satisfies d^2/du^2 d^2/dv^2 Phi = (u, v, w) / r^3, r = |(u, v, w)| [akoun1984]:

        Phi_x = 1/2 (v^2 - w^2) ln(r - u) + u v ln(r - v) + v w atan(uv / (r w)) + 1/2 r u
        Phi_y = 1/2 (u^2 - w^2) ln(r - v) + u v ln(r - u) + u w atan(uv / (r w)) + 1/2 r v
        Phi_z = -u w ln(r - u) - v w ln(r - v) + u v atan(uv / (r w)) - r w

    ``u, v, w`` are target-minus-source coordinate differences (arrays, broadcast). Where |w| <= w_tol
    the w = 0 one-sided limit is used: atan(uv/(r w)) -> (pi/2) sign(uv) * side, where ``side`` = +1
    if the target's charge lies on the +w side of the plane (w -> 0+), -1 for w -> 0-. Only Phi_z keeps
    a finite contribution, (pi/2)|uv| side (the contact-pressure term the old code dropped).
    Units: if u, v, w are in m, Phi is in m^2.
    """
    u, v, w, side = np.broadcast_arrays(*(np.asarray(t, dtype=float) for t in (u, v, w, side)))
    zero_w = np.abs(w) <= w_tol
    w = np.where(zero_w, 0.0, w)
    u2, v2, w2 = u * u, v * v, w * w
    r = np.sqrt(u2 + v2 + w2)
    lu = _log_r_minus(u, r, v2 + w2)
    lv = _log_r_minus(v, r, u2 + w2)
    uv = u * v
    with np.errstate(divide="ignore", invalid="ignore"):
        at = np.where(zero_w, 0.5 * np.pi * np.sign(uv) * np.sign(side), np.arctan(uv / (r * w)))
    at = np.where(np.isfinite(at), at, 0.0)
    px = 0.5 * (v2 - w2) * lu + uv * lv + v * w * at + 0.5 * r * u
    py = 0.5 * (u2 - w2) * lv + uv * lu + u * w * at + 0.5 * r * v
    pz = -u * w * lu - v * w * lv + uv * at - r * w
    return np.stack([px, py, pz], axis=-1)


def rect_pair_integral(t_s, t_t, s_s, s_t, w, side=1.0, w_tol=0.0):
    """Integral of (r_t - r_s)/|r_t - r_s|^3 over two parallel rectangles (exact, corner sum).

    Rectangles lie in planes normal to the common third axis; ``w`` = (target plane - source plane)
    along it. ``t_s``/``t_t`` are the target's (lo, hi) extents along the first / second in-plane axis,
    ``s_s``/``s_t`` the source's. Arrays of shape (..., 2) broadcast together; ``w``, ``side`` shape (...).
    Returns (..., 3) in the (axis1, axis2, normal) frame. The force on the target rectangle is
    sigma_t * sigma_s / (4 pi MU0) times this [akoun1984]. Units m^2 when lengths are in m.
    """
    t_s, t_t, s_s, s_t = (np.asarray(a, dtype=float) for a in (t_s, t_t, s_s, s_t))
    w = np.asarray(w, dtype=float)
    side = np.asarray(side, dtype=float)
    out = 0.0
    for a in range(2):
        for b in range(2):
            u = t_s[..., a] - s_s[..., b]
            for c in range(2):
                for d in range(2):
                    v = t_t[..., c] - s_t[..., d]
                    sgn = -1.0 if (a + b + c + d) % 2 else 1.0
                    out = out + sgn * rect_pair_phi(u, v, w, side, w_tol)
    return out


def akoun_yonnet_force(J1, J2, dims1, dims2, offset):
    """Force on cuboid 2 from cuboid 1, both uniformly polarised along +z (signed J) [akoun1984].

    Parameters
    ----------
    J1, J2 : float, polarisation of magnet 1 / 2 along +z [T] (negative = along -z).
    dims1, dims2 : full side lengths (dx, dy, dz) [m]; sides parallel to the global axes.
    offset : centre of magnet 2 minus centre of magnet 1 (dx, dy, dz) [m].

    Returns
    -------
    F : ndarray (3,), force on magnet 2 [N].

    Sum over the four pole-face pairs of ``rect_pair_integral``, i.e. the 64-term Akoun-Yonnet corner
    sum. At touching faces (w = 0 for one face pair) the one-sided limit is taken from the side where
    magnet 2 lies, which makes F continuous as the gap -> 0+. Lengths are scaled by the largest side
    before evaluation (the corner sum cancels like (size/distance)^4 far away; fine to ~100x size).
    Raises ValueError if the cuboids overlap.
    """
    d1 = np.asarray(dims1, dtype=float)
    d2 = np.asarray(dims2, dtype=float)
    off = np.asarray(offset, dtype=float)
    if np.any(d1 <= 0) or np.any(d2 <= 0):
        raise ValueError("cuboid dimensions must be positive")
    L = float(max(d1.max(), d2.max()))
    a, A = d1 / (2 * L), d2 / (2 * L)
    o = off / L
    tol = 1e-12
    if np.all(np.abs(o) < a + A - tol):
        raise ValueError("cuboids overlap")
    F = np.zeros(3)
    # target faces (magnet 2): top +J2 at o_z + C, bottom -J2 at o_z - C; side = -n_out,z
    for zt, st, side in ((o[2] + A[2], J2, -1.0), (o[2] - A[2], -J2, 1.0)):
        for zs, ss in ((a[2], J1), (-a[2], -J1)):
            I = rect_pair_integral(
                np.array([o[0] - A[0], o[0] + A[0]]),
                np.array([o[1] - A[1], o[1] + A[1]]),
                np.array([-a[0], a[0]]),
                np.array([-a[1], a[1]]),
                zt - zs,
                side,
                w_tol=tol,
            )
            F += st * ss * I
    return F * L**2 / (4.0 * np.pi * MU0)


# ---------------------------------------------------------------------------------------------
# Lipschitz-Hankel kernels for coaxial equal-radius discs [vokoun2009]
# ---------------------------------------------------------------------------------------------
_PHI_INF = 4.0 / (3.0 * np.pi)
_ASYM_FROM = 100.0
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_N_PANELS = 40


def _theta_rule(b):
    """Quadrature nodes t = pi/2 - theta in (0, pi/2] graded geometrically toward t = 0 on scale b.

    The integrands below vary on the scale t ~ b/2 near theta = pi/2 (cos theta -> 0), so the panels
    shrink geometrically from pi/2 down to t_min = 1e-3 * min(b, 1). Returns (t, w) of shape
    (..., n_nodes) for b of shape (...).
    """
    b = np.asarray(b, dtype=float)
    tmin = 1e-3 * np.clip(b, 1e-12, 1.0)
    k = np.arange(_N_PANELS + 1) / _N_PANELS
    edges = (0.5 * np.pi) * (tmin[..., None] / (0.5 * np.pi)) ** k  # descending from pi/2 to tmin
    edges = np.concatenate([edges, np.zeros(edges.shape[:-1] + (1,))], axis=-1)
    lo, hi = edges[..., 1:], edges[..., :-1]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    t = mid[..., None] + half[..., None] * _GL_X
    wt = half[..., None] * _GL_W
    return t.reshape(b.shape + (-1,)), wt.reshape(b.shape + (-1,))


def force_kernel_G(beta):
    """G(b) = int_0^inf J1(x)^2 exp(-b x) / x dx, b = |dz| / R >= 0 [vokoun2009].

    Evaluated as (1/pi) int_0^{pi/2} 4c^2 / (s + b)^2 dtheta, c = cos(theta), s = sqrt(b^2 + 4c^2)
    (the cancellation-free form of ``magnet_force.py:56-61``), with a geometrically graded
    Gauss-Legendre rule toward theta = pi/2 so small b stays accurate. G(0) = 1/2, G ~ 1/(4 b^2).
    The force between coaxial discs of radius R with charge densities s1, s2 [T] at axial distance d
    is F = pi R^2 s1 s2 G(d/R) / MU0 [N] (repulsive for s1 s2 > 0).
    """
    b = np.asarray(beta, dtype=float)
    if np.any(b < 0):
        raise ValueError("beta must be >= 0")
    t, wt = _theta_rule(b)
    c = np.sin(t)  # cos(theta) with theta = pi/2 - t
    bb = b[..., None]
    s = np.sqrt(bb * bb + 4.0 * c * c)
    val = (4.0 * c * c / (s + bb) ** 2 * wt).sum(-1) / np.pi
    return np.where(b == 0.0, 0.5, val)


def field_kernel_Phi(beta):
    """Phi(b) = int_0^inf J1(x)^2 (1 - exp(-b x)) / x^2 dx [vokoun2009].

    Cancellation-free theta form (``magnet_force.py:64-70``):
    (1/pi) int_0^{pi/2} [4c/3 - 4c^2 (2s + b) / (3 (s + b)^2)] dtheta. Phi(0) = 0,
    Phi(inf) = 4/(3 pi), Phi(b) ~ b/2 for b -> 0. Volume-average of the axial field of a charged disc.
    """
    b = np.asarray(beta, dtype=float)
    if np.any(b < 0):
        raise ValueError("beta must be >= 0")
    t, wt = _theta_rule(b)
    c = np.sin(t)
    bb = b[..., None]
    s = np.sqrt(bb * bb + 4.0 * c * c)
    psi = (4.0 / 3.0) * c - 4.0 * c * c * (2.0 * s + bb) / (3.0 * (s + bb) ** 2)
    val = (psi * wt).sum(-1) / np.pi
    return np.where(b == 0.0, 0.0, val)


def field_kernel_Phi_diff(u1, u2):
    """Phi(u2) - Phi(u1) without cancellation for large arguments (``magnet_force.py:77-98``).

    Uses Phi(b) = 4/(3 pi) - 1/(4 b) + 1/(8 b^3) + O(b^-5) when both arguments exceed 100.
    """
    u1 = np.asarray(u1, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    u1, u2 = np.broadcast_arrays(u1, u2)
    out = np.array(field_kernel_Phi(u2) - field_kernel_Phi(u1), dtype=float, ndmin=1)
    far = np.atleast_1d(np.minimum(u1, u2) > _ASYM_FROM)
    if np.any(far):
        a1 = np.atleast_1d(u1)[far]
        a2 = np.atleast_1d(u2)[far]
        out[far] = (a2 - a1) / (4.0 * a1 * a2) + 1.0 / (8.0 * a2**3) - 1.0 / (8.0 * a1**3)
    return out.reshape(u1.shape)


def coaxial_disc_force(sigma1, sigma2, radius, dz):
    """Axial force on disc 2 from coaxial disc 1 (equal radius R) [vokoun2009].

    sigma1, sigma2 : surface charge densities J.n [T]; dz = z2 - z1 [m] (dz = 0 means disc 2 is
    approached from above, the d -> 0+ limit). Returns F_z on disc 2 [N].
    """
    dz = np.asarray(dz, dtype=float)
    direction = np.where(dz >= 0.0, 1.0, -1.0)
    return direction * np.pi * radius**2 * sigma1 * sigma2 * force_kernel_G(np.abs(dz) / radius) / MU0


def coaxial_cylinder_force(J1, J2, radius, L1, L2, gap):
    """Axial force on rigid cylinder 2 from coaxial rigid cylinder 1 of equal radius [vokoun2009].

    Cylinder 1 occupies z in [-L1, 0], cylinder 2 z in [gap, gap + L2]; both polarised along +z with
    signed J1, J2 [T]. Sum of the four pole-face pairs of ``coaxial_disc_force``. Returns F_z [N]
    (negative = attraction).
    """
    if gap < 0:
        raise ValueError("gap must be >= 0 (cylinders overlap)")
    F = 0.0
    for z2, s2 in ((gap, -J2), (gap + L2, J2)):
        for z1, s1 in ((-L1, -J1), (0.0, J1)):
            F += coaxial_disc_force(s1, s2, radius, z2 - z1)
    return float(F)


def cylinder_demag_factor(radius, length):
    """Magnetometric (volume-averaged) axial demagnetising factor of a cylinder, N = 2R Phi(L/R)/L.

    Chen-Brug-Goldfarb magnetometric factor for chi = 0 [chen1991], written with the Lipschitz-Hankel
    kernel Phi (``magnet_force.py:101-103``). N -> 1 for a thin disc, N -> 0 for a long rod.
    """
    return float(2.0 * radius * field_kernel_Phi(length / radius) / length)


def aharoni_prism_demag(dx, dy, dz):
    """Magnetometric demagnetising factor D_z of a rectangular prism (full sides dx, dy, dz) [aharoni1998].

    Aharoni, J. Appl. Phys. 83, 3432 (1998), eq. (1), for the prism -a<=x<=a, -b<=y<=b, -c<=z<=c
    magnetised along z (a, b, c are half sides). Permute arguments for D_x, D_y; D_x + D_y + D_z = 1.
    """
    a, b, c = 0.5 * float(dx), 0.5 * float(dy), 0.5 * float(dz)
    abc = np.sqrt(a * a + b * b + c * c)
    ab = np.sqrt(a * a + b * b)
    bc = np.sqrt(b * b + c * c)
    ac = np.sqrt(a * a + c * c)
    pi_D = (
        (b * b - c * c) / (2 * b * c) * np.log((abc - a) / (abc + a))
        + (a * a - c * c) / (2 * a * c) * np.log((abc - b) / (abc + b))
        + b / (2 * c) * np.log((ab + a) / (ab - a))
        + a / (2 * c) * np.log((ab + b) / (ab - b))
        + c / (2 * a) * np.log((bc - b) / (bc + b))
        + c / (2 * b) * np.log((ac - a) / (ac + a))
        + 2 * np.arctan(a * b / (c * abc))
        + (a**3 + b**3 - 2 * c**3) / (3 * a * b * c)
        + (a * a + b * b - 2 * c * c) / (3 * a * b * c) * abc
        + c / (a * b) * (ac + bc)
        - (ab**3 + bc**3 + ac**3) / (3 * a * b * c)
    )
    return float(pi_D / np.pi)


def dipole_force(m1, m2, r):
    """Force on point dipole m2 at displacement r = r2 - r1 from dipole m1 [N].

    F = 3 MU0 / (4 pi |r|^5) [(m1.r) m2 + (m2.r) m1 + (m1.m2) r - 5 (m1.r)(m2.r) r / |r|^2],
    moments in A m^2 (m = J V / MU0 for a uniformly polarised body).
    """
    m1, m2, r = (np.asarray(a, dtype=float) for a in (m1, m2, r))
    d = np.linalg.norm(r)
    a, b, c = m1 @ r, m2 @ r, m1 @ m2
    return 3.0 * MU0 / (4.0 * np.pi * d**5) * (a * m2 + b * m1 + c * r - 5.0 * a * b * r / d**2)


# ---------------------------------------------------------------------------------------------
# 1-D nonlinear slab benchmark (ported from magnet_force.py:221-418)
# ---------------------------------------------------------------------------------------------
class CoaxialRodPair:
    """Two identical coaxial rods, each cut into axial slabs of uniform J (1-D benchmark).

    Port of ``CoaxialRodPair`` from ``$PARENT/simulations_old/Force_compute/python/magnet_force.py``
    (lines 221-418), adapted to take a ``magsim.materials.base.Material`` instead of the old
    ``(1-h^p)^q`` curve: each slab's polarisation along its own axis follows ``material.response`` of
    the slab-averaged axial field (H_perp = 0), and history advances only through ``commit``.
    Geometry: rod A occupies z in [-L, 0], rod B z in [gap, gap + L]. ``orientation`` = (sA, sB) sets
    the sense of each rod's easy axis (+1 = +z); (+1, +1) attracts. The model assumes J radially
    uniform in every slab, so it is a benchmark, not a reference: it differs from a 3-D solution by
    the radial non-uniformity of J (largest near the pole faces of a high-permeability material).

    Coupling [vokoun2009]: H_avg(slab i) = sum_k M[i, k] J_k with
    M = (R / (MU0 dz_i)) [Phi_diff(top sheets) - Phi_diff(bottom sheets)]. Force on B:
    F = (pi R^2 / MU0) sum_ij sign(d_ij) sigma_A,i sigma_B,j G(|d_ij| / R) (negative = attraction).
    """

    def __init__(self, radius, length, material, n_slabs=24, temperature=293.15, initial="+sat",
                 orientation=(1, 1)):
        self.radius = float(radius)
        self.length = float(length)
        self.material = material
        self.n_slabs = int(n_slabs)
        self.temperature = temperature
        self.state = material.init_state(2 * self.n_slabs, initial)
        self.set_orientation(*orientation)
        self._J_last = None

    def set_orientation(self, rod_a=1, rod_b=1):
        n = self.n_slabs
        self.orient = np.concatenate([np.full(n, float(rod_a)), np.full(n, float(rod_b))])
        self._J_last = None
        return self

    def _edges(self, gap):
        n, L = self.n_slabs, self.length
        return np.linspace(-L, 0.0, n + 1), np.linspace(gap, gap + L, n + 1)

    def coupling(self, gap):
        """Matrix M [A/(m T)] with H_z,avg(slab i) = sum_k M[i, k] J_z,k (signed, along +z)."""
        R = self.radius
        ea, eb = self._edges(gap)
        lo = np.concatenate([ea[:-1], eb[:-1]])
        hi = np.concatenate([ea[1:], eb[1:]])
        dz = hi - lo

        def avg(z_sheet):
            u1 = np.abs(lo[:, None] - z_sheet[None, :]) / R
            u2 = np.abs(hi[:, None] - z_sheet[None, :]) / R
            return (R / (MU0 * dz[:, None])) * field_kernel_Phi_diff(u1, u2)

        return avg(hi) - avg(lo)

    def _law(self, Hpar):
        n = Hpar.shape[0]
        Jp, dJ, _, _ = self.material.response(Hpar, np.zeros((n, 2)), self.state, self.temperature)
        return Jp, dJ

    def _newton(self, Ms, J, tol, max_iter):
        """Damped Newton on r(J) = J - law(Ms J) (J along each slab's own axis)."""
        eye = np.eye(J.size)
        Jp, dJ = self._law(Ms @ J)
        r = J - Jp
        rn = float(np.max(np.abs(r)))
        for _ in range(max_iter):
            if rn < tol:
                break
            try:
                step = np.linalg.solve(eye - dJ[:, None] * Ms, -r)
            except np.linalg.LinAlgError:
                break
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 5e-3, 1e-3):
                Jt = J + alpha * step
                Jp_t, dJ_t = self._law(Ms @ Jt)
                rt = Jt - Jp_t
                rtn = float(np.max(np.abs(rt)))
                if rtn < rn:
                    J, r, rn, dJ = Jt, rt, rtn, dJ_t
                    break
            else:
                break
        return J, rn

    def solve(self, gap, tol=1e-10, max_iter=80):
        """Self-consistent slab polarisations at ``gap`` [m] (trial: history not advanced).

        Returns (J_z signed [T] (2n,), H_par along each slab's own axis [A/m] (2n,)).
        Raises RuntimeError if neither Newton nor coupling continuation converges.
        """
        M = self.coupling(gap)
        s = self.orient
        Ms = M * s[None, :] * s[:, None]
        J0 = self._J_last if self._J_last is not None else self._law(np.zeros(2 * self.n_slabs))[0]
        J, res = self._newton(Ms, J0.copy(), tol, max_iter)
        if res > tol:
            J = self._law(np.zeros(2 * self.n_slabs))[0]
            for lam in np.linspace(0.1, 1.0, 10):
                J, res = self._newton(lam * Ms, J, tol, max_iter)
            if res > tol:
                raise RuntimeError(f"rod solver did not converge (residual {res:.2e} T at gap {gap:g} m)")
        self._J_last = J
        return s * J, Ms @ J

    def commit(self, gap):
        """Solve at ``gap`` and advance the material history (e.g. the open-circuit excursion)."""
        _, H = self.solve(gap)
        self.state = self.material.commit(H, self.state, self.temperature)
        return self

    def open_circuit(self, separation=None):
        """Drive both rods to their open-circuit operating point and commit it (``magnet_force.py:402``)."""
        if separation is None:
            separation = 200.0 * self.length
        saved = self.orient.copy()
        self.orient = np.ones_like(saved)
        self._J_last = None
        self.commit(separation)
        self.orient = saved
        self._J_last = None
        return self

    def force(self, gap):
        """Axial force on rod B [N] (negative = attraction) at ``gap`` [m], trial solve."""
        J, _ = self.solve(gap)
        n = self.n_slabs
        ea, eb = self._edges(gap)
        JA, JB = J[:n], J[n:]
        sA = np.concatenate([[-JA[0]], JA[:-1] - JA[1:], [JA[-1]]])
        sB = np.concatenate([[-JB[0]], JB[:-1] - JB[1:], [JB[-1]]])
        d = eb[None, :] - ea[:, None]
        k = force_kernel_G(np.abs(d) / self.radius)
        direction = np.where(d >= 0.0, 1.0, -1.0)
        return float(np.pi * self.radius**2 / MU0 * np.sum(direction * np.outer(sA, sB) * k))
