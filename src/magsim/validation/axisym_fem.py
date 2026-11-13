"""Axisymmetric nonlinear magnetostatic FEM for coaxial rods in free space (independent check, gate F6).

Port of the needed parts of ``$PARENT/analysis/axisym_fem.py`` (no iron), driven by the ``Material``
protocol (``magsim.materials.base``).

Formulation (A_phi, scikit-fem P2 (default) or P1 triangles [scikit-fem: Gustafsson & McBain 2020, JOSS 5(52) 2369]):

    B_r = -dA/dz,   B_z = A/r + dA/dr,
    int [nu_r B_r(A) B_r(v) + nu_z B_z(A) B_z(v)] r dr dz = int nu_z J_irr B_z(v) r dr dz,

with A = 0 on the axis and on the outer box (r_far, |z| = z_far, default 40 x the axial extent of the rods;
the truncation error of a dipole far field is ~(extent / r_far)^3 ~ 2e-5). In a magnet the reversible part
is elementwise exact: nu_z = 1/(MU0 mu_op) with mu_op = ``material.mu_rec`` (1 when absent),
nu_r = 1/(MU0 mu_perp), mu_perp = 1 + chi_perp of ``material.response`` at H = 0. The irreversible part
J_irr = J_par(H) - MU0 (mu_op - 1) H is uniform per "slab" (an n_r x n_z grid of sub-regions per rod) and
evaluated at the slab-volume-mean axial field H (the same cell-average closure as the MoM), so the slab
mean of J equals ``material.response`` at the slab mean H.

Nonlinear solve (the old module's "permeance matrix" structure): the field is linear in the slab values x,
so H_slab = P x with P built by one LU factorisation and n_slab back-substitutions; then x = t(P x) is
solved by damped Newton with the analytic tangent dt/dH = dJ_par/dH - MU0 (mu_op - 1), with continuation
on P as fallback. Without iron P is exact, so there is no outer loop. Fixed bugs of the old code avoided:
fields are always recomputed from the final x (the old iron loop returned stale fields); the mesh size is
tied to the magnet (h), not to the far-field box (the old screening mesh gave 1.5 elements across a rod);
the convergence tolerance is scaled on the strongest slab, not slab 0.

Mesh: tensor grid of node lines that land exactly on every rod face and slab edge, element size
h * sigma(x) with sigma = 1 in the rods, f_hot = 0.25 at the rod faces and rim (sigma = f_hot + d/lam_hot,
lam_hot = 0.5 mm) and geometric growth into the air (sigma = 1 + d/lam_air, lam_air = 1 mm), so refining h
refines every element by the same ratio (systematic refinement for Richardson / GCI [celik2008]).

Force on the upper rod: Maxwell stress on the air layer between the rods (quadrature of the FE field),
F_z = -(1/g) int_layer (B_z^2 - B_r^2) / (2 MU0) dV (negative = attraction): the stress through every plane
of the layer is the same, so the layer average equals the plane value with lower discretisation noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, Functional, ElementTriP0, ElementTriP1, ElementTriP2, LinearForm, MeshTri, asm
from skfem.helpers import grad

from magsim.constants import MU0

__all__ = ["Rod", "AxisymRods", "node_axis"]


@dataclass
class Rod:
    """Coaxial rod: radius [m], z range [z0, z0 + length] [m], Material, easy axis +z (direction +1) or -z."""

    radius: float
    length: float
    z0: float
    material: object
    direction: float = 1.0
    initial: str | float = "+sat"
    temperature: float = 293.15
    slabs: tuple[int, int] = (1, 24)  # (n_r, n_z) irreversible-part sub-regions


def node_axis(breaks, h, sigma, n_sample=4000) -> np.ndarray:
    """Node coordinates hitting every break exactly, local spacing ~ h * sigma(x).

    In each interval the nodes are equally spaced in the stretched coordinate int dx / (h sigma(x)).
    """
    b = np.unique(np.asarray(breaks, dtype=float))
    pts = [b[:1]]
    for lo, hi in zip(b[:-1], b[1:]):
        x = np.linspace(lo, hi, n_sample)
        q = 1.0 / (h * sigma(x))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(x))])
        n = max(1, int(np.ceil(cum[-1] - 1e-9)))
        pts.append(np.interp(np.linspace(0, cum[-1], n + 1)[1:], cum, x))
        pts[-1][-1] = hi
    return np.concatenate(pts)


def _dist(x, pts):
    return np.min(np.abs(np.asarray(x)[:, None] - np.asarray(pts)[None, :]), axis=1)


def _dist_interval(x, lo, hi):
    return np.maximum(0.0, np.maximum(lo - x, x - hi))


class AxisymRods:
    """Coaxial rods (no iron) on an axisymmetric P1 mesh; solve / commit / force like ``MagnetSystem``.

    h [m]: element size inside the rods. far_factor: outer box = far_factor x axial extent.
    """

    F_HOT, LAM_HOT, LAM_AIR = 0.25, 0.5e-3, 1.0e-3

    def __init__(self, rods: list[Rod], h: float, far_factor: float = 40.0, intorder: int = 4, order: int = 2):
        self.rods = list(rods)
        self.h = float(h)
        R = max(r.radius for r in rods)
        z_lo = min(r.z0 for r in rods)
        z_hi = max(r.z0 + r.length for r in rods)
        far = far_factor * (z_hi - z_lo)
        zc = 0.5 * (z_lo + z_hi)
        faces = sorted({r.z0 for r in rods} | {r.z0 + r.length for r in rods})
        rims = sorted({r.radius for r in rods})
        rb, zb = {0.0, far}, {zc - far, zc + far}
        for r in rods:
            nr, nz = r.slabs
            rb.update(np.linspace(0.0, r.radius, nr + 1))
            zb.update(np.linspace(r.z0, r.z0 + r.length, nz + 1))
        fh, lh, la = self.F_HOT, self.LAM_HOT, self.LAM_AIR

        def sig_r(x):
            return np.minimum(1.0 + _dist_interval(x, 0.0, R) / la, fh + _dist(x, rims) / lh)

        def sig_z(x):
            return np.minimum(1.0 + _dist_interval(x, z_lo, z_hi) / la, fh + _dist(x, faces) / lh)

        rs = node_axis(sorted(rb), self.h, sig_r)
        zs = node_axis(sorted(zb), self.h, sig_z)
        self.rs, self.zs = rs, zs
        self.mesh = MeshTri.init_tensor(rs, zs)
        self.b1 = Basis(self.mesh, ElementTriP1() if order == 1 else ElementTriP2(), intorder=intorder)
        self.b0 = Basis(self.mesh, ElementTriP0(), intorder=intorder)
        ne = self.mesh.t.shape[1]
        rc = self.mesh.p[0][self.mesh.t].mean(0)
        zcen = self.mesh.p[1][self.mesh.t].mean(0)
        self.rc, self.zc = rc, zcen

        # tags and slabs
        self.rod_of = np.full(ne, -1)
        self.slab_of = np.full(ne, -1)
        self.slab_rod = []
        k = 0
        nu_r = np.full(ne, 1.0 / MU0)
        nu_z = np.full(ne, 1.0 / MU0)
        self.mu_op = []
        for i, r in enumerate(rods):
            inside = (rc < r.radius) & (zcen > r.z0) & (zcen < r.z0 + r.length)
            self.rod_of[inside] = i
            mu_op = float(getattr(r.material, "mu_rec", 1.0))
            st = r.material.init_state(1, "+sat")
            chi = float(np.asarray(r.material.response(np.zeros(1), np.zeros((1, 2)), st, r.temperature)[3])[0])
            nu_z[inside] = 1.0 / (MU0 * mu_op)
            nu_r[inside] = 1.0 / (MU0 * (1.0 + chi))
            self.mu_op.append(mu_op)
            nr, nz = r.slabs
            re = np.linspace(0.0, r.radius, nr + 1)
            ze = np.linspace(r.z0, r.z0 + r.length, nz + 1)
            ir = np.clip(np.searchsorted(re, rc, side="right") - 1, 0, nr - 1)
            iz = np.clip(np.searchsorted(ze, zcen, side="right") - 1, 0, nz - 1)
            self.slab_of[inside] = k + ir[inside] * nz + iz[inside]
            self.slab_rod += [i] * (nr * nz)
            k += nr * nz
        self.slab_rod = np.array(self.slab_rod)
        self.n_slab = k
        self.nu_r, self.nu_z = nu_r, nu_z
        self.mu_e = np.where(self.rod_of >= 0, 1.0 / (MU0 * nu_z), 1.0)
        self.dir_slab = np.array([rods[i].direction for i in self.slab_rod], dtype=float)

        # operators
        nqp = self.b1.X.shape[1]
        q = lambda v: np.tile(v[:, None], (1, nqp))  # noqa: E731

        @BilinearForm
        def stiff(u, v, w):
            r = np.maximum(w.x[0], 1e-15)
            gu, gv = grad(u), grad(v)
            return (w["nur"] * gu[1] * gv[1] + w["nuz"] * (u / r + gu[0]) * (v / r + gv[0])) * r

        @BilinearForm
        def src(u, v, w):  # u: P0 J_irr, v: P1 test
            r = np.maximum(w.x[0], 1e-15)
            return w["nuz"] * u * (v / r + grad(v)[0]) * r

        @BilinearForm
        def bz_op(u, v, w):  # u: P1 A, v: P0
            r = np.maximum(w.x[0], 1e-15)
            return (u / r + grad(u)[0]) * v * r

        @BilinearForm
        def br_op(u, v, w):
            return -grad(u)[1] * v * w.x[0]

        @LinearForm
        def rvol(v, w):
            return v * w.x[0]

        K = asm(stiff, self.b1, nur=q(nu_r), nuz=q(nu_z))
        self.G = asm(src, self.b0, self.b1, nuz=q(nu_z)).tocsr()  # (n_nodes, n_elem)
        wr = asm(rvol, self.b0)  # int r dA per element
        self.vol = 2.0 * np.pi * wr
        self.Pz = csr_matrix(asm(bz_op, self.b1, self.b0).multiply(1.0 / wr[:, None]))
        self.Pr = csr_matrix(asm(br_op, self.b1, self.b0).multiply(1.0 / wr[:, None]))
        D = self.b1.get_dofs(lambda x: (np.abs(x[0]) < 1e-14) | (np.abs(x[0] - rs[-1]) < 1e-12)
                             | (np.abs(x[1] - zs[0]) < 1e-12) | (np.abs(x[1] - zs[-1]) < 1e-12)).all()
        free = np.setdiff1d(np.arange(self.b1.N), D)
        self.free = free
        self.lu = splu(K.tocsr()[free][:, free].tocsc())
        # slab expansion / averaging
        mag = self.slab_of >= 0
        self.S = csr_matrix((np.ones(mag.sum()), (np.nonzero(mag)[0], self.slab_of[mag])), shape=(ne, self.n_slab))
        Vs = np.bincount(self.slab_of[mag], weights=self.vol[mag], minlength=self.n_slab)
        self.slab_vol = Vs
        self.Wavg = csr_matrix((self.vol[mag] / Vs[self.slab_of[mag]], (self.slab_of[mag], np.nonzero(mag)[0])),
                               shape=(self.n_slab, ne))
        self._P = None
        self.states = [r.material.init_state(r.slabs[0] * r.slabs[1], r.initial) for r in rods]
        self.x_last = None

    # ---- linear algebra -----------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.b1.N

    def potential(self, Jirr_elem) -> np.ndarray:
        """A_phi (n_nodes,) for element-wise irreversible polarisation along +z [T]; A = 0 on the boundary."""
        f = self.G @ Jirr_elem
        A = np.zeros(self.b1.N)
        A[self.free] = self.lu.solve(f[self.free])
        return A

    def fields(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Element-mean B_r, B_z [T] and H_z [A/m] for slab irreversible values x (signed along +z)."""
        Je = self.S @ x
        A = self.potential(Je)
        Bz, Br = self.Pz @ A, self.Pr @ A
        Hz = (Bz - Je) / (MU0 * self.mu_e)
        return Br, Bz, Hz

    def permeance(self) -> np.ndarray:
        """P (n_slab, n_slab) [A/(m T)]: slab-mean H_z = P @ x (x: slab J_irr along +z)."""
        if self._P is None:
            P = np.zeros((self.n_slab, self.n_slab))
            for c0 in range(0, self.n_slab, 32):
                cols = np.arange(c0, min(self.n_slab, c0 + 32))
                Je = self.S[:, cols].toarray()
                f = (self.G @ Je)[self.free]
                A = np.zeros((self.b1.N, cols.size))
                A[self.free] = self.lu.solve(f)
                Hz = (self.Pz @ A - Je) / (MU0 * self.mu_e[:, None])
                P[:, cols] = self.Wavg @ Hz
            self._P = P
        return self._P

    # ---- material ------------------------------------------------------------------------
    def _target(self, Hz):
        """Slab J_irr demanded by the material law at slab-mean H_z, and dJ_irr/dH_z (both along +z)."""
        t = np.zeros(self.n_slab)
        dt = np.zeros(self.n_slab)
        for i, r in enumerate(self.rods):
            sl = self.slab_rod == i
            Hp = r.direction * Hz[sl]
            Jp, dJ, _, _ = r.material.response(Hp, np.zeros((Hp.size, 2)), self.states[i], r.temperature)
            rev = MU0 * (self.mu_op[i] - 1.0)
            t[sl] = r.direction * (np.asarray(Jp) - rev * Hp)
            dt[sl] = np.asarray(dJ) - rev
        return t, dt

    def solve(self, tol=1e-10, max_iter=80):
        """Self-consistent slab state (trial). Returns dict x, H_slab, J_slab (total, along +z), residual."""
        P = self.permeance()
        x = self.x_last.copy() if self.x_last is not None else self._target(np.zeros(self.n_slab))[0]
        scale = max(float(np.max(np.abs(self._target(np.zeros(self.n_slab))[0]))), 1e-9)

        def newton(x, lam, iters):
            t, dt = self._target(lam * P @ x)
            r = x - t
            rn = float(np.max(np.abs(r)))
            for _ in range(iters):
                if rn < tol * scale:
                    return x, rn, True
                Jac = np.eye(self.n_slab) - lam * dt[:, None] * P
                step = np.linalg.solve(Jac, -r)
                for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 5e-3):
                    xt = x + a * step
                    tt, dtt = self._target(lam * P @ xt)
                    rt = xt - tt
                    if float(np.max(np.abs(rt))) < rn:
                        x, r, dt, rn = xt, rt, dtt, float(np.max(np.abs(rt)))
                        break
                else:
                    return x, rn, rn < tol * scale
            return x, rn, rn < tol * scale

        x1, res, ok = newton(x, 1.0, max_iter)
        if not ok:
            xc = self._target(np.zeros(self.n_slab))[0]
            for lam in np.linspace(0.05, 1.0, 40):
                xc, res, ok = newton(xc, lam, max_iter)
            x1 = xc
        if not ok:
            raise RuntimeError(f"axisymmetric FEM solve did not converge (residual {res:.2e} T)")
        self.x_last = x1
        Hs = P @ x1
        Jtot = x1 + np.array([MU0 * (self.mu_op[i] - 1.0) for i in self.slab_rod]) * Hs
        return {"x": x1, "H_slab": Hs, "J_slab": Jtot, "residual": res}

    def commit(self, sol) -> None:
        """Advance every rod's material history with its slab-mean H_par."""
        for i, r in enumerate(self.rods):
            sl = self.slab_rod == i
            self.states[i] = r.material.commit(r.direction * sol["H_slab"][sl], self.states[i], r.temperature)

    def mean_J(self, sol, rod=0) -> float:
        """Volume-mean J.e of a rod [T] (e = its easy axis)."""
        sl = self.slab_rod == rod
        return float(self.rods[rod].direction * np.sum(sol["J_slab"][sl] * self.slab_vol[sl]) / self.slab_vol[sl].sum())

    def mean_H(self, sol, rod=0) -> float:
        sl = self.slab_rod == rod
        return float(self.rods[rod].direction * np.sum(sol["H_slab"][sl] * self.slab_vol[sl]) / self.slab_vol[sl].sum())

    def force_upper(self, sol) -> float:
        """Axial force [N] on the rod(s) above the air layer between the two rods (negative = attraction)."""
        if len(self.rods) != 2:
            raise ValueError("force needs exactly two rods")
        lo_rod, hi_rod = sorted(self.rods, key=lambda r: r.z0)
        z_a, z_b = lo_rod.z0 + lo_rod.length, hi_rod.z0
        if z_b - z_a <= 0:
            raise ValueError("the Maxwell-stress layer needs a positive gap")
        A = self.potential(self.S @ sol["x"])
        lay = ((self.zc > z_a) & (self.zc < z_b)).astype(float)
        nqp = self.b1.X.shape[1]

        @Functional
        def tzz(w):  # (B_z^2 - B_r^2) / (2 MU0) * 2 pi r on the layer elements (exact quadrature of the FE field)
            r = np.maximum(w.x[0], 1e-15)
            bz = w["A"].value / r + w["A"].grad[0]
            br = -w["A"].grad[1]
            return w["lay"] * (bz**2 - br**2) / (2.0 * MU0) * 2.0 * np.pi * r

        Fl = tzz.assemble(self.b1, A=self.b1.interpolate(A), lay=np.tile(lay[:, None], (1, nqp)))
        return float(-Fl / (z_b - z_a))


# ---------------------------------------------------------------------------------------------
# gate F6 cases: open circuit, then the pair at a gap
# ---------------------------------------------------------------------------------------------
def fem_rod_case(material, radius, length, gap, h, slabs=(6, 36), far_factor=40.0, order=2) -> dict:
    """FEM open-circuit and pair results for two identical coaxial rods (easy axes +z, attracting).

    History: +sat -> one rod alone (open circuit), solved and committed -> both rods (each with that
    committed state) at ``gap`` [m], solved. Returns open-circuit mean J [T], mean H [A/m], the factor
    -MU0<H>/<J>, the pair force on the upper rod [N] and its mean J, and node counts.
    """
    single = AxisymRods([Rod(radius, length, -0.5 * length, material, slabs=slabs)], h, far_factor, order=order)
    s1 = single.solve()
    J_oc, H_oc = single.mean_J(s1), single.mean_H(s1)
    single.commit(s1)
    out = {"h": h, "J_oc": J_oc, "H_oc": H_oc, "N_oc": -MU0 * H_oc / J_oc, "nodes_single": single.n_nodes}
    if gap is not None:
        pair = AxisymRods([Rod(radius, length, -length, material, slabs=slabs),
                           Rod(radius, length, gap, material, slabs=slabs)], h, far_factor, order=order)
        pair.states = [single.states[0], single.states[0]]
        s2 = pair.solve()
        out.update(F=pair.force_upper(s2), J_pair=pair.mean_J(s2, 1), nodes_pair=pair.n_nodes)
    return out


def fem_replay(material, radius, length, gaps, n_rep=3, h=0.3e-3, slabs=(6, 36), contact_gap=1e-6):
    """FEM replay of the E2 handling history (same protocol as ``experiment.e2_replay``).

    +sat -> open circuit (each rod alone, committed) -> for each gap: n_rep x (place: solve + commit,
    measure: solve, force, commit, separate: each rod alone, solve + commit). Gaps below ``contact_gap`` [m]
    are evaluated at ``contact_gap`` (the Maxwell-stress layer needs a positive thickness; at 1 um the force
    differs from contact by ~dF/dg x 1 um, ~0.2 % for the E2 rods). Returns rows (gap_m, rep, F = -Fz, J_B).
    """
    single = AxisymRods([Rod(radius, length, -0.5 * length, material, slabs=slabs)], h)

    def open_circuit(state):
        single.states, single.x_last = [state], None
        single.commit(single.solve())
        return single.states[0]

    st = open_circuit(material.init_state(slabs[0] * slabs[1], "+sat"))
    sA = sB = st
    rows = []
    for g in gaps:
        gg = max(float(g), contact_gap)
        pair = AxisymRods([Rod(radius, length, -length, material, slabs=slabs),
                           Rod(radius, length, gg, material, slabs=slabs)], h)
        for k in range(n_rep):
            pair.states, pair.x_last = [sA, sB], None
            pair.commit(pair.solve())  # place
            sol = pair.solve()  # measure
            F = pair.force_upper(sol)
            rows.append({"gap_m": float(g), "rep": k + 1, "F": -F, "J_mean_B": pair.mean_J(sol, 1)})
            pair.commit(sol)
            sA, sB = pair.states
            sA, sB = open_circuit(sA), open_circuit(sB)  # separate
    return rows
