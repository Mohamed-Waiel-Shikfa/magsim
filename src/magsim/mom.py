"""Nonlinear magnetostatic volume-integral method of moments (task 3).

Method (Radia / MacMMems family [chubar1998, chadebec2006, carpentier2013]): every cell carries a
uniform polarisation J_k (global 3-vector, T). Its H-field is the exact field of a uniformly
polarised cuboid / cylinder / cylinder segment (magpylib kernels [ortner2020, ortner2023,
slanovc2022, caciagli2018]), so

    H_i = H_app,i + sum_k N_ik J_k,     N_ik = sum_g w_ig  H_k(x_ig) / |J|   [A/m per T],

with the target-cell average taken over its Gauss points x_ig (weights w_ig, sum 1 per cell). The
material law closes the system per cell in the local frame (e, e1, e2) of the easy axis:

    R(J) = J - [J_par(H_par) e + J_perp(H_perp)],   J_perp = MU0 chi_perp H_perp.

Damped Newton with Jacobian I - D N, D = dJ_dH_par e e^T + MU0 chi_perp (e1 e1^T + e2 e2^T), backtracking
line search on max|R|, and coupling/applied-field continuation H -> lam (H_app + N J), lam: 0 -> 1, as
fallback (pattern of ``$PARENT/analysis/fem3d.py:397-545``). Convergence: max|R| < ``tol`` [T]
AND the last Newton update max|dJ| < ``step_tol`` [T] (defaults 1e-9 T each).

Blocks: the self-block of every body is computed once in its body frame and rotated with the pose
(N_glob = R N_body R^T per 3x3 sub-block); cross-blocks are rebuilt when a pose changes. Assembly is
vectorised over (point, cell) pairs and chunked so temporary memory stays ~0.1 GB; the dense matrix
plus the Newton Jacobian are capped at ``mem_limit`` (1 GB default).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import magpylib
import magpylib.func as mpf
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from magsim.constants import MU0
from magsim.geometry import ANNULUS, CUBOID, CYLINDER, SEGMENT, Magnet, Pose, check_overlap, perp_axes

__all__ = ["MagnetSystem", "Solution", "cell_field"]

_PAIR_CHUNK = 60_000  # (point, cell) pairs per magpylib call (~0.05-0.1 GB temporaries)
# magpylib returns H = B / mu0_magpylib - J with its own mu0 (CODATA 2022, 6.8e-10 relative from the
# project's CODATA 2018 MU0). Rescale so every H in magsim is consistent with magsim.constants.MU0.
_MU0_RATIO = float(magpylib.mu_0) / MU0


# ---------------------------------------------------------------------------------------------
# Exact cell kernels
# ---------------------------------------------------------------------------------------------
def _h_local(kind: int, obs: np.ndarray, dims: np.ndarray, pol: np.ndarray) -> np.ndarray:
    """H [A/m] at cell-local, length-scaled observers (n, 3) from n cells of one ``kind``.

    ``dims`` (n, 5) length-scaled (angles in rad), ``pol`` (n, 3) local polarisation [T]. Fields are
    scale invariant, so lengths may be in any common unit.
    """
    if obs.shape[0] == 0:
        return np.zeros((0, 3))
    if kind == CUBOID:
        return mpf.cuboid_field("H", obs, dims[:, :3], pol, squeeze=False)
    if kind == CYLINDER:
        return mpf.cylinder_field("H", obs, np.stack([2 * dims[:, 1], dims[:, 2]], 1), pol, squeeze=False)
    if kind == ANNULUS:
        Ho = mpf.cylinder_field("H", obs, np.stack([2 * dims[:, 1], dims[:, 2]], 1), pol, squeeze=False)
        Hi = mpf.cylinder_field("H", obs, np.stack([2 * dims[:, 0], dims[:, 2]], 1), pol, squeeze=False)
        return Ho - Hi
    if kind == SEGMENT:
        d = np.stack([dims[:, 0], dims[:, 1], dims[:, 2], np.rad2deg(dims[:, 3]), np.rad2deg(dims[:, 4])], 1)
        return mpf.cylinder_segment_field("H", obs, d, pol, squeeze=False)
    raise ValueError(f"unknown cell kind {kind}")


def cell_field(points, kind, pos, dims, rot, pol, ell=1.0) -> np.ndarray:
    """H [A/m] at global ``points`` (n, 3) from cells given per row (exact kernels).

    kind (n,), pos (n, 3) source position [m], dims (n, 5) [m, rad], rot (n, 3, 3) body->global,
    pol (n, 3) global polarisation [T]; ``ell`` [m] rescales lengths (kernels are scale invariant).
    Rows with zero polarisation are skipped. Chunked; returns (n, 3).
    """
    points = np.asarray(points, dtype=float)
    out = np.zeros_like(points)
    nz = np.any(pol != 0.0, axis=1)
    for kd in np.unique(kind[nz]):
        sel = np.nonzero(nz & (kind == kd))[0]
        for c0 in range(0, sel.size, _PAIR_CHUNK):
            s = sel[c0:c0 + _PAIR_CHUNK]
            R = rot[s]
            loc = np.einsum("nji,nj->ni", R, points[s] - pos[s]) / ell
            pl = np.einsum("nji,nj->ni", R, pol[s])
            d = dims[s].copy()
            d[:, :3] /= ell
            if kd != CUBOID:
                d[:, 3:] = dims[s, 3:]
            Hl = _h_local(int(kd), loc, d, pl)
            out[s] = np.einsum("nij,nj->ni", R, Hl) * _MU0_RATIO
    return out


# ---------------------------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------------------------
@dataclass
class Solution:
    """Result of ``MagnetSystem.solve`` (all per cell, global frame, SI).

    J [T], H [A/m] (cell average, incl. H_app), B = MU0 H + J [T], H_par = H . e [A/m], H_app [A/m]
    (cell average of the applied field), iterations (Newton steps), residual max|R| [T], converged,
    history (max|R| after every accepted step), continuation (fallback used), version (pose version
    of the system the solution belongs to).
    """

    J: np.ndarray
    H: np.ndarray
    B: np.ndarray
    H_par: np.ndarray
    H_app: np.ndarray
    iterations: int
    residual: float
    converged: bool
    history: list = field(default_factory=list)
    continuation: bool = False
    version: int = 0


# ---------------------------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------------------------
class MagnetSystem:
    """Set of magnets solved together (interface contract in docs/SPEC.md).

    ``magnets``: list of ``magsim.geometry.Magnet`` with unique names. Overlapping bodies raise
    ValueError. ``mem_limit`` [bytes] caps the dense interaction matrix + Newton Jacobian.
    """

    def __init__(self, magnets, mem_limit: float = 1e9, check: bool = True):
        self.magnets: list[Magnet] = list(magnets)
        if not self.magnets:
            raise ValueError("need at least one magnet")
        names = [m.name for m in self.magnets]
        if len(set(names)) != len(names):
            raise ValueError(f"magnet names must be unique, got {names}")
        self.mem_limit = float(mem_limit)
        self._check = check
        counts = [m.n_cells for m in self.magnets]
        self.offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)
        self.n_cells = int(self.offsets[-1])
        need = 2 * (3 * self.n_cells) ** 2 * 8
        if need > self.mem_limit:
            raise MemoryError(f"{self.n_cells} cells need ~{need / 1e9:.2f} GB for N and the Jacobian "
                              f"(limit {self.mem_limit / 1e9:.2f} GB); coarsen the mesh")
        self.ell = float(max(m.shape.size for m in self.magnets))
        self.states = [m.material.init_state(m.n_cells, m.initial) for m in self.magnets]
        self._self_body = {}
        self._cross = {}
        self._N = None
        self.version = 0
        self._J_last = None
        self._update_geometry(range(len(self.magnets)))

    # ---- bookkeeping ---------------------------------------------------------------------
    def index(self, name_or_index) -> int:
        if isinstance(name_or_index, (int, np.integer)):
            return int(name_or_index)
        for i, m in enumerate(self.magnets):
            if m.name == name_or_index:
                return i
        raise KeyError(f"no magnet named {name_or_index!r}")

    def body_slice(self, b) -> slice:
        b = self.index(b)
        return slice(int(self.offsets[b]), int(self.offsets[b + 1]))

    def _update_geometry(self, bodies):
        if self._check:
            check_overlap(self.magnets)
        if not hasattr(self, "kind"):
            n = self.n_cells
            self.kind = np.concatenate([m.body_mesh.kind for m in self.magnets])
            self.dims = np.concatenate([m.body_mesh.dims for m in self.magnets])
            self.volume = np.concatenate([m.body_mesh.volume for m in self.magnets])
            self.body = np.concatenate([np.full(m.n_cells, i) for i, m in enumerate(self.magnets)])
            self.bs_r = np.concatenate([m.body_mesh.bsphere_r for m in self.magnets])
            for a in ("pos", "centroid", "easy", "e1", "e2", "bs_c"):
                setattr(self, a, np.zeros((n, 3)))
            self.rot = np.zeros((n, 3, 3))
            gcounts = [m.body_mesh.gw.size for m in self.magnets]
            self.goffsets = np.concatenate([[0], np.cumsum(gcounts)]).astype(int)
            self.gp = np.zeros((int(self.goffsets[-1]), 3))
            self.gw = np.concatenate([m.body_mesh.gw for m in self.magnets])
            self.gcell = np.concatenate([m.body_mesh.gcell + self.offsets[i] for i, m in enumerate(self.magnets)])
            self.gphi0 = np.concatenate([m.body_mesh.gphi0 for m in self.magnets])
            self.nphi_g = np.concatenate([m.body_mesh.nphi_g for m in self.magnets])
            self._ring_body = np.array([bool(np.all(m.body_mesh.kind != SEGMENT)) and m.body_mesh.kind[0] != CUBOID
                                        for m in self.magnets])
            self.gstart = np.concatenate([[0], np.cumsum(np.bincount(self.gcell, minlength=n))]).astype(int)
        for b in bodies:
            m = self.magnets[b]
            sl = self.body_slice(b)
            R = m.pose.matrix
            bm = m.body_mesh
            self.pos[sl] = m.pose.apply(bm.pos)
            self.centroid[sl] = m.pose.apply(bm.centroid)
            self.bs_c[sl] = m.pose.apply(bm.bsphere_c)
            self.rot[sl] = R
            e = R @ m.easy_axis
            e1, e2 = perp_axes(e)
            self.easy[sl], self.e1[sl], self.e2[sl] = e, e1, e2
            gs = slice(int(self.goffsets[b]), int(self.goffsets[b + 1]))
            self.gp[gs] = m.pose.apply(bm.gp)

    def set_pose(self, name, pose) -> None:
        """Move one body (Pose or YAML-like dict). Cross-blocks involving it are rebuilt lazily."""
        self.set_poses({name: pose})

    def set_poses(self, poses: dict) -> None:
        """Move several bodies at once (the overlap check runs once, on the final configuration)."""
        bodies = []
        old = {}
        for name, pose in poses.items():
            b = self.index(name)
            old[b] = self.magnets[b].pose
            self.magnets[b].pose = pose if isinstance(pose, Pose) else Pose.from_dict(pose)
            bodies.append(b)
        try:
            self._update_geometry(bodies)
        except ValueError:
            for b, p in old.items():
                self.magnets[b].pose = p
            self._update_geometry(list(old))
            raise
        for key in list(self._cross):
            if key[0] in bodies or key[1] in bodies:
                del self._cross[key]
        self._N = None
        self.version += 1

    def field_points(self):
        """Gauss points (n_pts, 3) [m] and weights (n_pts,) (sum 1 per cell) where H_app must be given."""
        return self.gp.copy(), self.gw.copy()

    def cell_average(self, values) -> np.ndarray:
        """Average point values (n_pts, k) over each cell's Gauss points -> (n_cells, k)."""
        v = np.asarray(values, dtype=float)
        return np.add.reduceat(v * self.gw[:, None], self.gstart[:-1], axis=0)

    # ---- fields --------------------------------------------------------------------------
    def field(self, points, J, bodies=None) -> np.ndarray:
        """Exact H [A/m] at global ``points`` (n, 3) from cells with polarisation ``J`` (n_cells, 3).

        ``bodies``: iterable of names / indices to include (default all). Chunked over points.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        J = np.asarray(J, dtype=float)
        if bodies is None:
            cells = np.arange(self.n_cells)
        else:
            cells = np.concatenate([np.arange(self.n_cells)[self.body_slice(b)] for b in bodies])
        cells = cells[np.any(J[cells] != 0.0, axis=1)]
        out = np.zeros_like(points)
        if cells.size == 0:
            return out
        step = max(1, _PAIR_CHUNK // cells.size)
        for p0 in range(0, points.shape[0], step):
            P = points[p0:p0 + step]
            pi = np.repeat(np.arange(P.shape[0]), cells.size)
            ci = np.tile(cells, P.shape[0])
            H = cell_field(P[pi], self.kind[ci], self.pos[ci], self.dims[ci], self.rot[ci], J[ci], self.ell)
            out[p0:p0 + step] = H.reshape(P.shape[0], cells.size, 3).sum(axis=1)
        return out

    def _block(self, tcells, scells, axis_rot=None) -> np.ndarray:
        """Interaction tensor (n_t, n_s, 3, 3) [A/(m T)]: H average over target cell a per unit J of k.

        ``axis_rot``: rotation matrix of the target body when target and source cells are all full rings
        on one common axis (its 3rd column). Then the field tensor is covariant under rotations about
        that axis, so only the first of the n_phi equally spaced azimuthal Gauss points is evaluated and
        the cell average is the exact C_nphi average  (1/n) sum_j Q_j T Q_j^T  (n_phi x fewer kernels).
        """
        nt, ns = tcells.size, scells.size
        out = np.zeros((nt, ns, 3, 3))
        if axis_rot is not None:
            gsel = self.gphi0
            gscale = self.nphi_g[self.gcell].astype(float)
        else:
            gsel = np.ones(self.gw.size, dtype=bool)
            gscale = np.ones(self.gw.size)
        allg = [np.arange(self.gstart[c], self.gstart[c + 1]) for c in tcells]
        allg = [g[gsel[g]] for g in allg]
        npts = np.array([g.size for g in allg])
        a0 = 0
        while a0 < nt:
            # chunk of target cells such that (points x sources x 3) stays bounded
            cum = np.cumsum(npts[a0:]) * ns * 3
            a1 = a0 + max(1, int(np.searchsorted(cum, _PAIR_CHUNK * 3, side="right")))
            gidx = np.concatenate(allg[a0:a1])
            P = self.gp[gidx]
            W = self.gw[gidx] * gscale[gidx]
            m = P.shape[0]
            pi = np.repeat(np.arange(m), ns)
            ci = np.tile(scells, m)
            T = np.zeros((m * ns, 3, 3))
            for j in range(3):
                # unit polarisation along the source cell's local axis j, expressed globally
                pol = self.rot[ci][:, :, j]
                T[:, :, j] = cell_field(P[pi], self.kind[ci], self.pos[ci], self.dims[ci], self.rot[ci], pol, self.ell)
            # column j is the response to local unit pol -> rotate columns to global: T_g = T_l R^T
            T = np.einsum("nij,nkj->nik", T, self.rot[ci])
            T = T.reshape(m, ns, 3, 3) * W[:, None, None, None]
            starts = np.concatenate([[0], np.cumsum(npts[a0:a1])[:-1]])
            out[a0:a1] = np.add.reduceat(T, starts, axis=0)
            a0 = a1
        if axis_rot is not None:
            nphi = int(self.nphi_g[tcells[0]])
            acc = np.zeros_like(out)
            for j in range(nphi):
                c, s_ = np.cos(2 * np.pi * j / nphi), np.sin(2 * np.pi * j / nphi)
                Q = axis_rot @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ axis_rot.T
                acc += np.einsum("ij,abjk,lk->abil", Q, out, Q)
            out = acc / nphi
        return out

    def _coaxial_rings(self, i, j):
        """True if bodies i and j are both meshed in full rings (nphi = 1) about one common axis."""
        if not (self._ring_body[i] and self._ring_body[j]):
            return False
        if i == j:
            return True
        Ri, Rj = self.magnets[i].pose.matrix, self.magnets[j].pose.matrix
        a = Ri[:, 2]
        off = self.magnets[j].pose.position - self.magnets[i].pose.position
        off = off - (off @ a) * a
        return abs(abs(a @ Rj[:, 2]) - 1.0) < 1e-12 and np.linalg.norm(off) < 1e-9 * self.ell

    def _self_block(self, b) -> np.ndarray:
        """Self-block of body b in the global frame (cached in the body frame)."""
        R = self.magnets[b].pose.matrix
        if b not in self._self_body:
            cells = np.arange(self.n_cells)[self.body_slice(b)]
            Ng = self._block(cells, cells, R if self._coaxial_rings(b, b) else None)
            self._self_body[b] = np.einsum("ji,abjk,kl->abil", R, Ng, R)  # R^T N R
        Nb = self._self_body[b]
        return np.einsum("ij,abjk,lk->abil", R, Nb, R)  # R N R^T

    def interaction_matrix(self) -> np.ndarray:
        """Dense N (3n, 3n) [A/(m T)] with H_cell = H_app + N @ J.ravel(); J ordered (cell, xyz)."""
        if self._N is not None:
            return self._N
        n = self.n_cells
        N = np.zeros((n, 3, n, 3))
        nb = len(self.magnets)
        for i in range(nb):
            si = self.body_slice(i)
            N[si, :, si, :] = self._self_block(i).transpose(0, 2, 1, 3)
            for j in range(nb):
                if i == j:
                    continue
                if (i, j) not in self._cross:
                    ci = np.arange(n)[si]
                    cj = np.arange(n)[self.body_slice(j)]
                    ax = self.magnets[i].pose.matrix if self._coaxial_rings(i, j) else None
                    self._cross[(i, j)] = self._block(ci, cj, ax)
                N[si, :, self.body_slice(j), :] = self._cross[(i, j)].transpose(0, 2, 1, 3)
        self._N = N.reshape(3 * n, 3 * n)
        return self._N

    # ---- material ------------------------------------------------------------------------
    def _law(self, H):
        """Material response at cell fields H (n, 3): target J (n, 3), tangent D (n, 3, 3), H_par (n,)."""
        e, e1, e2 = self.easy, self.e1, self.e2
        Hpar = np.einsum("ij,ij->i", H, e)
        Hperp = np.stack([np.einsum("ij,ij->i", H, e1), np.einsum("ij,ij->i", H, e2)], 1)
        Jt = np.zeros_like(H)
        D = np.zeros((H.shape[0], 3, 3))
        for b, m in enumerate(self.magnets):
            sl = self.body_slice(b)
            Jp, dJ, Jq, chi = m.material.response(Hpar[sl], Hperp[sl], self.states[b], m.temperature)
            Jp, dJ, Jq, chi = (np.asarray(x, dtype=float) for x in (Jp, dJ, Jq, chi))
            Jt[sl] = Jp[:, None] * e[sl] + Jq[:, 0:1] * e1[sl] + Jq[:, 1:2] * e2[sl]
            D[sl] = (dJ[:, None, None] * np.einsum("ni,nj->nij", e[sl], e[sl])
                     + MU0 * chi[:, None, None] * (np.einsum("ni,nj->nij", e1[sl], e1[sl])
                                                   + np.einsum("ni,nj->nij", e2[sl], e2[sl])))
        return Jt, D, Hpar

    # ---- solve ---------------------------------------------------------------------------
    def _happ_cells(self, H_app):
        n = self.n_cells
        if H_app is None:
            return np.zeros((n, 3))
        H_app = np.asarray(H_app, dtype=float)
        if H_app.shape == (3,):
            return np.tile(H_app, (n, 1))
        if H_app.shape == (self.gp.shape[0], 3):
            return self.cell_average(H_app)
        raise ValueError(f"H_app must have shape (3,) or ({self.gp.shape[0]}, 3) (see field_points()), got {H_app.shape}")

    def solve(self, H_app=None, J0=None, tol=1e-9, step_tol=1e-9, max_iter=60, continuation=True) -> Solution:
        """Self-consistent J at the current poses and applied field (trial: history is not advanced).

        H_app : None, uniform (3,), or (n_pts, 3) sampled at ``field_points()`` [A/m].
        J0 : initial guess (n_cells, 3) [T]; default the last committed / solved J, else the material
            response to H_app alone. tol, step_tol [T]: see module docstring.
        """
        n = self.n_cells
        Happ = self._happ_cells(H_app)
        N = self.interaction_matrix()
        Nr = N.reshape(n, 3, 3 * n)
        if J0 is not None:
            J = np.array(J0, dtype=float).reshape(n, 3)
        elif self._J_last is not None:
            J = self._J_last.copy()
        else:
            J = self._law(Happ)[0]

        def residual(J, lam):
            H = lam * (Happ + (N @ J.ravel()).reshape(n, 3))
            Jt, D, Hpar = self._law(H)
            return J - Jt, D, H, Hpar

        def newton(J, lam, iters):
            hist = []
            R, D, H, _ = residual(J, lam)
            rn = float(np.max(np.abs(R)))
            for k in range(1, iters + 1):
                A = -lam * np.einsum("aim,amc->aic", D, Nr).reshape(3 * n, 3 * n)
                A[np.diag_indices(3 * n)] += 1.0
                try:
                    lu = lu_factor(A, overwrite_a=True, check_finite=False)
                    dJ = lu_solve(lu, -R.ravel(), check_finite=False).reshape(n, 3)
                except (np.linalg.LinAlgError, ValueError):
                    return J, False, rn, k, hist
                if not np.all(np.isfinite(dJ)):
                    return J, False, rn, k, hist
                accepted = False
                for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 5e-3, 1e-3):
                    Jtry = J + alpha * dJ
                    Rt, Dt, Ht, _ = residual(Jtry, lam)
                    rt = float(np.max(np.abs(Rt)))
                    if rt <= (1.0 - 1e-4 * alpha) * rn or rt < 0.1 * tol:
                        step = alpha * float(np.max(np.abs(dJ)))
                        J, R, D, H, rn = Jtry, Rt, Dt, Ht, rt
                        accepted = True
                        break
                if not accepted:
                    return J, rn < tol, rn, k, hist
                hist.append(rn)
                if rn < tol and step < step_tol:
                    return J, True, rn, k, hist
            return J, rn < tol, rn, iters, hist

        J1, ok, res, its, hist = newton(J, 1.0, max_iter)
        used_cont = False
        if not ok and continuation:
            used_cont = True
            Jc = self._law(np.zeros((n, 3)))[0]
            lam, dlam = 0.0, 0.1
            ok_c = True
            while lam < 1.0:
                lt = min(1.0, lam + dlam)
                Jt, okt, rt, kt, ht = newton(Jc, lt, max_iter)
                its += kt
                if okt:
                    lam, Jc = lt, Jt
                    dlam = min(2 * dlam, 0.25)
                else:
                    dlam *= 0.5
                    if dlam < 1e-4:
                        ok_c = False
                        break
            if ok_c:
                J1, ok, res, k2, h2 = newton(Jc, 1.0, max_iter)
                its += k2
                hist += h2
        R, _, H, Hpar = residual(J1, 1.0)
        res = float(np.max(np.abs(R)))
        self._J_last = J1.copy()
        return Solution(J=J1, H=H, B=MU0 * H + J1, H_par=Hpar, H_app=Happ, iterations=its, residual=res,
                        converged=bool(ok and res < tol), history=hist, continuation=used_cont,
                        version=self.version)

    def commit(self, sol: Solution) -> None:
        """Accept ``sol`` into every body's material history (``material.commit`` with H_par)."""
        if sol.version != self.version:
            raise ValueError("solution belongs to an older pose configuration; re-solve before commit")
        for b, m in enumerate(self.magnets):
            sl = self.body_slice(b)
            self.states[b] = m.material.commit(sol.H_par[sl], self.states[b], m.temperature)
        self._J_last = sol.J.copy()
