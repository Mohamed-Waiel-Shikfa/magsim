"""Force and torque on a body by magnetic-charge integration (task 3).

With uniform J per cell, the only magnetic charges are surface charges on cell faces: on a face with
normal n between cells "minus" (-n side) and "plus" (+n side), sigma = (J_minus - J_plus) . n [T]
(external faces: J . n_out). The force and torque on the target body from all OTHER bodies are

    F = sum_faces  integral sigma H_other dA                                     [N]
    T = sum_faces  integral (r - p) x sigma H_other dA                           [N m]

(J in T, H in A/m -> N/m^2). This is the moment of the Coulomb charge forces. NOTE: docs/SPEC.md
also lists a volume term sum_cells V J x H_other; for the charge formulation that term is already
contained in the moment of the surface charges (for a uniformly polarised body in a uniform field,
integral (r x sigma H) dA = V J x H exactly), so adding it double counts. The
Kelvin form (r x (J.grad)H + J x H) and the charge form above are equal after integration by parts;
the finite-difference virtual-work test in ``tests/test_forces.py`` confirms the charge form and
rejects the extra term.

H_other comes from the other bodies' cells with magpylib's exact kernels. Evaluation points on
external target faces are moved 1e-10 * ell into the target, so at contact (gap 0) the field is the
one-sided limit on the target side.

Dispatch per (target face panel, source cell) pair, with d = estimated panel-cell distance (exact
signed distance at 9x9 panel samples minus the sampling resolution; bounding spheres when far) and
L = panel size:
- d >= L/5: one tensor Gauss-Legendre rule on the panel, order per direction from the
  Bernstein-ellipse bound rho^(-2q) <= quad_tol (rho = 2d/L_dir + sqrt((2d/L_dir)^2 + 1); 2 <= q <= 30;
  quad_tol = 1e-8 default).
- d < L/5 ("near"): parameter intervals split at the projections of the source cell's corners /
  edges, tanh-sinh on every sub-interval (37 nodes, nodes within 1e-10 of an end dropped). This handles
  the log / jump (near-)singularities that sit at the sub-interval ends for aligned and coaxial
  geometries. For non-aligned near pairs the sub-intervals are further cut into pieces no longer
  than ~2d; accuracy there is lower (``force_torque_detail(error_estimate=True)`` reports it).
- Near pairs that are aligned cuboids (target cuboid face, source cuboid cell whose frame differs from
  the target frame by a signed permutation): the charges of the source faces parallel to the target
  face interact in closed form (Akoun-Yonnet corner sums [akoun1984, allag2009],
  ``magsim.analytic.rect_pair_integral``), exact at any gap including contact; only the remaining
  (perpendicular) source charges are integrated numerically. The torque is integrated numerically,
  with the lever-arm part (c_face - p) x F_face replaced by the exact face force.
Nodes that land within ~1e-8 ell of a source edge (only possible at contact, weights ~1e-15) make
magpylib's cuboid kernel return nan (0 * inf); they are dropped and counted in ``n_pairs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magsim.analytic import rect_pair_integral
from magsim.constants import MU0
from magsim.geometry import ANNSEC, CUBOID, CYLINDER, CYLPATCH, RECT, SEGMENT, Cuboid
from magsim.mom import MagnetSystem, Solution, cell_field

__all__ = ["force_torque", "force_torque_detail", "ForceResult"]

_NODE_BATCH = 400_000


@dataclass
class ForceResult:
    """Force [N] and torque [N m] on the target, plus quadrature bookkeeping."""

    F: np.ndarray
    T: np.ndarray
    pivot: np.ndarray
    n_nodes: int
    n_pairs: dict
    error_F: float | None = None
    error_T: float | None = None


# ---------------------------------------------------------------------------------------------
# 1-D rules on [0, 1]
# ---------------------------------------------------------------------------------------------
def _gl(q):
    x, w = np.polynomial.legendre.leggauss(int(q))
    return 0.5 * (x + 1.0), 0.5 * w


def _ts(level):
    """Tanh-sinh nodes / weights on [0, 1]; dense near both ends (endpoint log / jump singularities)."""
    K = 18 * (2**level)
    tmax = 3.2
    h = tmax / K
    t = np.arange(-K, K + 1) * h
    u = 0.5 * np.pi * np.sinh(t)
    x = np.tanh(u)
    w = h * 0.5 * np.pi * np.cosh(t) / np.cosh(u) ** 2
    keep = (1.0 - np.abs(x)) > 1e-10  # nodes closer to an end carry < ~3e-9 of a log-singular integral
    return 0.5 * (x[keep] + 1.0), 0.5 * w[keep]


_NEAR = 0.2  # d / L below which the breakpoint + tanh-sinh rule is used


def _q_far(ratio, tol, level):
    """Gauss-Legendre order per direction so that rho^(-2q) <= tol, rho = Bernstein-ellipse parameter
    of the nearest singularity at distance d = ratio * L from a panel of size L."""
    x = 2.0 * np.asarray(ratio, dtype=float)
    rho = x + np.sqrt(x * x + 1.0)
    q = np.ceil(np.log(1.0 / tol) / (2.0 * np.log(rho))).astype(int) + 2 * level
    return np.clip(q, 2, 30)


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------
def _map(kind, o, e1, e2, e3, R, s, t):
    """Face parameter (s, t) -> point, unit normal (convention +n), jacobian. Arrays per node."""
    cs, sn = np.cos(s), np.sin(s)
    ct, st = np.cos(t), np.sin(t)
    rect = kind == RECT
    ann = kind == ANNSEC
    cyl = kind == CYLPATCH
    P = np.empty(o.shape)
    nrm = np.empty(o.shape)
    jac = np.ones(s.shape)
    if np.any(rect):
        P[rect] = o[rect] + s[rect, None] * e1[rect] + t[rect, None] * e2[rect]
        nrm[rect] = e3[rect]
    if np.any(ann):
        P[ann] = o[ann] + s[ann, None] * (ct[ann, None] * e1[ann] + st[ann, None] * e2[ann])
        nrm[ann] = e3[ann]
        jac[ann] = s[ann]
    if np.any(cyl):
        rh = cs[cyl, None] * e1[cyl] + sn[cyl, None] * e2[cyl]
        P[cyl] = o[cyl] + R[cyl, None] * rh + t[cyl, None] * e3[cyl]
        nrm[cyl] = rh
        jac[cyl] = R[cyl]
    return P, nrm, jac


def _param(kind, o, e1, e2, e3, P):
    """Inverse map: points (n, 3) -> face parameters (s, t) for each row's face."""
    d = P - o
    a, b, c = (d * e1).sum(-1), (d * e2).sum(-1), (d * e3).sum(-1)
    s = np.where(kind == RECT, a, np.where(kind == ANNSEC, np.hypot(a, b), np.arctan2(b, a)))
    t = np.where(kind == RECT, b, np.where(kind == ANNSEC, np.arctan2(b, a), c))
    return s, t


def _cell_sdf(P, kind, pos, dims, rot):
    """Signed distance (lower bound for segments) from points P (n, 3) to cells given per row."""
    loc = np.einsum("nji,nj->ni", rot, P - pos)
    out = np.empty(P.shape[0])
    cub = kind == CUBOID
    if np.any(cub):
        q = np.abs(loc[cub]) - 0.5 * dims[cub, :3]
        out[cub] = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    cy = ~cub
    if np.any(cy):
        lc, dm, kd = loc[cy], dims[cy], kind[cy]
        rho = np.hypot(lc[:, 0], lc[:, 1])
        dr = np.where(kd == CYLINDER, rho - dm[:, 1], np.maximum(dm[:, 0] - rho, rho - dm[:, 1]))
        dd = np.stack([dr, np.abs(lc[:, 2]) - 0.5 * dm[:, 2]], 1)
        sd = np.linalg.norm(np.maximum(dd, 0), axis=1) + np.minimum(dd.max(1), 0)
        seg = kd == SEGMENT
        if np.any(seg):
            p1, p2 = dm[seg, 3], dm[seg, 4]
            n1 = np.stack([np.sin(p1), -np.cos(p1)], 1)
            n2 = np.stack([-np.sin(p2), np.cos(p2)], 1)
            xy = lc[seg, :2]
            wedge = np.maximum((n1 * xy).sum(1), (n2 * xy).sum(1))
            sd[seg] = np.maximum(sd[seg], wedge)
        out[cy] = sd
    return out


def _cell_features(kind, pos, dims, rot):
    """Feature points (n, 8, 3) whose projections mark the cell's edges; flag (n,) = phi meaningful."""
    n = kind.size
    loc = np.zeros((n, 8, 3))
    sg = np.array([[i, j, k] for i in (-0.5, 0.5) for j in (-0.5, 0.5) for k in (-0.5, 0.5)])
    cub = kind == CUBOID
    loc[cub] = sg[None] * dims[cub, None, :3]
    cy = ~cub
    if np.any(cy):
        r = np.stack([dims[cy, 0], dims[cy, 0], dims[cy, 0], dims[cy, 0], dims[cy, 1], dims[cy, 1], dims[cy, 1], dims[cy, 1]], 1)
        full = kind[cy] != SEGMENT
        ph1 = np.where(full, 0.0, dims[cy, 3])
        ph2 = np.where(full, 0.0, dims[cy, 4])
        ph = np.stack([ph1, ph1, ph2, ph2, ph1, ph1, ph2, ph2], 1)
        z = np.outer(dims[cy, 2], [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5])
        loc[cy] = np.stack([r * np.cos(ph), r * np.sin(ph), z], -1)
    P = np.einsum("nij,nkj->nki", rot, loc) + pos[:, None, :]
    return P, kind == SEGMENT


# ---------------------------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------------------------
def force_torque(system: MagnetSystem, sol: Solution, target, pivot=None):
    """Force [N] and torque [N m] on body ``target`` from all other bodies (see module docstring).

    ``pivot`` [m] defaults to the target's volume centroid. Returns (F, T), each shape (3,).
    """
    r = force_torque_detail(system, sol, target, pivot)
    return r.F, r.T


def force_torque_detail(system: MagnetSystem, sol: Solution, target, pivot=None, quad_tol=1e-8,
                        level=0, torque=True, error_estimate=False, closed_form=True) -> ForceResult:
    """``force_torque`` with options and bookkeeping.

    quad_tol : target relative quadrature error of the Gauss-Legendre pairs (Bernstein bound).
    level : rule refinement (0 default; each level adds Gauss points and doubles tanh-sinh density).
    torque : False skips the torque-only quadrature of the closed-form pairs (faster).
    error_estimate : also run ``level + 1`` and report |dF|, |dT| as the quadrature error estimate.
    closed_form : False forces quadrature for every pair (used to test the quadrature path).
    """
    res = _force(system, sol, target, pivot, quad_tol, level, torque, closed_form)
    if error_estimate:
        r2 = _force(system, sol, target, pivot, quad_tol * 1e-2, level + 1, torque, closed_form)
        res.error_F = float(np.linalg.norm(r2.F - res.F))
        res.error_T = float(np.linalg.norm(r2.T - res.T))
    return res


def _force(system, sol, target, pivot, quad_tol, level, torque, closed_form=True) -> ForceResult:
    if sol.version != system.version:
        raise ValueError("solution belongs to an older pose configuration; re-solve first")
    b = system.index(target)
    mag = system.magnets[b]
    tsl = system.body_slice(b)
    ell = system.ell
    Rt = mag.pose.matrix
    if pivot is None:
        V = system.volume[tsl]
        pivot = (system.centroid[tsl] * V[:, None]).sum(0) / V.sum()
    pivot = np.asarray(pivot, dtype=float)
    src = np.nonzero(system.body != b)[0]
    src = src[np.any(sol.J[src] != 0.0, axis=1)]
    zero = ForceResult(np.zeros(3), np.zeros(3), pivot, 0, {})
    if src.size == 0:
        return zero

    # ---- target faces (global) and their net charge vectors ------------------------------
    bf = mag.body_mesh.faces
    Jt = sol.J[tsl]
    Jm = np.where(bf.minus[:, None] >= 0, Jt[np.maximum(bf.minus, 0)], 0.0)
    Jp = np.where(bf.plus[:, None] >= 0, Jt[np.maximum(bf.plus, 0)], 0.0)
    dJ = Jm - Jp
    o = mag.pose.apply(bf.o)
    e1, e2, e3 = bf.e1 @ Rt.T, bf.e2 @ Rt.T, bf.e3 @ Rt.T
    kind, R, rng = bf.kind, bf.R, bf.rng
    inward = np.where(bf.minus < 0, 1.0, np.where(bf.plus < 0, -1.0, 0.0))
    # skip charge-free faces
    nn = np.abs((dJ * e3).sum(1))
    inplane = np.linalg.norm(dJ - (dJ * e3).sum(1)[:, None] * e3, axis=1)
    live = np.where(kind == CYLPATCH, inplane, nn) > 1e-300
    fidx = np.nonzero(live)[0]
    if fidx.size == 0:
        return zero

    # ---- base panels -----------------------------------------------------------------
    P_f, P_s0, P_s1, P_t0, P_t1 = [], [], [], [], []
    for f in fidx:
        s0, s1, t0, t1 = rng[f]
        if kind[f] == ANNSEC:
            k = max(1, int(np.ceil((t1 - t0) / (0.5 * np.pi) - 1e-9)))
            ts = np.linspace(t0, t1, k + 1)
            for i in range(k):
                P_f.append(f); P_s0.append(s0); P_s1.append(s1); P_t0.append(ts[i]); P_t1.append(ts[i + 1])
        elif kind[f] == CYLPATCH:
            k = max(1, int(np.ceil((s1 - s0) / (0.5 * np.pi) - 1e-9)))
            ss = np.linspace(s0, s1, k + 1)
            for i in range(k):
                P_f.append(f); P_s0.append(ss[i]); P_s1.append(ss[i + 1]); P_t0.append(t0); P_t1.append(t1)
        else:
            P_f.append(f); P_s0.append(s0); P_s1.append(s1); P_t0.append(t0); P_t1.append(t1)
    pf = np.array(P_f)
    ps0, ps1, pt0, pt1 = (np.array(a) for a in (P_s0, P_s1, P_t0, P_t1))
    npan = pf.size
    # panel samples (9 x 9) -> centre, size; physical extents along s and t
    g = np.linspace(0, 1, 9)
    gs, gt = np.meshgrid(g, g, indexing="ij")
    gs, gt = gs.ravel(), gt.ravel()
    rep = np.repeat(np.arange(npan), gs.size)
    fs = pf[rep]
    S = ps0[rep] + (ps1 - ps0)[rep] * np.tile(gs, npan)
    T_ = pt0[rep] + (pt1 - pt0)[rep] * np.tile(gt, npan)
    Psamp, _, _ = _map(kind[fs], o[fs], e1[fs], e2[fs], e3[fs], R[fs], S, T_)
    Psamp = Psamp.reshape(npan, gs.size, 3)
    pc = Psamp.mean(1)
    pr = np.linalg.norm(Psamp - pc[:, None], axis=2).max(1) * 1.05
    Lp = 2.0 * pr
    pk = kind[pf]
    Ls = np.where(pk == RECT, ps1 - ps0, np.where(pk == ANNSEC, ps1 - ps0, R[pf] * (ps1 - ps0)))
    Lt = np.where(pk == RECT, pt1 - pt0, np.where(pk == ANNSEC, ps1 * (pt1 - pt0), pt1 - pt0))

    # ---- pair classification ---------------------------------------------------------
    ns = src.size
    d_lb = np.maximum(0.0, np.linalg.norm(pc[:, None, :] - system.bs_c[src][None], axis=2) - pr[:, None] - system.bs_r[src][None])
    dist = d_lb.copy()
    close = np.nonzero(d_lb < 2.0 * Lp[:, None])
    if close[0].size:
        ip, ik = close
        rows = np.repeat(np.arange(ip.size), gs.size)
        Pq = Psamp[ip].reshape(-1, 3)
        kk = src[ik][rows]
        sd = _cell_sdf(Pq, system.kind[kk], system.pos[kk], system.dims[kk], system.rot[kk]).reshape(ip.size, gs.size)
        dist[ip, ik] = np.maximum(0.0, sd.min(1) - 0.09 * Lp[ip])  # minus the sampling resolution
    ratio = dist / Lp[:, None]

    # aligned cuboid pairs (closed form for the parallel charges)
    tgt_cub = isinstance(mag.shape, Cuboid)
    aligned_body = np.zeros(len(system.magnets), dtype=bool)
    coaxial_body = np.zeros(len(system.magnets), dtype=bool)
    perm = {}
    for j, mj in enumerate(system.magnets):
        if j == b:
            continue
        M = Rt.T @ mj.pose.matrix
        A = np.abs(M)
        if np.all((np.abs(A - 1.0) < 1e-12) | (A < 1e-12)) and np.allclose(A.sum(0), 1.0) and np.allclose(A.sum(1), 1.0):
            aligned_body[j] = True
            perm[j] = A
        if not tgt_cub and not isinstance(mj.shape, Cuboid):
            at, aj = Rt[:, 2], mj.pose.matrix[:, 2]
            off = mj.pose.position - mag.pose.position
            off -= (off @ at) * at
            if abs(abs(at @ aj) - 1.0) < 1e-12 and np.linalg.norm(off) < 1e-9 * ell:
                coaxial_body[j] = True
    sbody = system.body[src]
    cf_ok = (tgt_cub & (bf.axis_id[pf] >= 0))[:, None] & (system.kind[src] == CUBOID)[None, :] & aligned_body[sbody][None, :]
    cf_ok &= dist < 30.0 * np.maximum(Lp[:, None], 2.0 * system.bs_r[src][None, :])
    near = ratio < _NEAR
    cf_ok &= near  # elsewhere Gauss-Legendre is already at quad_tol: no need to split the source charges
    if not closed_form:
        cf_ok[:] = False
    aligned_pair = cf_ok | (tgt_cub & aligned_body[sbody][None, :] & (system.kind[src] == CUBOID)[None, :]) | \
        ((not tgt_cub) & coaxial_body[sbody])[None, :]

    # ---- closed-form parallel forces ------------------------------------------------------
    F_cf_face = np.zeros((bf.kind.size, 3))
    if np.any(cf_ok):
        ip, ik = np.nonzero(cf_ok)
        f = pf[ip]
        kcell = src[ik]
        a = bf.axis_id[f]
        bb, cc = (a + 1) % 3, (a + 2) % 3
        posb = mag.pose.to_body(system.pos[kcell]) / ell  # source centre in target body frame
        Ab = np.stack([perm[j] for j in system.body[kcell]])  # |R_t^T R_s|
        hdim = np.einsum("nij,nj->ni", Ab, system.dims[kcell, :3]) * 0.5 / ell
        Jb = sol.J[kcell] @ Rt  # source J in target body frame
        dJb = dJ[f] @ Rt
        ar = np.arange(f.size)
        sig_t = dJb[ar, a]
        Jsa = Jb[ar, a]
        side = np.where(bf.minus[f] < 0, 1.0, np.where(bf.plus[f] < 0, -1.0, 1.0))
        zt = bf.o[f, a] / ell
        tsr = np.stack([ps0[ip], ps1[ip]], 1) / ell
        ttr = np.stack([pt0[ip], pt1[ip]], 1) / ell
        ssr = np.stack([posb[ar, bb] - hdim[ar, bb], posb[ar, bb] + hdim[ar, bb]], 1)
        str_ = np.stack([posb[ar, cc] - hdim[ar, cc], posb[ar, cc] + hdim[ar, cc]], 1)
        Floc = np.zeros((f.size, 3))
        for sgn in (1.0, -1.0):
            zs = posb[ar, a] + sgn * hdim[ar, a]
            Iv = rect_pair_integral(tsr, ttr, ssr, str_, zt - zs, side, w_tol=1e-12)
            Floc += (sig_t * sgn * Jsa)[:, None] * Iv
        Floc *= ell**2 / (4.0 * np.pi * MU0)
        # local (b, c, a) components -> body vector -> global
        vb = np.zeros((f.size, 3))
        vb[ar, bb] = Floc[:, 0]
        vb[ar, cc] = Floc[:, 1]
        vb[ar, a] = Floc[:, 2]
        np.add.at(F_cf_face, f, vb @ Rt.T)

    # ---- quadrature node generation -----------------------------------------------------------
    # every node: (panel, source cell, s, t, weight) with s, t, weight on the unit square
    nodes = []
    far = ~near

    ip, ik = np.nonzero(far)
    if ip.size:
        qa = _q_far(dist[ip, ik] / Ls[ip], quad_tol, level)
        qb = _q_far(dist[ip, ik] / Lt[ip], quad_tol, level)
        key = qa * 100 + qb
        for kq in np.unique(key):
            m = key == kq
            xa, wa = _gl(kq // 100)
            xb, wb = _gl(kq % 100)
            X, Y = np.meshgrid(xa, xb, indexing="ij")
            W = np.outer(wa, wb).ravel()
            X, Y = X.ravel(), Y.ravel()
            pi, ki = ip[m], ik[m]
            nodes.append((np.repeat(pi, X.size), np.repeat(ki, X.size), np.tile(X, pi.size), np.tile(Y, pi.size),
                          np.tile(W, pi.size)))
    ip, ik = np.nonzero(near)
    if ip.size:
        tx, tw = _ts(level)
        kc = src[ik]
        feat, phi_ok = _cell_features(system.kind[kc], system.pos[kc], system.dims[kc], system.rot[kc])
        f = pf[ip]
        fr = np.repeat(f, 8)
        fs_, ft_ = _param(kind[fr], o[fr], e1[fr], e2[fr], e3[fr], feat.reshape(-1, 3))
        fs_, ft_ = fs_.reshape(-1, 8), ft_.reshape(-1, 8)
        for n_ in range(ip.size):
            p, kfull = ip[n_], ik[n_]
            fk = kind[pf[p]]
            lims = [(ps0[p], ps1[p]), (pt0[p], pt1[p])]
            per_dir = []
            for dirn, (lo, hi) in enumerate(lims):
                vals = fs_[n_] if dirn == 0 else ft_[n_]
                is_angle = (fk == ANNSEC and dirn == 1) or (fk == CYLPATCH and dirn == 0)
                if is_angle:
                    if not phi_ok[n_]:
                        vals = np.array([])
                    else:
                        vals = lo + np.mod(vals - lo, 2 * np.pi)
                span = hi - lo
                v = np.unique(vals[(vals > lo + 1e-9 * span) & (vals < hi - 1e-9 * span)])
                if v.size > 1:
                    keep = np.concatenate([[True], np.diff(v) > 1e-9 * span])
                    v = v[keep]
                br = np.concatenate([[lo], v[:12], [hi]])
                if not aligned_pair[p, kfull]:
                    dd = max(dist[p, kfull], 0.01 * Lp[p])
                    fine = []
                    for x0, x1 in zip(br[:-1], br[1:]):
                        L1 = (x1 - x0) * (Lp[p] / max(span, 1e-300))
                        mpc = int(np.clip(np.ceil(L1 / (2.0 * dd)), 1, 8))
                        fine.append(np.linspace(x0, x1, mpc + 1)[:-1])
                    br = np.concatenate(fine + [[hi]])
                xs = ((br[:-1, None] + (br[1:] - br[:-1])[:, None] * tx[None, :]).ravel() - lo) / span
                ws = ((br[1:] - br[:-1])[:, None] * tw[None, :]).ravel() / span
                per_dir.append((xs, ws))
            (xs, wsx), (ys, wsy) = per_dir
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            W = np.outer(wsx, wsy).ravel()
            nodes.append((np.full(W.size, p), np.full(W.size, kfull), X.ravel(), Y.ravel(), W))

    pan = np.concatenate([n[0] for n in nodes])
    kix = np.concatenate([n[1] for n in nodes])
    U = np.concatenate([n[2] for n in nodes])
    Vv = np.concatenate([n[3] for n in nodes])
    Wq = np.concatenate([n[4] for n in nodes])
    n_pairs = {"gauss": int(far.sum()), "near": int(near.sum()), "closed_form": int(cf_ok.sum())}

    # ---- evaluate ---------------------------------------------------------------------------
    F_full = np.zeros(3)
    F_q_par_face = np.zeros((bf.kind.size, 3))
    T_q = np.zeros(3)
    n_eval = 0
    n_bad = 0

    def field(Pe, cell, pol):
        # magpylib's cuboid kernel cancels catastrophically within ~1e-8 ell of a source edge
        # (0 * inf -> nan). Only tanh-sinh nodes next to a touching edge get there; their weights are
        # ~1e-15, so dropping them is far below the quadrature tolerance. They are counted.
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            H = cell_field(Pe, system.kind[cell], system.pos[cell], system.dims[cell], system.rot[cell], pol, ell)
        bad = ~np.all(np.isfinite(H), axis=1)
        H[bad] = 0.0
        return H, int(bad.sum())

    for c0 in range(0, pan.size, _NODE_BATCH):
        sl = slice(c0, c0 + _NODE_BATCH)
        p, kk = pan[sl], kix[sl]
        f = pf[p]
        s = ps0[p] + (ps1 - ps0)[p] * U[sl]
        t = pt0[p] + (pt1 - pt0)[p] * Vv[sl]
        P, nrm, jac = _map(kind[f], o[f], e1[f], e2[f], e3[f], R[f], s, t)
        w = Wq[sl] * (ps1 - ps0)[p] * (pt1 - pt0)[p] * jac
        sig = (dJ[f] * nrm).sum(1)
        Pe = P + (1e-10 * ell) * inward[f, None] * nrm
        cell = src[kk]
        cfm = cf_ok[p, kk]
        Jk = sol.J[cell]
        # split polarisation for closed-form pairs
        Jpar = np.where(cfm[:, None], (Jk * e3[f]).sum(1)[:, None] * e3[f], 0.0)
        Jq = Jk - Jpar  # full for ordinary pairs, perpendicular part for cf pairs
        H, nb = field(Pe, cell, Jq)
        n_bad += nb
        dF = (sig * w)[:, None] * H
        F_full += dF.sum(0)
        if torque:
            T_q += np.cross(P - pivot, dF).sum(0)
            if np.any(cfm):
                m = cfm
                Hp, nb = field(Pe[m], cell[m], Jpar[m])
                n_bad += nb
                dFp = (sig[m] * w[m])[:, None] * Hp
                T_q += np.cross(P[m] - pivot, dFp).sum(0)
                np.add.at(F_q_par_face, f[m], dFp)
        n_eval += p.size
    n_pairs["nonfinite_nodes"] = n_bad
    F = F_full + F_cf_face.sum(0)
    T = T_q
    if torque and np.any(cf_ok):
        cf_faces = np.unique(pf[np.nonzero(cf_ok)[0]])
        fc = o[cf_faces] + 0.5 * (rng[cf_faces, 0] + rng[cf_faces, 1])[:, None] * e1[cf_faces] + \
            0.5 * (rng[cf_faces, 2] + rng[cf_faces, 3])[:, None] * e2[cf_faces]
        T = T + np.cross(fc - pivot, F_cf_face[cf_faces] - F_q_par_face[cf_faces]).sum(0)
    return ForceResult(F=F, T=T if torque else np.full(3, np.nan), pivot=pivot, n_nodes=int(n_eval), n_pairs=n_pairs)
