"""Per-cell scalar Preisach hysteresis along the easy axis plus a reversible part (task 2).

Material law (docs/SPEC.md "Interface contracts"), with H_par = H.e and H_perp the two components
orthogonal to the easy axis e:

    J_par  = Js_irr(T) * P(H_par; history) + MU0 * (mu_rec - 1) * H_par
    J_perp = MU0 * (mu_perp - 1) * H_perp

P in [-1, 1] is the classical scalar Preisach output [mayergoyz1991]: the signed measure of the
relay hysterons gamma_{alpha,beta} (switch up at alpha, down at beta, alpha >= beta) that are "up".
It is evaluated through the Everett function

    E(alpha, beta) = mass of hysterons with beta <= beta' <= alpha' <= alpha

and the alternating-extrema (staircase) memory with the wiping-out property [mayergoyz1991, ch. 1]:
for the input history base -> t_1 -> ... -> t_k -> h (base = +inf or -inf saturation),

    P = s0 + 2 * sum_j sign(y_j - y_{j-1}) * E(max(y_j, y_{j-1}), min(y_j, y_{j-1})).

Distribution (justification in `PreisachDistribution`): product of a truncated log-normal
switching-field density f(Hc) and an interaction-field density g(Hu) (Lorentzian for cast Alnico,
Gaussian for sintered/bonded NdFeB), Hc = (alpha - beta)/2, Hu = (alpha + beta)/2. Then

    E(alpha, beta) = int_0^{(alpha-beta)/2} f(Hc) [G(alpha - Hc) - G(beta + Hc)] dHc,

with G the CDF of Hu. Fields inside the model are reduced, h = H / H_scale(T), so a temperature
change rescales the field axis without touching the stored history.

All fields in A/m, polarisations in T, temperatures in K (docs/SPEC.md "Conventions").
"""

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from magsim.constants import MU0

# Truncation of the log-normal Hc density at |ln(Hc/h0)/s| <= Z_MAX. Part of the model definition
# (the density is renormalised), so it is not an approximation error: switching fields below
# h0*exp(-4s) or above h0*exp(4s) simply do not exist in the model.
Z_MAX = 4.0
# Gauss-Legendre nodes per panel; three panels per Everett evaluation (see `everett`).
N_PANEL = 24
_GL_T, _GL_W = np.polynomial.legendre.leggauss(N_PANEL)
_PHI_NORM = special.ndtr(Z_MAX) - special.ndtr(-Z_MAX)
_SQRT2PI = np.sqrt(2.0 * np.pi)


class PreisachDistribution:
    """Preisach density in (Hc, Hu) = ((alpha - beta)/2, (alpha + beta)/2), reduced field units.

    mu(alpha, beta) dalpha dbeta = f(Hc) g(Hu) dHc dHu, normalised so E(+inf, -inf) = 1.

    f: log-normal switching-field density, median ``h0``, log-standard-deviation ``s``, truncated at
       |z| <= Z_MAX standard deviations and renormalised (optionally skewed, see ``skew``). Positive support (Hc > 0 is required of a relay);
       for small s it is indistinguishable from a Gaussian of relative width s, which suits the
       narrow switching-field distribution of sintered NdFeB (loop squareness from Hk/Hcj), while
       larger s gives the broad, right-skewed spread of Alnico rods / isotropic bonded powders.
    g: interaction-field density of width ``w`` (``kind``):
       "lorentz" (Cauchy, half-width w): heavy symmetric tails give the gradual shoulder of cast
       Alnico demagnetisation curves. Identified on the FEMM LNG37 curve it gives RMS 1.6 mT, against
       12.7 mT for a Gaussian x Gaussian product (prototype fit, task 2 report).
       "gauss" (standard deviation w): light tails keep the sintered-NdFeB curve straight (purely
       reversible) up to the knee, as datasheet curves show.
    Both CDFs are closed form (arctan / ndtr), so E needs a single 1-D quadrature over Hc.
    skew: Tukey-g shape of ln Hc, Hc = h0 exp(s (exp(skew z) - 1)/skew), z ~ N(0, 1) truncated at |z| <= Z_MAX
       (skew = 0: log-normal). Lets the identification shape the knee asymmetrically (the FEMM LNG37 fit
       drops from RMS 3.0 mT to 1.0 mT with it); the quadrature in z is unchanged.
    """

    def __init__(self, h0: float, s: float, w: float, kind: str = "lorentz", skew: float = 0.0):
        for label, v in (("h0", h0), ("s", s), ("w", w)):
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{label} must be finite and > 0, got {v}")
        if kind not in ("lorentz", "gauss"):
            raise ValueError(f"kind must be 'lorentz' or 'gauss', got {kind!r}")
        if not np.isfinite(skew) or abs(skew) > 2.0:
            raise ValueError(f"skew must be finite with |skew| <= 2, got {skew}")
        self.h0, self.s, self.w, self.kind, self.skew = float(h0), float(s), float(w), kind, float(skew)

    def __repr__(self) -> str:
        return (f"PreisachDistribution(h0={self.h0:.6g}, s={self.s:.6g}, w={self.w:.6g}, kind={self.kind!r}, "
                f"skew={self.skew:.6g})")

    def as_dict(self) -> dict:
        return {"h0": self.h0, "s": self.s, "w": self.w, "kind": self.kind, "skew": self.skew}

    def hc_of_z(self, z):
        """Switching field Hc(z) = h0 exp(s (exp(skew z) - 1)/skew) (-> h0 exp(s z) for skew = 0)."""
        z = np.asarray(z, dtype=float)
        if self.skew == 0.0:
            return self.h0 * np.exp(self.s * z)
        return self.h0 * np.exp(self.s * np.expm1(self.skew * z) / self.skew)

    def z_of_hc(self, v):
        """Inverse of `hc_of_z` for v > 0 (+-inf beyond the bounded tail); -inf for v <= 0."""
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            L = np.log(np.where(v > 0, v, 1.0) / self.h0) / self.s
            if self.skew == 0.0:
                z = L
            else:
                arg = self.skew * L
                z = np.where(arg > -1.0, np.log1p(np.maximum(arg, -1.0 + 1e-300)) / self.skew,
                             np.inf if self.skew < 0 else -np.inf)
        return np.where(v > 0, z, -np.inf)

    @property
    def hc_range(self) -> tuple[float, float]:
        """Support of the truncated switching-field density (reduced units)."""
        return float(self.hc_of_z(-Z_MAX)), float(self.hc_of_z(Z_MAX))

    def G(self, y):
        """CDF of Hu."""
        if self.kind == "lorentz":
            return 0.5 + np.arctan(y / self.w) / np.pi
        return special.ndtr(y / self.w)

    def g(self, y):
        """Density of Hu."""
        u = y / self.w
        if self.kind == "lorentz":
            return 1.0 / (np.pi * self.w * (1.0 + u * u))
        return np.exp(-0.5 * u * u) / (self.w * _SQRT2PI)

    def everett(self, alpha, beta, which=None):
        """Everett function E(alpha, beta) [mayergoyz1991] and optionally one partial derivative.

        alpha >= beta (reduced fields, broadcastable arrays; +inf / -inf allowed for saturation).
        which: None, or an array of +1 / -1 per element selecting dE/dalpha / dE/dbeta.
        Returns E, or (E, dE).

        Quadrature: substitute Hc = hc_of_z(z) so that f(Hc) dHc = phi(z) dz / norm on
        z in [-Z_MAX, min(z_c, Z_MAX)], z_c = z_of_hc((alpha - beta)/2). The interval is split at
        the centres of the two G terms (alpha - Hc = 0 and beta + Hc = 0), where the integrand varies
        fastest, and each of the three panels gets N_PANEL Gauss-Legendre nodes. The integrand
        vanishes at Hc = (alpha - beta)/2, so the moving upper limit introduces no kink, E is a
        continuous function of (alpha, beta), and the derivatives
            dE/dalpha =  int f(Hc) g(alpha - Hc) dHc,   dE/dbeta = -int f(Hc) g(beta + Hc) dHc
        (boundary terms vanish) use the same nodes. Measured against adaptive quadrature over the parameter
        range the identification allows (w/(h0 s) >= 0.25, |skew| <= 0.5, support within [h0/10, 3 h0]):
        |E error| < 1e-6 and dE error < 5e-4 of max|dE| (tests/test_preisach.py, tests/test_library.py).
        """
        alpha, beta = np.broadcast_arrays(np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float))
        shape = alpha.shape
        a = alpha.ravel()
        b = beta.ravel()
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            c = 0.5 * (a - b)
            zc = self.z_of_hc(c)
        E = np.zeros(a.shape)
        dE = np.zeros(a.shape)
        act = np.flatnonzero(zc > -Z_MAX)  # empty interval (all Hc above (alpha-beta)/2): E = dE = 0
        if act.size:
            a, b, zc = a[act], b[act], zc[act]
            wsel = None if which is None else np.broadcast_to(np.asarray(which), shape).ravel()[act]
            E[act], dE_act = self._everett_active(a, b, zc, wsel)
            if which is not None:
                dE[act] = dE_act
        E = E.reshape(shape)
        if which is None:
            return E
        return E, dE.reshape(shape)

    def _everett_active(self, a, b, zc, which):
        """Quadrature of `everett` on 1-D arrays with a non-empty Hc interval.

        Panels: [-Z_MAX, hi] split at the G-term centres p1 = ln(alpha/h0)/s, p2 = ln(-beta/h0)/s.
        Zero-length panels are dropped, so cells are grouped by their number of panels (1-3) and by
        whether alpha = +inf / beta = -inf (constant G term skipped); every group is vectorised.
        """
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p1 = self.z_of_hc(a)
            p2 = self.z_of_hc(-b)
        lo = -Z_MAX
        hi = np.minimum(zc, Z_MAX)
        p1 = np.clip(p1, lo, hi)
        p2 = np.clip(p2, lo, hi)
        edges = np.stack([np.full_like(hi, lo), np.minimum(p1, p2), np.maximum(p1, p2), hi], axis=-1)
        live = edges[:, 1:] > edges[:, :-1]  # (m, 3) non-degenerate panels
        npan = live.sum(axis=1)
        key = npan * 4 + 2 * np.isposinf(a) + np.isneginf(b)
        E = np.zeros(a.shape)
        dE = None if which is None else np.zeros(a.shape)
        for kval in np.unique(key):
            rows = np.flatnonzero(key == kval)
            npk = int(kval // 4)
            if npk == 0:
                continue
            # keep the lower edge plus the upper edge of every live panel
            keep = np.concatenate([np.ones((rows.size, 1), bool), live[rows]], axis=1)
            ek = edges[rows][keep].reshape(rows.size, npk + 1)
            wk = None if which is None else which[rows]
            Ek, dEk = self._quad(a[rows], b[rows], ek, wk, bool(kval & 2), bool(kval & 1))
            E[rows] = Ek
            if which is not None:
                dE[rows] = dEk
        return E, dE

    def _quad(self, a, b, edges, which, a_inf, b_inf):
        """Gauss-Legendre over the panels `edges` (m, p+1) of one group of cells."""
        half = 0.5 * (edges[:, 1:] - edges[:, :-1])  # (m, p)
        mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
        # Nodes (m, p, N); temporaries are reused in place (this is the hot loop of `response`).
        z = half[..., None] * _GL_T
        z += mid[..., None]
        wz = z * z
        wz *= -0.5
        np.exp(wz, out=wz)
        wz *= half[..., None] * (_GL_W / (_SQRT2PI * _PHI_NORM))
        if self.skew == 0.0:
            x = z
            x *= self.s
        else:
            x = z * self.skew
            np.expm1(x, out=x)
            x *= self.s / self.skew
        np.exp(x, out=x)
        x *= self.h0
        inv_w = 1.0 / self.w
        lor = self.kind == "lorentz"
        # dG = G(alpha - Hc) - G(beta + Hc), reduced arguments ya, yb (constant terms skipped)
        ya = None if a_inf else (a[:, None, None] - x) * inv_w
        yb = None if b_inf else (x + b[:, None, None]) * inv_w
        if lor:
            dG = (0.5 if a_inf else np.arctan(ya) / np.pi) - (-0.5 if b_inf else np.arctan(yb) / np.pi)
        else:
            dG = (1.0 if a_inf else special.ndtr(ya)) - (0.0 if b_inf else special.ndtr(yb))
        dG = dG * wz
        E = dG.sum(axis=(1, 2))
        if which is None:
            return E, None
        up = which > 0
        if a_inf:
            u = yb.copy()
        elif b_inf:
            u = ya.copy()
        else:
            u = np.where(up[:, None, None], ya, yb)  # reduced argument of g
        u *= u
        if lor:
            u += 1.0
            np.divide(wz, u, out=u)
            u *= 1.0 / (np.pi * self.w)
        else:
            u *= -0.5
            np.exp(u, out=u)
            u *= wz
            u *= 1.0 / (self.w * _SQRT2PI)
        dE = u.sum(axis=(1, 2))
        if a_inf:  # E(+inf, beta) does not depend on alpha
            return E, np.where(up, 0.0, -dE)
        if b_inf:  # E(alpha, -inf) does not depend on beta
            return E, np.where(up, dE, 0.0)
        return E, np.where(up, 1.0, -1.0) * dE


@dataclass(frozen=True)
class PreisachState:
    """History of n cells (reduced field units; arrays are never modified in place).

    s0     (n,)          base: +1 = history starts at +inf (positive saturation), -1 = at -inf.
    k      (n,)          number of stored turning points.
    t      (n, depth)    alternating turning points t_1..t_k (NaN beyond k), nested (wiping-out).
    Pc     (n, depth+1)  P at the base (column 0 = s0) and at each turning point (cached at commit).
    h_last (n,)          last committed input (+-inf before the first commit).
    P_last (n,)          P at h_last.
    """

    s0: np.ndarray
    k: np.ndarray
    t: np.ndarray
    Pc: np.ndarray
    h_last: np.ndarray
    P_last: np.ndarray

    @property
    def n(self) -> int:
        return self.k.shape[0]

    @property
    def P(self) -> np.ndarray:
        """Irreversible fraction J_irr/Js_irr at the last committed input, (n,)."""
        return self.P_last.copy()


class PreisachMaterial:
    """Scalar Preisach hard magnet implementing the `Material` protocol (materials/base.py).

    Parameters
    ----------
    Br : remanence on the descending major branch at T_ref [T]; Js_irr = Br / P_r with
         P_r = P on that branch at H = 0 (so J(H=0) = Br exactly).
    dist : PreisachDistribution in reduced fields h = H / H_scale.
    H_scale : field unit of the distribution at T_ref [A/m] (the library uses the datasheet Hcj).
    mu_rec : reversible relative permeability along the easy axis: the recoil slope dB/dH at a
             reversal point equals MU0*mu_rec exactly (the Everett term has zero slope there).
    mu_perp : relative permeability across the easy axis (linear, history-free).
    alpha_Br, beta_Hcj : reversible temperature coefficients [1/K]:
             Br(T) = Br (1 + alpha_Br (T - T_ref)) scales Js_irr;
             Hcj(T) = Hcj (1 + beta_Hcj (T - T_ref)) scales H_scale (the field axis of the distribution).
             History is stored in reduced units, so a temperature change keeps every hysteron's state;
             a later commit at the same H but a different T moves the reduced input H/H_scale(T) and
             can switch hysterons (coercivity-loss-driven irreversible change). Thermally activated
             (viscous) loss is a non-goal (docs/SPEC.md).
    depth : maximum number of stored turning points per cell (memory is (n, depth) arrays).

    Stack overflow: when a reversal would push turning point depth+1, the innermost reversal is not
    stored; the cell keeps following the branch from t_depth (the excursion t_depth -> h_last -> h is
    treated as reversible). J stays continuous; the error in P is at most 2*E(t_depth <-> h_last), the
    Everett weight of that innermost minor loop, and vanishes once the input returns to h_last's
    branch end or passes t_depth (tested in tests/test_preisach.py).
    """

    def __init__(self, Br: float, dist: PreisachDistribution, H_scale: float, mu_rec: float = 1.05,
                 mu_perp: float = 1.05, alpha_Br: float = 0.0, beta_Hcj: float = 0.0,
                 T_ref: float = 293.15, name: str = "preisach", depth: int = 32):
        if not isinstance(dist, PreisachDistribution):
            raise TypeError("dist must be a PreisachDistribution")
        if not np.isfinite(Br) or Br <= 0:
            raise ValueError(f"Br must be finite and > 0, got {Br}")
        if not np.isfinite(H_scale) or H_scale <= 0:
            raise ValueError(f"H_scale must be finite and > 0, got {H_scale}")
        for label, mu in (("mu_rec", mu_rec), ("mu_perp", mu_perp)):
            if not np.isfinite(mu) or mu < 1.0:
                raise ValueError(f"{label} must be finite and >= 1, got {mu}")
        for label, v in (("alpha_Br", alpha_Br), ("beta_Hcj", beta_Hcj)):
            if not np.isfinite(v):
                raise ValueError(f"{label} must be finite, got {v}")
        if not np.isfinite(T_ref) or T_ref <= 0:
            raise ValueError(f"T_ref must be finite and > 0 [K], got {T_ref}")
        if not isinstance(depth, (int, np.integer)) or depth < 2:
            raise ValueError(f"depth must be an integer >= 2, got {depth!r}")
        self.dist = dist
        self.Br = float(Br)
        self.H_scale = float(H_scale)
        self.mu_rec = float(mu_rec)
        self.mu_perp = float(mu_perp)
        self.alpha_Br = float(alpha_Br)
        self.beta_Hcj = float(beta_Hcj)
        self.T_ref = float(T_ref)
        self.name = name
        self.depth = int(depth)
        # Remanent fraction on the descending major branch: P_r = 1 - 2 E(+inf, 0).
        self.P_r = float(1.0 - 2.0 * dist.everett(np.inf, 0.0))
        self.Js_irr = self.Br / self.P_r
        self._init_cache: dict = {}

    def __repr__(self) -> str:
        return (f"PreisachMaterial(name={self.name!r}, Br={self.Br:.6g}, H_scale={self.H_scale:.6g}, "
                f"mu_rec={self.mu_rec:.6g}, mu_perp={self.mu_perp:.6g}, {self.dist!r})")

    # ------------------------------------------------------------------ temperature scaling
    def H_scale_at(self, T) -> np.ndarray:
        """Field unit at temperature T [A/m]: H_scale (1 + beta_Hcj (T - T_ref))."""
        S = self.H_scale * (1.0 + self.beta_Hcj * (np.asarray(T, dtype=float) - self.T_ref))
        if np.any(S <= 0):
            raise ValueError("temperature outside the range where Hcj(T) > 0")
        return S

    def Js_at(self, T) -> np.ndarray:
        """Js_irr(T) = Js_irr (1 + alpha_Br (T - T_ref)) [T]."""
        Js = self.Js_irr * (1.0 + self.alpha_Br * (np.asarray(T, dtype=float) - self.T_ref))
        if np.any(Js <= 0):
            raise ValueError("temperature outside the range where Br(T) > 0")
        return Js

    # ------------------------------------------------------------------ validation
    def _check(self, H_par, H_perp, state: PreisachState, T):
        if not isinstance(state, PreisachState):
            raise TypeError("state must be a PreisachState from init_state/commit")
        n = state.n
        H_par = np.asarray(H_par, dtype=float)
        if H_par.shape != (n,):
            raise ValueError(f"H_par must have shape ({n},), got {H_par.shape}")
        if H_perp is not None:
            H_perp = np.asarray(H_perp, dtype=float)
            if H_perp.shape != (n, 2):
                raise ValueError(f"H_perp must have shape ({n}, 2), got {H_perp.shape}")
            if not np.all(np.isfinite(H_perp)):
                raise ValueError("H_par and H_perp must be finite")
        if not np.all(np.isfinite(H_par)):
            raise ValueError("H_par and H_perp must be finite")
        T = np.asarray(T, dtype=float)
        if T.shape not in ((), (n,)):
            raise ValueError(f"T must be a scalar or shape ({n},), got {T.shape}")
        if not np.all(np.isfinite(T)) or np.any(T <= 0):
            raise ValueError("T must be finite and positive [K]")
        return H_par, H_perp, T

    # ------------------------------------------------------------------ core history step
    def _step(self, h: np.ndarray, st: PreisachState, build: bool):
        """P(h), dP/dh for the monotone segment h_last -> h; new state if build."""
        n, D = st.n, self.depth
        idx = np.arange(n)
        k = st.k
        km1 = np.maximum(k - 1, 0)
        t_prev = np.where(k > 0, st.t[idx, km1], st.s0 * np.inf)
        with np.errstate(invalid="ignore"):
            d_last = np.where(k > 0, np.sign(st.h_last - t_prev), -st.s0)
            dh = h - st.h_last
        rev = np.sign(dh) == -d_last  # dh == 0 -> continuation (same branch, same P)
        d = np.where(rev, -d_last, d_last)
        kk = k + rev
        # Augmented history: a reversal turns h_last into a provisional turning point.
        A = np.full((n, D + 1), np.nan)
        A[:, :D] = st.t
        Pa = np.full((n, D + 2), np.nan)
        Pa[:, : D + 1] = st.Pc
        r = np.flatnonzero(rev)
        A[r, k[r]] = st.h_last[r]
        Pa[r, k[r] + 1] = st.P_last[r]
        # Wiping-out: drop pairs whose same-type extremum (positions kk-2, kk-4, ...) h reaches.
        j = np.arange(D + 1)[None, :]
        rel = kk[:, None] - 2 - j
        chk = (rel >= 0) & (rel % 2 == 0)
        with np.errstate(invalid="ignore"):
            passed = d[:, None] * (h[:, None] - A) >= 0
        m = np.sum(chk & passed, axis=1)
        kk2 = kk - 2 * m
        ovf = kk2 > D  # stack full: forget the innermost reversal (see class docstring)
        kk2 = np.where(ovf, D, kk2)
        d = np.where(ovf, d_last, d)
        start = np.where(kk2 > 0, A[idx, np.maximum(kk2 - 1, 0)], st.s0 * np.inf)
        alpha = np.where(d > 0, h, start)
        beta = np.where(d > 0, start, h)
        E, dE = self.dist.everett(alpha, beta, which=d)
        P = Pa[idx, kk2] + 2.0 * d * E
        dPdh = 2.0 * d * dE
        if not build:
            return P, dPdh, None
        t_new = A[:, :D].copy()
        t_new[np.arange(D)[None, :] >= kk2[:, None]] = np.nan
        Pc_new = Pa[:, : D + 1].copy()
        Pc_new[np.arange(D + 1)[None, :] > kk2[:, None]] = np.nan
        new = PreisachState(s0=st.s0.copy(), k=kk2.astype(np.int64), t=t_new, Pc=Pc_new,
                            h_last=h.copy(), P_last=P)
        return P, dPdh, new

    # ------------------------------------------------------------------ Material protocol
    def init_state(self, n: int, initial: str | float) -> PreisachState:
        """Initial history for n cells.

        "+sat" / "-sat": base +inf / -inf, nothing committed yet, so the first response at any H is
        the descending / ascending major branch.
        "thermal": demagnetised state, the Preisach staircase along alpha = -beta [mayergoyz1991]
        represented by K = min(16, depth // 2) decaying alternating turning points from +inf
        (geometric between 5*h_max and 0.95*h_min of the Hc support; below h_min the staircase holds
        no hysterons), resting at h = 0. The last turning point is tuned by bisection so that P = 0
        exactly at h = 0 (the finite staircase otherwise leaves |P| ~ 1e-3). Leaves depth - K slots.
        float P0 in [-1, 1] (= J_irr/Js_irr): with P_r the remanent fraction of the major loop,
          |P0| <= P_r: recoil state at H = 0, base sign(P0)*inf, reversal at the field h* found by
                       bisection so that P = P0 after returning to H = 0;
          P_r < |P0| < 1: no state at H = 0 can reach it (all hysterons with beta' > 0 are down at
                       h = 0), so the state sits on the major branch at the field h* (beyond 0) where
                       P = P0; P0 = +-1 is identical to "+-sat".
        """
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n!r}")
        if isinstance(initial, str):
            if initial not in ("+sat", "-sat", "thermal"):
                raise ValueError(f"initial must be '+sat', '-sat', 'thermal' or a float in [-1, 1], got {initial!r}")
            key = initial
        else:
            P0 = float(initial)
            if not np.isfinite(P0) or not -1.0 <= P0 <= 1.0:
                raise ValueError(f"initial must lie in [-1, 1], got {initial}")
            key = "+sat" if P0 == 1.0 else "-sat" if P0 == -1.0 else P0
        if key not in self._init_cache:
            self._init_cache[key] = self._build_initial(key)
        one = self._init_cache[key]
        rep = lambda a: np.repeat(a, int(n), axis=0)  # noqa: E731
        return PreisachState(s0=rep(one.s0), k=rep(one.k), t=rep(one.t), Pc=rep(one.Pc),
                             h_last=rep(one.h_last), P_last=rep(one.P_last))

    def _base(self, s0: float) -> PreisachState:
        D = self.depth
        Pc = np.full((1, D + 1), np.nan)
        Pc[0, 0] = s0
        return PreisachState(s0=np.array([s0]), k=np.array([0], dtype=np.int64), t=np.full((1, D), np.nan),
                             Pc=Pc, h_last=np.array([s0 * np.inf]), P_last=np.array([s0]))

    def _walk(self, st: PreisachState, path) -> PreisachState:
        for h in path:
            st = self._step(np.array([float(h)]), st, build=True)[2]
        return st

    def _build_initial(self, key) -> PreisachState:
        if key == "+sat":
            return self._base(1.0)
        if key == "-sat":
            return self._base(-1.0)
        if key == "thermal":
            return self._thermal()
        P0 = float(key)
        sgn = 1.0 if P0 >= 0 else -1.0
        target = abs(P0)  # build the mirror image for P0 < 0
        base = self._base(sgn)

        def P_after(path):
            st = self._walk(base, [sgn * p for p in path])
            return sgn * st.P_last[0], st

        if target <= self.P_r:
            # recoil: descend (in the mirrored frame) to -x, return to 0; P rises from -P_r.. to P_r
            def f(x):
                return P_after([-x, 0.0])[0] - target
            hi = self.dist.hc_range[1] + 10 * self.dist.w
            while f(hi) > 0:
                hi *= 2.0
            x = optimize.brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-12) if f(0.0) > 0 else 0.0
            return P_after([-x, 0.0])[1] if x > 0 else P_after([0.0])[1]

        # P_r < target < 1: on the major branch at a field beyond 0
        def f(x):
            return P_after([x])[0] - target
        hi = self.dist.hc_range[1] + 10 * self.dist.w
        while f(hi) < 0:
            hi *= 2.0
        x = optimize.brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-12)
        return P_after([x])[1]

    def _thermal(self) -> PreisachState:
        K = min(16, self.depth // 2)
        lo_hc, hi_hc = self.dist.hc_range
        w = self.dist.w
        # one large swing (no bistable hysteron has |alpha'|, |beta'| beyond 2*hi + 20 w), then the
        # remaining K - 1 decaying extrema spread over the Hc support, where the density lives
        amps = np.concatenate([[2.0 * hi_hc + 20.0 * w], np.geomspace(hi_hc + 3.0 * w, 0.95 * lo_hc, K - 1)])
        signs = -((-1.0) ** np.arange(K))  # -, +, -, ... (first move from +inf is down)

        def build(amp):
            return self._walk(self._base(1.0), list(signs * amp) + [0.0])

        # Tune one extremum inside the Hc support (innermost first, nesting kept) so that P(0) = 0.
        for j in range(K - 2, 0, -1):
            if not lo_hc < amps[j] < hi_hc:
                continue

            def f(aj, j=j):
                amp = amps.copy()
                amp[j] = aj
                return build(amp).P_last[0]
            a_lo, a_hi = amps[j + 1] * (1 + 1e-9), amps[j - 1] * (1 - 1e-9)
            f_lo, f_hi = f(a_lo), f(a_hi)
            if f_lo * f_hi < 0:
                amps[j] = optimize.brentq(f, a_lo, a_hi, xtol=1e-15, rtol=1e-13)
                break
        return build(amps)

    def response(self, H_par, H_perp, state: PreisachState, T):
        """Trial evaluation, does not modify state. See module docstring for the law.

        Returns (J_par [T] (n,), dJ_dH_par [T m/A] (n,), J_perp [T] (n, 2), chi_perp [-] (n,)).
        dJ_dH_par = Js_irr(T) dP/dh / H_scale(T) + MU0 (mu_rec - 1), with dP/dh from the analytic
        Everett derivative of the active (last) term on the current branch.
        """
        H_par, H_perp, T = self._check(H_par, H_perp, state, T)
        S = self.H_scale_at(T)
        Js = self.Js_at(T)
        P, dPdh, _ = self._step(H_par / S, state, build=False)
        n = state.n
        J_par = Js * P + MU0 * (self.mu_rec - 1.0) * H_par
        dJ = Js * dPdh / S + MU0 * (self.mu_rec - 1.0)
        J_perp = MU0 * (self.mu_perp - 1.0) * H_perp
        chi_perp = np.full(n, self.mu_perp - 1.0)
        return J_par, dJ, J_perp, chi_perp

    def commit(self, H_par, state: PreisachState, T) -> PreisachState:
        """New state after the monotone segment from the last committed H_par to H_par."""
        H_par, _, T = self._check(H_par, None, state, T)
        return self._step(H_par / self.H_scale_at(T), state, build=True)[2]

    # ------------------------------------------------------------------ major loop and datasheet values
    def major_loop(self, H, branch: str = "descending", T=None):
        """J and dJ/dH on the major loop branch ("descending" from +sat, "ascending" from -sat)."""
        T = self.T_ref if T is None else T
        H = np.atleast_1d(np.asarray(H, dtype=float))
        st = self.init_state(H.shape[0], "+sat" if branch == "descending" else "-sat")
        J, dJ, _, _ = self.response(H, np.zeros((H.shape[0], 2)), st, T)
        return J, dJ

    def characteristics(self, T=None) -> dict:
        """Datasheet quantities of the model's descending major branch at temperature T [SI].

        Br = J(0); Hcj: J = 0; Hcb: B = 0; Hk: J = 0.9 Br; (BH)max = max(-B H) in the second
        quadrant (with H_d, B_d at the optimum); mu_rec_chord = average recoil permeability from the
        (BH)max point back to H = 0, (B_recoil(0) - B_d)/(MU0 (0 - H_d)), the MMPA 0100-00
        definition of recoil permeability ("average slope of the recoil hysteresis loop").
        Fields in A/m (positive magnitudes), Br/B_d in T, (BH)max in J/m^3.
        """
        T = self.T_ref if T is None else float(T)
        S = float(self.H_scale_at(T))
        Js = float(self.Js_at(T))
        rev = MU0 * (self.mu_rec - 1.0)

        def J(Hv):  # descending major branch, P = 1 - 2 E(+inf, h)
            Hv = np.asarray(Hv, dtype=float)
            return Js * (1.0 - 2.0 * self.dist.everett(np.inf, Hv / S)) + rev * Hv

        def root(fun, lo, hi):
            return optimize.brentq(lambda Hv: float(fun(Hv)), lo, hi, xtol=1e-12 * S, rtol=1e-14)

        Br = float(J(0.0))
        span = self.dist.hc_range[1] + 10 * self.dist.w
        while J(-span * S) > 0:
            span *= 2.0
        Hg = -S * np.linspace(0.0, span, 4001)
        Jg = J(Hg)
        Bg = MU0 * Hg + Jg

        def crossing(vals, fun):
            i = int(np.flatnonzero(vals <= 0)[0])
            return -root(fun, Hg[i], Hg[i - 1])

        Hcj = crossing(Jg, J)
        Hcb = crossing(Bg, lambda Hv: MU0 * Hv + J(Hv))
        Hk = crossing(Jg - 0.9 * Br, lambda Hv: J(Hv) - 0.9 * Br)
        BH = np.where(Hg >= -Hcb, -Hg * Bg, -np.inf)
        i = int(np.argmax(BH))
        a, b = Hg[min(i + 1, Hg.size - 1)], Hg[max(i - 1, 0)]
        res = optimize.minimize_scalar(lambda Hv: float(Hv * (MU0 * Hv + J(Hv))), bounds=(a, b),
                                       method="bounded", options={"xatol": 1e-11 * S})
        H_d = float(res.x)
        B_d = float(MU0 * H_d + J(H_d))
        # recoil from (H_d, B_d) to H = 0: P = P_d(h_d) + 2 E(0, h_d)
        h_d = H_d / S
        P_rec = 1.0 - 2.0 * self.dist.everett(np.inf, h_d) + 2.0 * self.dist.everett(0.0, h_d)
        B_rec0 = Js * float(P_rec)
        mu_chord = (B_rec0 - B_d) / (MU0 * (0.0 - H_d))
        return {"Br": Br, "Hcj": Hcj, "Hcb": Hcb, "Hk": Hk, "BHmax": -H_d * B_d, "H_d": H_d, "B_d": B_d,
                "mu_rec_chord": mu_chord}
