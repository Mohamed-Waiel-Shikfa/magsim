"""Scalar Preisach material (materials/preisach.py): gates P2 and P4 of docs/SPEC.md plus contract checks.

References: classical Preisach properties (wiping-out, congruency of minor loops, return-point memory,
odd major loop for a symmetric density) [mayergoyz1991, ch. 1]; independent reference for the Everett
function: adaptive quadrature (scipy.integrate.quad) of its defining double integral.
"""

import time

import numpy as np
import pytest
from scipy import integrate, special

from magsim.constants import MU0
from magsim.materials.preisach import Z_MAX, PreisachDistribution, PreisachMaterial, PreisachState

T0 = 293.15
DISTS = [  # (kind, h0, s, w, skew): NdFeB-like, LNG37-femm-like, broad, edge of the identification range
    ("gauss", 1.0, 0.03, 0.03, 0.0),
    ("lorentz", 1.01, 0.077, 0.05, 0.385),
    ("lorentz", 1.0, 0.2, 0.08, -0.2),
    ("gauss", 1.0, 0.1, 0.025, 0.3),  # w/(h0 s) = 0.25 (lower bound) with skew: hardest for the quadrature
]
# Quadrature of the exact derivative vs finite differences of the quadrature of E: both carry the quadrature
# error, which is largest at the edge of the allowed parameter range (last entry of DISTS).
DERIV_RTOL = [1e-5, 1e-5, 1e-5, 1e-3]


def make(kind="lorentz", h0=1.0, s=0.06, w=0.04, skew=0.0, mu_rec=1.5, depth=32, **kw):
    return PreisachMaterial(1.2, PreisachDistribution(h0, s, w, kind, skew), 50e3, mu_rec=mu_rec, depth=depth, **kw)


def walk(m, path, n=1, T=T0, st=None):
    st = m.init_state(n, "+sat") if st is None else st
    for H in path:
        st = m.commit(np.full(n, float(H)), st, T)
    return st


def J_at(m, st, H, T=T0):
    H = np.atleast_1d(np.asarray(H, dtype=float))
    return m.response(H, np.zeros((H.size, 2)), st, T)


def everett_ref(d, a, b):
    """E(a, b) and dE/da by adaptive quadrature of int phi(z) [G(a - Hc(z)) - G(b + Hc(z))] dz."""
    c = (a - b) / 2.0
    zc = min(float(d.z_of_hc(c)), Z_MAX) if c > 0 else -np.inf
    if zc <= -Z_MAX:
        return 0.0, 0.0
    norm = special.ndtr(Z_MAX) - special.ndtr(-Z_MAX)

    def phi(z):
        return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) / norm

    pts = [float(d.z_of_hc(v)) for v in (a, -b) if 0 < v < np.inf]
    pts = [p for p in pts if -Z_MAX < p < zc] or None
    kw = dict(points=pts, limit=400, epsabs=1e-14, epsrel=1e-12)
    E = integrate.quad(lambda z: phi(z) * (d.G(a - d.hc_of_z(z)) - d.G(b + d.hc_of_z(z))), -Z_MAX, zc, **kw)[0]
    Ea = integrate.quad(lambda z: phi(z) * d.g(a - d.hc_of_z(z)), -Z_MAX, zc, **kw)[0] if np.isfinite(a) else 0.0
    return E, Ea


# ----------------------------------------------------------------------------------------------- Everett function
@pytest.mark.parametrize("kind,h0,s,w,skew", DISTS)
def test_everett_matches_adaptive_quadrature(kind, h0, s, w, skew):
    d = PreisachDistribution(h0, s, w, kind, skew)
    rng = np.random.default_rng(0)
    a = rng.uniform(-2.0, 2.0, 40)
    b = a - rng.uniform(0.0, 3.0, 40)
    b[:8] = -np.inf
    a[8:16] = np.inf
    E, dEa = d.everett(a, b, which=np.ones(40))
    ref = np.array([everett_ref(d, ai, bi) for ai, bi in zip(a, b)])
    assert np.max(np.abs(E - ref[:, 0])) < 1e-6
    assert np.max(np.abs(dEa - ref[:, 1])) < 1e-4 * max(1.0, np.abs(ref[:, 1]).max())


def test_everett_limits_and_symmetry():
    d = PreisachDistribution(1.0, 0.06, 0.05, "lorentz", 0.3)
    assert d.everett(np.inf, -np.inf) == pytest.approx(1.0, abs=1e-14)
    assert d.everett(0.4, 0.4) == 0.0  # zero-width triangle
    h = np.linspace(-3, 3, 41)
    # symmetric density mu(alpha, beta) = mu(-beta, -alpha): E(+inf, h) = E(-h, -inf)
    assert np.allclose(d.everett(np.inf, h), d.everett(-h, -np.inf), atol=1e-14)
    # monotone in alpha (non-decreasing) and in beta (non-increasing)
    assert np.all(np.diff(d.everett(h, -1.5)) >= -1e-12)
    assert np.all(np.diff(d.everett(1.5, h)) <= 1e-12)


def test_distribution_validation():
    for bad in (dict(h0=0.0), dict(s=-1.0), dict(w=np.nan), dict(kind="cauchy"), dict(skew=5.0)):
        kw = dict(h0=1.0, s=0.05, w=0.05, kind="gauss", skew=0.0) | bad
        with pytest.raises(ValueError):
            PreisachDistribution(**kw)


# ----------------------------------------------------------------------------------------------- P2: loop properties
@pytest.mark.parametrize("kind", ["gauss", "lorentz"])
def test_major_loop_odd_symmetry(kind):
    m = make(kind, s=0.05, w=0.05, skew=0.2)
    H = np.linspace(-150e3, 150e3, 301)
    Jd, dJd = m.major_loop(H, "descending")
    Ja, dJa = m.major_loop(-H, "ascending")
    assert np.allclose(Jd, -Ja, atol=1e-12)
    assert np.allclose(dJd, dJa, rtol=1e-10, atol=1e-18)
    assert m.major_loop([0.0])[0][0] == pytest.approx(m.Br, rel=1e-13)


def test_P_bounded_random_histories():
    m = make("lorentz", s=0.08, w=0.05, skew=0.3)
    rng = np.random.default_rng(1)
    n = 400
    st = m.init_state(n, "+sat")
    for _ in range(60):
        H = rng.normal(0.0, 70e3, n)
        Ptrial = m._step(H / m.H_scale, st, build=False)[0]
        assert np.all(Ptrial <= 1 + 1e-12) and np.all(Ptrial >= -1 - 1e-12)
        st = m.commit(H, st, T0)
        assert np.all(st.P_last <= 1 + 1e-12) and np.all(st.P_last >= -1 - 1e-12)
        assert np.all(st.k <= m.depth)


@pytest.mark.parametrize("kind", ["gauss", "lorentz"])
def test_recoil_slope_equals_mu0_mu_rec(kind):
    """P2: dB/dH at a reversal point = MU0 mu_rec (the Everett term has zero slope there) - within 1 %."""
    m = make(kind, s=0.05, w=0.04, mu_rec=1.7)
    for Hd in (-30e3, -48e3, -52e3, -70e3):  # before, at and past the knee
        st = walk(m, [Hd])
        J1, dJ1, _, _ = J_at(m, st, Hd + 1e-3)  # just after the reversal (ascending)
        assert MU0 + dJ1[0] == pytest.approx(MU0 * m.mu_rec, rel=1e-9)
        dH = 1.0  # 1 A/m finite-difference recoil slope
        Jfd = (J_at(m, st, Hd + dH)[0][0] - J_at(m, st, Hd)[0][0]) / dH
        assert MU0 + Jfd == pytest.approx(MU0 * m.mu_rec, rel=1e-2)


@pytest.mark.parametrize("kind", ["gauss", "lorentz"])
def test_wiping_out_closed_minor_loop(kind):
    """A closed minor loop leaves the history as before (wiping-out / return-point memory)."""
    m = make(kind, s=0.06, w=0.05)
    S = walk(m, [-60e3, 20e3, -45e3, -10e3])  # ascending at -10 kA/m with 3 turning points
    loop = walk(m, [-35e3, -20e3, -30e3, 5e3], st=S)  # nested excursions, then past -10 kA/m and beyond
    direct = walk(m, [5e3], st=S)
    assert np.array_equal(loop.k, direct.k)
    assert np.allclose(loop.t, direct.t, equal_nan=True, atol=0)
    assert np.allclose(loop.Pc, direct.Pc, equal_nan=True, atol=1e-14)
    assert loop.P_last[0] == pytest.approx(direct.P_last[0], abs=1e-14)
    # return point: back at exactly -10 kA/m, P equals the value before the loop
    back = walk(m, [-35e3, -10e3], st=S)
    assert back.P_last[0] == pytest.approx(S.P_last[0], abs=1e-14)


@pytest.mark.parametrize("kind", ["gauss", "lorentz"])
def test_congruency_of_minor_loops(kind):
    """Minor loops between the same input limits are congruent for different histories [mayergoyz1991]."""
    m = make(kind, s=0.06, w=0.05)
    h1, h2 = -60e3, 30e3  # amplitude beyond the smallest switching field, so the loop is hysteretic
    A = walk(m, [-70e3, 40e3, h1])  # two different histories, both reversing at h1 below a max >= h2
    B = walk(m, [-90e3, 0.0, -80e3, 50e3, h1])
    grid = np.linspace(h1, h2, 21)
    up = []
    down = []
    for st in (A, B):
        Pa = np.array([J_at(m, st, H)[0][0] for H in grid])
        top = walk(m, [h2], st=st)
        Pd = np.array([J_at(m, top, H)[0][0] for H in grid[::-1]])
        up.append(Pa - Pa[0])
        down.append(Pd - Pd[0])
    assert np.allclose(up[0], up[1], atol=1e-12)
    assert np.allclose(down[0], down[1], atol=1e-12)
    assert np.ptp(up[0]) > 0.05  # the loop is not trivially reversible


# ----------------------------------------------------------------------------------------------- continuity, derivative
@pytest.mark.parametrize("kind,h0,s,w,skew", DISTS)
def test_continuity_of_J(kind, h0, s, w, skew):
    """J is continuous in H: at a reversal point, at a wiped turning point, and along branches."""
    m = PreisachMaterial(1.3, PreisachDistribution(h0, s, w, kind, skew), 50e3, mu_rec=1.3)
    st = walk(m, [-55e3, -20e3, -50e3, -30e3])  # turning points -55, -20, -50; ascending at -30
    eps = 1e-4  # A/m
    for H0 in (-30e3, -20e3, -50e3):  # h_last (reversal), outer max (wipe), outer min (wipe on descent)
        Jm = J_at(m, st, H0 - eps)[0][0]
        Jp = J_at(m, st, H0 + eps)[0][0]
        slope = max(abs(J_at(m, st, H0 + 1.0)[0][0] - Jp), abs(Jm - J_at(m, st, H0 - 1.0)[0][0]))
        assert abs(Jp - Jm) <= 2 * eps * slope + 1e-12
    H = np.linspace(-80e3, 80e3, 4001)
    J = np.array([J_at(m, st, Hv)[0][0] for Hv in H[::40]])
    assert np.all(np.isfinite(J))


@pytest.mark.parametrize("kind,h0,s,w,skew,rtol", [d + (r,) for d, r in zip(DISTS, DERIV_RTOL)])
def test_analytic_derivative_matches_finite_differences(kind, h0, s, w, skew, rtol):
    m = PreisachMaterial(1.3, PreisachDistribution(h0, s, w, kind, skew), 50e3, mu_rec=1.3, alpha_Br=-1e-3,
                         beta_Hcj=-5e-3)
    rng = np.random.default_rng(2)
    n = 60
    st = m.init_state(n, "+sat")
    for H in (-55e3, -20e3, -48e3):
        st = m.commit(H * (1 + 0.05 * rng.standard_normal(n)), st, 330.0)
    # trial fields on both sides of h_last, away from the reversal point itself
    H = -48e3 * (1 + 0.05 * rng.standard_normal(n)) + rng.choice([-1, 1], n) * 3e3
    _, dJ, _, _ = m.response(H, np.zeros((n, 2)), st, 330.0)
    dH = 0.05
    Jp = m.response(H + dH, np.zeros((n, 2)), st, 330.0)[0]
    Jm = m.response(H - dH, np.zeros((n, 2)), st, 330.0)[0]
    fd = (Jp - Jm) / (2 * dH)
    assert np.allclose(dJ, fd, rtol=rtol, atol=1e-6 * MU0)
    assert np.all(dJ >= MU0 * (m.mu_rec - 1) - 1e-15)  # irreversible part never decreases J along a branch


# ----------------------------------------------------------------------------------------------- stack overflow
def test_stack_overflow_drops_innermost_reversal():
    """Depth-6 stack vs depth-64 reference on a decaying oscillation (12 nested reversals)."""
    kw = dict(kind="lorentz", s=0.1, w=0.05)
    small, ref = make(depth=6, **kw), make(depth=64, **kw)
    amps = 80e3 * 0.85 ** np.arange(14)
    path = [(-1) ** (i + 1) * a for i, a in enumerate(amps)]  # -80, +68, -57.8, ...
    s_small = small.init_state(1, "+sat")
    s_ref = ref.init_state(1, "+sat")
    for i, H in enumerate(path):
        P_trial = small.response(np.array([H]), np.zeros((1, 2)), s_small, T0)[0][0]
        s_small = small.commit(np.array([H]), s_small, T0)
        s_ref = ref.commit(np.array([H]), s_ref, T0)
        assert s_small.k[0] <= 6
        assert s_small.P_last[0] == pytest.approx((P_trial - MU0 * (small.mu_rec - 1) * H) / small.Js_irr, abs=1e-12)
        # error bound: Everett weight of the loop between the last two stored turning points of the reference
        if s_ref.k[0] > 6:
            t = s_ref.t[0]
            a, b = max(t[4], t[5]), min(t[4], t[5])
            bound = 2 * float(ref.dist.everett(a, b))
            assert abs(s_small.P_last[0] - s_ref.P_last[0]) <= bound + 1e-12
        else:
            assert s_small.P_last[0] == pytest.approx(s_ref.P_last[0], abs=1e-13)
    assert s_ref.k[0] > 6  # overflow actually happened
    # a large excursion wipes every dropped loop: both models agree again exactly
    for H in (-90e3, 30e3):
        s_small = small.commit(np.array([H]), s_small, T0)
        s_ref = ref.commit(np.array([H]), s_ref, T0)
    assert s_small.P_last[0] == pytest.approx(s_ref.P_last[0], abs=1e-13)
    assert s_small.k[0] == s_ref.k[0]


# ----------------------------------------------------------------------------------------------- init_state variants
@pytest.mark.parametrize("kind", ["gauss", "lorentz"])
def test_init_states(kind):
    m = make(kind, s=0.06, w=0.05)
    n = 4
    Hp = np.zeros((n, 2))
    zero = np.zeros(n)
    for init, sign in (("+sat", 1), ("-sat", -1)):
        st = m.init_state(n, init)
        assert st.P.shape == (n,) and np.all(st.P == sign)
        assert np.allclose(m.response(zero, Hp, st, T0)[0], sign * m.Br, rtol=1e-13)
    th = m.init_state(n, "thermal")
    assert np.all(np.abs(th.P) < 1e-10)
    assert np.all(th.k == min(16, m.depth // 2)) and np.all(th.h_last == 0.0)
    assert np.allclose(m.response(zero, Hp, th, T0)[0], 0.0, atol=1e-10)
    # virgin curve from the thermal state rises and stays below the major descending branch
    th1 = m.init_state(1, "thermal")
    Jv = np.array([J_at(m, th1, H)[0][0] for H in (10e3, 40e3, 60e3)])
    assert np.all(np.diff(Jv) > 0) and np.all(Jv < m.major_loop([10e3, 40e3, 60e3])[0])
    for P0 in (0.5, -0.3, 0.0, 0.9):
        st = m.init_state(n, P0)
        assert np.allclose(st.P, P0, atol=1e-9)
        assert np.allclose(m.response(zero, Hp, st, T0)[0], m.Js_irr * P0, atol=1e-9)
        mirror = m.init_state(n, -P0)
        assert np.allclose(mirror.P, -P0, atol=1e-9)
    # above the remanent fraction P_r no state at H = 0 exists: the state sits at a positive field
    P0 = 0.5 * (1 + m.P_r)
    st = m.init_state(1, P0)
    assert st.P[0] == pytest.approx(P0, abs=1e-9) and st.h_last[0] > 0
    for init in ("+sat", "-sat"):
        sat, num = m.init_state(1, init), m.init_state(1, 1.0 if init == "+sat" else -1.0)
        assert np.array_equal(sat.h_last, num.h_last) and np.array_equal(sat.P, num.P)
    for bad in ("sat", 1.5, -2.0, np.nan):
        with pytest.raises(ValueError):
            m.init_state(2, bad)
    with pytest.raises(ValueError):
        m.init_state(-1, "+sat")
    assert m.init_state(0, "+sat").n == 0


# ----------------------------------------------------------------------------------------------- contract and validation
def test_contract_shapes_and_state_immutability():
    m = make("gauss", s=0.04, w=0.04, mu_rec=1.05, mu_perp=1.17)
    n = 5
    st = m.init_state(n, "+sat")
    st = m.commit(np.full(n, -30e3), st, T0)
    snapshot = [a.copy() for a in (st.s0, st.k, st.t, st.Pc, st.h_last, st.P_last)]
    H = np.linspace(-60e3, 20e3, n)
    Hp = np.arange(2 * n, dtype=float).reshape(n, 2) * 1e3
    J, dJ, Jp, chi = m.response(H, Hp, st, T0)
    assert J.shape == dJ.shape == chi.shape == (n,) and Jp.shape == (n, 2)
    assert np.allclose(Jp, MU0 * (m.mu_perp - 1) * Hp)
    assert np.allclose(chi, m.mu_perp - 1)
    new = m.commit(H, st, T0)
    assert isinstance(new, PreisachState) and new is not st
    for before, after in zip(snapshot, (st.s0, st.k, st.t, st.Pc, st.h_last, st.P_last)):
        assert np.array_equal(before, after, equal_nan=True)


def test_input_validation():
    m = make()
    st = m.init_state(3, "+sat")
    good_H, good_Hp = np.zeros(3), np.zeros((3, 2))
    for H, Hp, T in ((np.zeros(2), good_Hp, T0), (good_H, np.zeros((3, 3)), T0), (good_H, good_Hp, np.ones(2)),
                     (np.array([0.0, np.nan, 0.0]), good_Hp, T0), (good_H, good_Hp, -1.0),
                     (good_H, np.full((3, 2), np.inf), T0)):
        with pytest.raises(ValueError):
            m.response(H, Hp, st, T)
    with pytest.raises(ValueError):
        m.commit(np.zeros(4), st, T0)
    with pytest.raises(TypeError):
        m.response(good_H, good_Hp, object(), T0)
    for bad in (dict(Br=-1.0), dict(H_scale=0.0), dict(mu_rec=0.9), dict(mu_perp=np.nan), dict(depth=1),
                dict(T_ref=0.0), dict(alpha_Br=np.inf)):
        kw = dict(Br=1.2, dist=PreisachDistribution(1.0, 0.05, 0.05), H_scale=5e4) | bad
        with pytest.raises(ValueError):
            PreisachMaterial(**kw)
    with pytest.raises(ValueError):  # Hcj(T) <= 0
        PreisachMaterial(1.2, PreisachDistribution(1.0, 0.05, 0.05), 5e4, beta_Hcj=-0.01).response(
            good_H, good_Hp, st, T0 + 150.0)


# ----------------------------------------------------------------------------------------------- temperature
def test_temperature_scaling():
    a, b = -1.2e-3, -6.0e-3
    m = make("gauss", s=0.04, w=0.04, mu_rec=1.05, alpha_Br=a, beta_Hcj=b)
    base = m.characteristics()
    for dT in (-40.0, 60.0):
        ch = m.characteristics(T0 + dT)
        assert ch["Br"] == pytest.approx(m.Br * (1 + a * dT), rel=1e-12)  # exact by construction
        # field axis scales with Hcj(T); the unscaled reversible term shifts Hcj by << 1 %
        assert ch["Hcj"] == pytest.approx(base["Hcj"] * (1 + b * dT), rel=5e-3)
    # history in reduced units: a state committed at T1 answers at T2 for the same reduced field
    st = walk(m, [-45e3], T=T0)
    S1, S2 = m.H_scale_at(T0), m.H_scale_at(T0 + 50)
    J1 = m.response(np.array([-20e3]), np.zeros((1, 2)), st, T0)[0][0]
    H2 = -20e3 * S2 / S1
    J2 = m.response(np.array([H2]), np.zeros((1, 2)), st, T0 + 50)[0][0]
    rev = MU0 * (m.mu_rec - 1)
    assert (J2 - rev * H2) / m.Js_at(T0 + 50) == pytest.approx((J1 - rev * -20e3) / m.Js_at(T0), abs=1e-13)
    # per-cell temperatures
    n = 3
    Tn = np.array([250.0, 293.15, 350.0])
    J, _, _, _ = m.response(np.zeros(n), np.zeros((n, 2)), m.init_state(n, "+sat"), Tn)
    assert np.allclose(J, m.Br * (1 + a * (Tn - T0)), rtol=1e-12)
    # heating at a fixed negative field lowers Hcj and switches hysterons irreversibly
    hot = m.commit(np.array([-0.97 * base["Hcj"]]), walk(m, [-0.97 * base["Hcj"]]), T0 + 40)
    back = m.commit(np.array([0.0]), hot, T0)
    assert back.P_last[0] < walk(m, [-0.97 * base["Hcj"], 0.0]).P_last[0] - 0.01


# ----------------------------------------------------------------------------------------------- P4 performance
def _best_of(fn, k=5):
    best = np.inf
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.parametrize("kind,s,w,skew", [("gauss", 0.05, 0.05, 0.0), ("lorentz", 0.077, 0.05, 0.385)])
def test_P4_response_time_1e4_cells(kind, s, w, skew):
    """P4: 1e4 cells x one response call < 50 ms (best of 5, worst case: every cell has a live Everett term
    with three quadrature panels)."""
    m = make(kind, s=s, w=w, skew=skew)
    n = 10_000
    rng = np.random.default_rng(3)
    st = m.init_state(n, "+sat")
    st = m.commit(np.full(n, -1.1 * 50e3), st, T0)
    st = m.commit(np.full(n, 0.9 * 50e3), st, T0)
    H = -0.95 * 50e3 * (1 + 0.02 * rng.standard_normal(n))
    Hp = np.zeros((n, 2))
    m.response(H, Hp, st, T0)  # warm-up
    assert _best_of(lambda: m.response(H, Hp, st, T0)) < 0.050
