"""Tests for magsim.analytic against brute-force quadrature, Bessel integrals and magpylib."""

import magpylib.func as mpf
import numpy as np
import pytest
from scipy import integrate, special

from magsim import analytic as an
from magsim.constants import MU0
from magsim.materials.base import LinearMaterial


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _graded01(levels=10, q=10):
    """GL nodes on [0, 1] graded geometrically toward both ends (edge log-singularities)."""
    e = np.array([0.0] + [0.5 * 0.5**k for k in range(levels, 0, -1)] + [0.5])
    e = np.concatenate([e, 1 - e[::-1][1:]])
    x, w = np.polynomial.legendre.leggauss(q)
    pts = np.concatenate([0.5 * (b - a) * x + 0.5 * (a + b) for a, b in zip(e[:-1], e[1:])])
    wts = np.concatenate([0.5 * (b - a) * w for a, b in zip(e[:-1], e[1:])])
    return pts, wts


@pytest.mark.parametrize("wz", [2.0, -1.5, 0.8])
def test_rect_pair_integral_vs_bruteforce(wz):
    """Corner-sum antiderivatives vs 4-D Gauss-Legendre of (r_t - r_s)/|r_t - r_s|^3 (smooth case)."""
    t = (0.1, 1.3, -0.4, 0.7)
    s = (-0.5, 0.6, 0.2, 1.5)
    xs, wx = _gl(t[0], t[1], 40)
    ys, wy = _gl(t[2], t[3], 40)
    Xs, wX = _gl(s[0], s[1], 40)
    Ys, wY = _gl(s[2], s[3], 40)
    U = xs[:, None, None, None] - Xs[None, None, :, None]
    V = ys[None, :, None, None] - Ys[None, None, None, :]
    R3 = np.sqrt(U**2 + V**2 + wz**2) ** 3
    W = wx[:, None, None, None] * wy[None, :, None, None] * wX[None, None, :, None] * wY[None, None, None, :]
    ref = np.array([(W * U / R3).sum(), (W * V / R3).sum(), (W * wz / R3).sum()])
    got = an.rect_pair_integral(np.array(t[:2]), np.array(t[2:]), np.array(s[:2]), np.array(s[2:]), wz)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_rect_pair_zero_w_limits():
    """w -> 0+: Phi_z keeps (pi/2)|uv|; coplanar overlapping squares give the contact pressure 2 pi A."""
    t = np.array([0.0, 1.0])
    I0 = an.rect_pair_integral(t, t, t, t, 0.0, side=1.0, w_tol=1e-15)
    Ism = an.rect_pair_integral(t, t, t, t, 1e-9, side=1.0)
    assert abs(I0[2] - 2 * np.pi) < 1e-7  # integral of w/r^3 over the plane -> 2 pi per unit overlap area
    assert np.allclose(I0, Ism, atol=1e-7)
    assert np.allclose(an.rect_pair_integral(t, t, t, t, 0.0, side=-1.0, w_tol=1e-15)[2], -2 * np.pi)
    # coplanar, non-overlapping: no normal force whatever the side
    far = np.array([2.0, 3.0])
    assert abs(an.rect_pair_integral(far, t, t, t, 0.0, 1.0, 1e-15)[2]) < 1e-12


def test_akoun_yonnet_vs_magpylib_quadrature():
    """Full-vector AY force vs GL quadrature of sigma * H_magpylib over the target poles."""
    d1 = np.array([1.0, 0.7, 0.5])
    d2 = np.array([0.8, 0.6, 0.4])
    off = np.array([0.3, -0.2, 0.6])
    x, w = np.polynomial.legendre.leggauss(30)
    F = np.zeros(3)
    for zf, s in ((off[2] + 0.2, 1.0), (off[2] - 0.2, -1.0)):
        X, Y = off[0] + 0.4 * x, off[1] + 0.3 * x
        P = np.stack(np.meshgrid(X, Y, [zf], indexing="ij"), -1).reshape(-1, 3)
        W = np.outer(w * 0.4, w * 0.3).ravel()
        H = mpf.cuboid_field("H", P, d1, (0, 0, 1.0)) * (1.25663706127e-6 / MU0)  # magpylib mu0 -> ours
        F += s * (W[:, None] * H).sum(0)
    ref = an.akoun_yonnet_force(1.0, 1.0, d1, d2, off)
    assert np.allclose(F, ref, rtol=1e-8, atol=1e-8 * np.linalg.norm(ref))


def test_akoun_yonnet_continuity_at_contact():
    """F(gap) is continuous as gap -> 0+ (the old code's zero-gap branch dropped (pi/2)|uv|)."""
    d1, d2 = np.array([1.0, 1.0, 0.5]), np.array([0.8, 0.6, 0.4])
    base = np.array([0.1, 0.05, 0.45])
    F0 = an.akoun_yonnet_force(1.2, 1.2, d1, d2, base)
    for g, tol in ((1e-12, 1e-10), (1e-9, 1e-7)):
        Fg = an.akoun_yonnet_force(1.2, 1.2, d1, d2, base + [0, 0, g])
        assert np.linalg.norm(Fg - F0) / np.linalg.norm(F0) < tol
    # the old branch (arctan term set to 0 at w = 0) is off by the contact pressure term
    ov = 0.8 * 0.6  # overlap area of the touching faces (target fully inside)
    contact = 1.2 * 1.2 * ov / (2 * MU0)
    assert contact / abs(F0[2]) > 0.5
    # target below the source (approached from w -> 0-)
    Fm = an.akoun_yonnet_force(1.2, 1.2, d1, d2, base * [1, 1, -1])
    Fm2 = an.akoun_yonnet_force(1.2, 1.2, d1, d2, base * [1, 1, -1] - [0, 0, 1e-12])
    assert np.linalg.norm(Fm - Fm2) / np.linalg.norm(Fm) < 1e-10
    with pytest.raises(ValueError):
        an.akoun_yonnet_force(1.0, 1.0, d1, d2, [0, 0, 0.4])


def test_akoun_yonnet_dipole_limit_and_symmetry():
    d = np.array([1.0, 1.0, 1.0])
    m = np.array([0, 0, 1.0]) / MU0
    for off in ([0, 0, 20.0], [3, 2, 20.0], [20, 0, 0.0]):
        F = an.akoun_yonnet_force(1.0, 1.0, d, d, off)
        Fd = an.dipole_force(m, m, np.array(off, dtype=float))
        assert np.linalg.norm(F - Fd) / np.linalg.norm(Fd) < 1e-4  # cube: first correction (a/d)^4 ~ 6e-6
    F12 = an.akoun_yonnet_force(1.1, 0.9, [1, 2, 0.5], [0.5, 0.7, 0.3], [0.3, 0.4, 0.6])
    F21 = an.akoun_yonnet_force(0.9, 1.1, [0.5, 0.7, 0.3], [1, 2, 0.5], [-0.3, -0.4, -0.6])
    assert np.allclose(F12, -F21, rtol=1e-12, atol=1e-12 * np.linalg.norm(F12))


@pytest.mark.parametrize("b", [0.05, 0.3, 1.0, 4.0])
def test_lipschitz_hankel_kernels(b):
    g = integrate.quad(lambda x: special.j1(x) ** 2 * np.exp(-b * x) / x, 0, np.inf, limit=2000)[0]
    assert abs(an.force_kernel_G(b) - g) < 1e-9
    # Phi' = G and Phi(0) = 0 (independent of the theta form of Phi)
    p = integrate.quad(an.force_kernel_G, 0, b, epsabs=1e-14, epsrel=1e-13)[0]
    assert abs(an.field_kernel_Phi(b) - p) < 1e-12


def test_kernel_limits():
    assert an.force_kernel_G(0.0) == 0.5
    assert abs(an.force_kernel_G(1e-6) - 0.5) < 1e-5
    assert abs(an.force_kernel_G(1e3) * 4e6 - 1) < 1e-4  # G ~ 1/(4 b^2)
    assert abs(an.field_kernel_Phi(1e-5) / 0.5e-5 - 1) < 1e-4  # Phi ~ b/2
    u1, u2 = np.array([150.0, 2e3]), np.array([160.0, 3e3])
    ref = (u2 - u1) / (4 * u1 * u2) + 1 / (8 * u2**3) - 1 / (8 * u1**3)
    assert np.allclose(an.field_kernel_Phi_diff(u1, u2), ref, rtol=1e-12)
    assert abs(an.field_kernel_Phi_diff(1.0, 2.0) - (an.field_kernel_Phi(2.0) - an.field_kernel_Phi(1.0))) < 1e-15


def test_coaxial_cylinder_force_vs_magpylib_and_dipole():
    R, L1, L2, gap = 1.0, 1.5, 1.0, 0.4
    ref = an.coaxial_cylinder_force(1.0, 1.0, R, L1, L2, gap)
    # sigma * H_z(magpylib) over the two pole discs of cylinder 2 (axisymmetric -> radial quadrature)
    r, wr = _graded01(12, 10)
    F = 0.0
    for zf, s in ((gap, -1.0), (gap + L2, 1.0)):
        P = np.stack([r * R, 0 * r, np.full_like(r, zf + L1 / 2)], 1)
        Hz = mpf.cylinder_field("H", P, (2 * R, L1), (0, 0, 1.0))[:, 2] * (1.25663706127e-6 / MU0)
        F += s * (2 * np.pi * R * R * r * wr * Hz).sum()
    assert abs(F / ref - 1) < 1e-8
    m1 = np.array([0, 0, np.pi * R**2 * L1]) / MU0
    m2 = np.array([0, 0, np.pi * R**2 * L2]) / MU0
    far = an.coaxial_cylinder_force(1.0, 1.0, R, L1, L2, 300.0)
    Fd = an.dipole_force(m1, m2, np.array([0, 0, 300.0 + (L1 + L2) / 2]))[2]
    assert abs(far / Fd - 1) < 1e-4


def _avg_hz_cylinder(R, L):
    s, ws = _graded01(10, 10)
    r, wr = R * s, R * ws * R * s
    z, wz = L * (s - 0.5), L * ws
    P = np.stack(np.meshgrid(r, [0.0], z, indexing="ij"), -1).reshape(-1, 3)
    W = np.outer(wr, wz).ravel() * 2 * np.pi
    H = mpf.cylinder_field("H", P, (2 * R, L), (0, 0, 1.0)) * (1.25663706127e-6 / MU0)
    return (W * H[:, 2]).sum() / (np.pi * R * R * L)


@pytest.mark.parametrize("LD", [0.5, 1.0, 2.6])
def test_cylinder_demag_factor_vs_volume_average(LD):
    """N = 2R Phi(L/R)/L [chen1991] vs the volume average of magpylib's exact cylinder field."""
    R = 1.0
    L = 2 * R * LD
    N_num = -MU0 * _avg_hz_cylinder(R, L)
    assert abs(an.cylinder_demag_factor(R, L) / N_num - 1) < 1e-8


def test_aharoni_prism():
    assert abs(an.aharoni_prism_demag(1, 1, 1) - 1 / 3) < 1e-14
    d = (20e-3, 10e-3, 5e-3)
    tot = an.aharoni_prism_demag(*d) + an.aharoni_prism_demag(d[1], d[2], d[0]) + an.aharoni_prism_demag(d[2], d[0], d[1])
    assert abs(tot - 1) < 1e-13
    dd = np.array(d)
    s, ws = _graded01(10, 8)
    X, Y, Z = (dd[k] * (s - 0.5) for k in range(3))
    P = np.stack(np.meshgrid(X, Y, Z, indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", ws * dd[0], ws * dd[1], ws * dd[2]).ravel()
    H = mpf.cuboid_field("H", P, dd, (0, 0, 1.0)) * (1.25663706127e-6 / MU0)
    N_num = -MU0 * (W * H[:, 2]).sum() / dd.prod()
    assert abs(an.aharoni_prism_demag(*d) / N_num - 1) < 1e-8


def test_dipole_force():
    m1, m2 = np.array([0.1, 0.2, 1.0]), np.array([-0.3, 0.5, 0.7])
    r = np.array([0.3, -0.4, 1.2])
    assert np.allclose(an.dipole_force(m1, m2, r), -an.dipole_force(m2, m1, -r), rtol=1e-14)
    # coaxial: F_z = -6 MU0 m^2 / (4 pi d^4)
    F = an.dipole_force([0, 0, 2.0], [0, 0, 3.0], [0, 0, 0.5])
    assert abs(F[2] / (-6 * MU0 * 6.0 / (4 * np.pi * 0.5**4)) - 1) < 1e-14


def test_coaxial_rod_pair_rigid_matches_closed_form():
    """Rigid material: slabs carry uniform J, so the 1-D solver reproduces the 4-disc closed form."""
    R, L = 2.375e-3, 12.5e-3
    rod = an.CoaxialRodPair(R, L, LinearMaterial(1.2, 1.0), n_slabs=8)
    for gap in (0.0, 1e-4, 2e-3):
        assert abs(rod.force(gap) / an.coaxial_cylinder_force(1.2, 1.2, R, L, L, gap) - 1) < 1e-10
    rod.set_orientation(1, -1)
    assert rod.force(1e-3) > 0  # repelling


def test_coaxial_rod_pair_linear_open_circuit():
    """Linear recoil: open-circuit slab solution satisfies the slab equations and the load line.

    Mean J and mean H obey <J> = Jr + MU0 (mu_rec - 1) <H> exactly; the mean demagnetising factor
    -MU0 <H>/<J> lies between the uniform-J value N(L/D) and 0.
    """
    R, L = 2.375e-3, 12.5e-3
    mat = LinearMaterial(1.2, 4.0)
    rod = an.CoaxialRodPair(R, L, mat, n_slabs=40)
    J, H = rod.solve(200 * L)
    n = rod.n_slabs
    assert abs(J[:n].mean() - (1.2 + MU0 * 3.0 * H[:n].mean())) < 1e-12
    Neff = -MU0 * H[:n].mean() / J[:n].mean()
    assert 0 < Neff < an.cylinder_demag_factor(R, L) * 1.05
    # residual of the slab equations
    M = rod.coupling(200 * L)
    assert np.max(np.abs(J - (1.2 + MU0 * 3.0 * (M @ J)))) < 1e-9
    rod.open_circuit()
    assert rod.force(1e-4) < 0
