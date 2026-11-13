"""Eddy-loop network: gate C4 (long cylinder vs Bessel) and the network's own consistency."""

import numpy as np
import pytest
from magpylib.func import cuboid_field, polyline_field
from scipy.integrate import dblquad
from scipy.special import jv
from scipy.spatial.transform import Rotation

from magsim import coil as C
from magsim.constants import MU0
from magsim.eddy import eddy_loops, harmonic_currents

RHO = 1.5e-6  # NdFeB typical resistivity [ohm m] (Eclipse Magnetics sheet via magnet_pulse_sim.py [M])


def _axis_ratio(n_r, delta, a=1e-3, n_z=10, length_over_a=20.0):
    """|H_in(0)/H0| of the loop model for a long rod (L = 20 a = 10 D) in a uniform axial AC field."""
    omega = 2 * RHO / (MU0 * delta**2)  # delta = sqrt(2 rho / (omega mu0))
    ed = eddy_loops("cylinder", (2 * a, length_over_a * a), n_r=n_r, n_z=n_z, resistivity=RHO)
    coupling = MU0 * ed.area_vectors()[:, 2]  # flux per unit source field H0 = 1 A/m
    i = harmonic_currents(ed.R, ed.M, coupling, omega)
    h = ed.field_per_amp(np.array([[0.0, 0.0, 0.0]]))[0, :, 2]
    return 1.0 + h @ i


@pytest.mark.parametrize("delta_over_a", [1.0, 1.0 / np.sqrt(np.pi)])
def test_C4_long_cylinder_vs_bessel(delta_over_a):
    """Gate C4: H_in(0)/H0 = 1/J0(k a), k = (1 - j)/delta. delta = a, and f = 1/tau_d with
    tau_d = mu0 a^2 / rho (delta = a / sqrt(pi)). n_r = 8 bands, n_z = 10 slabs, L = 10 D: < 3 %
    (achieved 0.14 % and 0.88 %)."""
    a = 1e-3
    delta = delta_over_a * a
    exact = 1.0 / jv(0, (1 - 1j) / delta * a)
    got = _axis_ratio(8, delta, a)
    assert abs(abs(got) - abs(exact)) / abs(exact) < 0.03
    assert abs(abs(got) - abs(exact)) / abs(exact) < 0.01
    assert abs(np.angle(got) - np.angle(exact)) < np.deg2rad(0.5)


def test_C4_convergence_with_n_r():
    a = 1e-3
    delta = a / np.sqrt(np.pi)
    exact = abs(1.0 / jv(0, (1 - 1j) / delta * a))
    errs = [abs(abs(_axis_ratio(n, delta, a)) - exact) / exact for n in (2, 4, 8, 16)]
    assert all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))
    assert errs[-1] < 5e-3  # 12.2 %, 3.0 %, 0.88 %, 0.36 %


@pytest.mark.parametrize("shape,dims", [("cylinder", (4.75e-3, 12.5e-3)), ("ring", (10e-3, 4e-3, 3e-3)),
                                        ("cuboid", (20e-3, 10e-3, 5e-3))])
def test_toeplitz_matrix_equals_plain_filament_sum(shape, dims):
    pose = ((1e-3, 2e-3, 3e-3), Rotation.from_euler("xyz", [10, 20, 30], degrees=True))
    ed = eddy_loops(shape, dims, pose=pose, n_r=3, n_z=2, resistivity=RHO)
    M2 = C.circuit_inductance(ed.filaments())
    assert np.max(np.abs(ed.M - M2)) < 1e-12 * np.max(np.abs(M2))
    np.linalg.cholesky(ed.M)


def test_resistances():
    ed = eddy_loops("ring", (10e-3, 4e-3, 3e-3), n_r=3, n_z=2, resistivity=RHO)
    r = np.linspace(2e-3, 5e-3, 4)
    rmid = np.tile(0.5 * (r[1:] + r[:-1]), 2)
    assert np.allclose(ed.R, RHO * 2 * np.pi * rmid / (1e-3 * 1.5e-3), rtol=1e-12)
    # cuboid band: P = int rho J^2 dV over the four trapezoids with uniform current per unit psi
    A, B, h = 20e-3, 10e-3, 5e-3
    ed = eddy_loops("cuboid", (A, B, h), n_r=4, n_z=1, resistivity=RHO)
    s1, s2 = 0.5, 0.75
    # x-running trapezoids: current density I / ((B/2) ds dz) over area (A s) (B/2) ds per side, two sides
    Jx = 1.0 / (0.5 * B * (s2 - s1) * h)
    Jy = 1.0 / (0.5 * A * (s2 - s1) * h)
    area_x = 2 * 0.5 * (A * s1 + A * s2) * 0.5 * B * (s2 - s1)
    area_y = 2 * 0.5 * (B * s1 + B * s2) * 0.5 * A * (s2 - s1)
    P = RHO * (Jx**2 * area_x * h + Jy**2 * area_y * h)
    assert ed.R[2] == pytest.approx(P, rel=1e-12)


def test_band_field_on_axis_closed_form():
    """Annular band with uniform azimuthal current density, on its axis:
    H_z = J/2 sum_(+-) +-(z_i - z) ln[(r2 + sqrt(r2^2+(z_i-z)^2)) / (r1 + sqrt(r1^2+(z_i-z)^2))]."""
    ed = eddy_loops("ring", (10e-3, 4e-3, 3e-3), n_r=1, n_z=1, resistivity=RHO)
    r1, r2, z1, z2 = 2e-3, 5e-3, -1.5e-3, 1.5e-3
    J = 1.0 / ((r2 - r1) * (z2 - z1))
    z = np.linspace(-6e-3, 6e-3, 13)
    H = ed.field_per_amp(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]), n_q=8)[:, 0, 2]

    def F(zz):
        return zz * np.log((r2 + np.hypot(r2, zz)) / (r1 + np.hypot(r1, zz)))

    ref = 0.5 * J * (F(z2 - z) - F(z1 - z))
    assert np.max(np.abs(H - ref) / ref) < 1e-8


def test_band_field_inside_band_vs_fine_sheets_and_filaments():
    """Cuboid band field: (i) points inside the band (split Gauss quadrature, n_q = 6) vs a 4000-sheet
    midpoint sum of the same exact sheet kernel; (ii) points outside the magnet vs 400 x 40 filaments
    (magpylib Polyline, independent kernel)."""
    A, B, h = 20e-3, 10e-3, 5e-3
    ed = eddy_loops("cuboid", (A, B, h), n_r=2, n_z=1, resistivity=RHO)
    inside = np.array([[0.7 * A / 2, 0.1 * B / 2, 0.3e-3], [0.0, 0.0, 0.0], [0.9 * A / 2, 0.85 * B / 2, -2e-3]])
    H = ed.field_per_amp(inside, n_q=6)[:, 1, :]  # outer band s in [0.5, 1]
    ns = 4000
    s = 0.5 + (np.arange(ns) + 0.5) / ns * 0.5
    ref = np.zeros_like(H)
    for k, pp in enumerate(inside):
        dims = np.column_stack([A * s, B * s, np.full(ns, h)])
        ref[k] = cuboid_field("B", np.tile(pp, (ns, 1)), dims, (0, 0, MU0 / h)).sum(axis=0) / (MU0 * ns)
    assert np.max(np.linalg.norm(H - ref, axis=1) / np.linalg.norm(ref, axis=1)) < 1e-4

    outside = np.array([[15e-3, 3e-3, 1e-3], [0.0, 0.0, 6e-3], [-12e-3, -9e-3, -4e-3]])
    H = ed.field_per_amp(outside)[:, 1, :]
    nf, nz = 400, 40
    s = 0.5 + (np.arange(nf) + 0.5) / nf * 0.5
    zz = -h / 2 + (np.arange(nz) + 0.5) / nz * h
    starts, ends = [], []
    for si in s:
        v = C.outline_polygon(0.5 * A * si, 0.5 * B * si, 0.0, 1)
        for z in zz:
            a = np.hstack([v, np.full((len(v), 1), z)])
            starts.append(a)
            ends.append(np.roll(a, -1, axis=0))
    S, E = np.vstack(starts), np.vstack(ends)
    ref = np.array([polyline_field("H", np.tile(pp, (len(S), 1)), S, E, 1.0 / (nf * nz)).sum(axis=0)
                    for pp in outside])
    assert np.max(np.linalg.norm(H - ref, axis=1) / np.linalg.norm(ref, axis=1)) < 1e-4


def test_coil_coupling_pose_invariance_and_passivity():
    coil = C.reference_coil_r1()
    ed = eddy_loops("cuboid", (20e-3, 20e-3, 5e-3), n_r=4, n_z=2, resistivity=RHO)
    m = ed.mutual_with(coil)
    L = ed.full_inductance(coil)
    np.linalg.cholesky(L)
    assert np.all(np.abs(L[0, 1:]) < np.sqrt(L[0, 0] * np.diag(L)[1:]))
    rot = Rotation.from_euler("zyx", [25, 40, -10], degrees=True)
    pos = np.array([0.01, -0.02, 0.005])
    coil2 = C.reference_coil_r1(position=pos, rotation=rot)
    ed2 = eddy_loops("cuboid", (20e-3, 20e-3, 5e-3), pose=(pos, rot), n_r=4, n_z=2, resistivity=RHO)
    assert np.allclose(ed2.mutual_with(coil2), m, rtol=1e-9)
    assert np.allclose(ed2.M, ed.M, rtol=1e-12)
    # field per amp rotates with the pose
    p = np.array([[1e-3, 2e-3, 0.5e-3]])
    assert np.allclose(ed2.field_per_amp(pos + rot.apply(p))[0], rot.apply(ed.field_per_amp(p)[0]), rtol=1e-9,
                       atol=1e-9)


def test_cuboid_axis_selection():
    """Loops normal to body x for axis=0: same network as the permuted cuboid with axis=2."""
    e0 = eddy_loops("cuboid", (5e-3, 20e-3, 10e-3), n_r=3, n_z=2, resistivity=RHO, axis=0)
    e2 = eddy_loops("cuboid", (20e-3, 10e-3, 5e-3), n_r=3, n_z=2, resistivity=RHO, axis=2)
    assert np.allclose(e0.M, e2.M, rtol=1e-12) and np.allclose(e0.R, e2.R, rtol=1e-12)
    assert np.allclose(e0.axis_global, [1, 0, 0])


def test_uniform_field_flux_matches_mutual_with_long_solenoid():
    """area_vectors (uniform-field flux) vs mutual with a long, wide solenoid around the rod."""
    a = 2e-3
    ed = eddy_loops("cylinder", (2 * a, 4e-3), n_r=3, n_z=2, resistivity=RHO)
    sol = C.RoundWireCoil("circular", bore_radius=20e-3, d_bare=0.2e-3, d_ins=0.2e-3, turns_per_layer=2000,
                          n_layers=1)
    n = 2000 / sol.winding_length
    m = ed.mutual_with(sol)
    flux = MU0 * n * ed.area_vectors()[:, 2]
    # finite solenoid (length 0.4 m, radius 20 mm): centre field 1/sqrt(1 + (2R/l)^2) of the ideal one
    corr = 1 / np.sqrt(1 + (2 * 20.1e-3 / sol.winding_length) ** 2)
    assert np.allclose(m, flux * corr, rtol=2e-4)


def test_band_self_inductance_vs_quadrature():
    """Single band (n_r = n_z = 1) of a square-section ring: sub-filament model vs a direct 4-D
    average of Maxwell's formula (2-D quadrature after the axial convolution)."""
    r1, r2, h = 4e-3, 5e-3, 1e-3
    ed = eddy_loops("ring", (2 * r2, 2 * r1, h), n_r=1, n_z=1, resistivity=RHO, n_sub=(16, 16))
    # reference: <M> = 1/(dr^2 h^2) int int dr dr' int_{-h}^{h} (h - |z|) M(r, r', z) dz, by nested quadrature
    from scipy.integrate import quad

    def inner(r, rp):
        f = lambda z: (h - z) * C.maxwell_mutual(r, rp, z)  # noqa: E731
        return 2 * quad(f, 0, h, limit=200, points=[0.0])[0]

    g = lambda rp, r: inner(r, rp)  # noqa: E731
    val, _ = dblquad(g, r1, r2, r1, r2, epsabs=0, epsrel=1e-6)
    ref = val / ((r2 - r1) ** 2 * h**2)
    assert ed.M[0, 0] == pytest.approx(ref, rel=2e-3)
