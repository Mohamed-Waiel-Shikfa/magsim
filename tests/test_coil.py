"""Coil kernels and coils: gates C1 (inductance), C2 (on-axis fields), C3 (R1 foil coil)."""

import numpy as np
import pytest
from magpylib.func import cuboid_field
from scipy.integrate import dblquad, quad
from scipy.spatial.transform import Rotation

from magsim import coil as C
from magsim.constants import ALPHA_CU, CP_CU, DENSITY_CU, MU0, RHO_CU_20

# ------------------------------------------------------------------------------------------- kernels


def _neumann_segments(P1, P2, Q1, Q2):
    u, v = P2 - P1, Q2 - Q1
    f = lambda t, s: 1.0 / np.linalg.norm(P1 + s * u - Q1 - t * v)  # noqa: E731
    val, _ = dblquad(f, 0, 1, 0, 1, epsabs=1e-14, epsrel=1e-12)
    return MU0 / (4 * np.pi) * np.dot(u, v) * val


@pytest.mark.parametrize("case", ["skew", "skew2", "touching", "coplanar", "oblique_small_angle"])
def test_segment_mutual_vs_quadrature(case):
    rng = np.random.default_rng({"skew": 1, "skew2": 7, "touching": 3, "coplanar": 4, "oblique_small_angle": 5}[case])
    P1, P2, Q1, Q2 = rng.normal(size=(4, 3))
    if case == "touching":
        Q1 = P2.copy()
    if case == "coplanar":
        P1[2] = P2[2] = Q1[2] = Q2[2] = 0.3
    if case == "oblique_small_angle":
        P1, P2 = np.zeros(3), np.array([1.0, 0, 0])
        Q1 = np.array([0.2, 0.1, 0.03])
        Q2 = Q1 + [np.cos(1e-3), np.sin(1e-3), 0]
    ref = _neumann_segments(P1, P2, Q1, Q2)
    assert C.segment_mutual(P1, P2, Q1, Q2) == pytest.approx(ref, rel=1e-9)


def test_parallel_partial_self_inductance_closed_form():
    """Straight filament vs its copy at distance g = exact GMD partial self inductance [grover1946]."""
    l, g = 0.1, 1e-4
    got = C.segment_mutual([0, 0, 0], [l, 0, 0], [0, g, 0], [l, g, 0])
    ref = MU0 / (2 * np.pi) * (l * np.arcsinh(l / g) - np.hypot(l, g) + g)
    assert got == pytest.approx(ref, rel=1e-12)
    # antiparallel offset pair: sign and value
    got = C.segment_mutual([0, 0, 0], [l, 0, 0], [1.5 * l, 0.02, 0], [0.5 * l, 0.02, 0])
    ref = _neumann_segments(*map(np.array, ([0, 0, 0.0], [l, 0, 0.0], [1.5 * l, 0.02, 0], [0.5 * l, 0.02, 0])))
    assert got == pytest.approx(ref, rel=1e-9)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("r1,r2,dz", [(1.0, 1.0, 0.1), (1.0, 0.7, 0.0), (0.01, 0.0101, 1e-5), (1.0, 1.0, 40.0),
                                      (1.0, 2.0, 3.0)])
def test_maxwell_vs_neumann_quadrature(r1, r2, dz):
    def f(phi):
        return np.cos(phi) / np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(phi) + dz**2)

    val, _ = quad(f, 0, np.pi, epsabs=0, epsrel=1e-13, limit=500, points=[0.0])
    ref = MU0 * r1 * r2 * val  # mu0/(4 pi) r1 r2 * 2 pi * 2 int_0^pi
    assert C.maxwell_mutual(r1, r2, dz) == pytest.approx(ref, rel=1e-9)


def test_gmd_rectangle_vs_quadrature():
    for a, b in [(1.0, 1.0), (1.0, 0.25), (0.2, 5.0)]:
        f = lambda y, x: (a - x) * (b - y) * 0.5 * np.log(x * x + y * y)  # noqa: E731
        val, _ = dblquad(f, 0, a, 0, b, epsabs=1e-13, epsrel=1e-11)
        ref = np.exp(4 * val / (a * a * b * b))
        assert C.gmd_rectangle(a, b) == pytest.approx(ref, rel=1e-8)
    assert C.gmd_rectangle(1.0, 1.0) == pytest.approx(0.44705, abs=5e-6)  # Grover's tabulated square value
    assert C.gmd_rectangle(1.0, 0.0) == pytest.approx(np.exp(-1.5), rel=1e-14)


def test_nagaoka_table():
    """Nagaoka coefficients for l/D = 0.5, 1, 2, 5, 10 (Grover's table, 4 digits)."""
    for l_over_d, want in [(0.5, 0.5255), (1.0, 0.6884), (2.0, 0.8181), (5.0, 0.9201), (10.0, 0.9588)]:
        assert C.nagaoka(0.5, l_over_d) == pytest.approx(want, abs=1.5e-4)


# ------------------------------------------------------------------------------------------- gate C1


def _strip_ln_gmd(k, h):
    """ln GMD of two collinear strips of width h at centre distance k h (k = 0: self, h e^-3/2)."""
    k = np.asarray(k, float)
    f = lambda x: np.where(x > 0, x * x * np.log(np.where(x > 0, x, 1.0)), 0.0)  # noqa: E731
    return 0.5 * (f(k + 1) + f(np.abs(k - 1)) - 2 * f(k)) - 1.5 + np.log(h)


@pytest.mark.parametrize("radius,length", [(0.01, 0.01), (0.01, 0.02), (0.005, 0.05)])
def test_C1_single_layer_sheet_vs_lorenz(radius, length):
    """Gate C1a: the coil inductance engine (Maxwell rings + GMD distances) on a current sheet split into
    400 coaxial strips, thin-conductor limit, vs the exact Lorenz/Nagaoka formula: < 1e-4."""
    n = 400
    h = length / n
    k = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    L = np.sum(C.maxwell_mutual(radius, radius, np.exp(_strip_ln_gmd(k, h)))) / n**2
    ref = C.lorenz_sheet_inductance(radius, length, 1)
    assert abs(L - ref) / ref < 1e-4


def test_C1_thick_coil_vs_brooks_published():
    """Gate C1b: Brooks coil (square section c x c, inner radius c, uniform current density) through
    FoilCoil (32 radial turns x 32 axial sub-cells, square cells) vs Grover's published
    L = 1.6994e-6 * a * N^2 [H], a = 1.5 c mean radius [grover1946, p. 98]: < 1e-3."""
    c, n = 0.1, 32
    coil = C.FoilCoil(bore_a=2 * c, bore_b=2 * c, corner_radius=c, strip_width=c, foil_thickness=c / n,
                      insulation_thickness=0.0, turns=n, n_sub=(1, n))
    assert coil.is_circular
    ref = 1.6994e-6 * 1.5 * c * n**2
    assert abs(coil.inductance() - ref) / ref < 1e-3
    assert abs(coil.inductance() - ref) / ref < 1e-4  # achieved: 3.7e-5 (published value has 5 digits)


def test_single_round_loop_thin_wire_limit():
    """One turn of round wire, uniform current: L = mu0 R (ln(8R/r) - 7/4) + O((r/R)^2) [grover1946]."""
    R, r = 0.05, 5e-5
    coil = C.RoundWireCoil("circular", bore_radius=R - r, d_bare=2 * r, d_ins=2 * r, turns_per_layer=1, n_layers=1)
    ref = MU0 * R * (np.log(8 * R / r) - 1.75)
    assert coil.inductance() == pytest.approx(ref, rel=1e-5)


def test_square_loop_partial_inductance_closed_form():
    """Sharp square filament loop with GMD g vs 4 (Lp(a, g) - Mp(a, a)) [grover1946, ruehli1972]."""
    a, g = 0.1, 1e-5
    v = C.outline_polygon(a / 2, a / 2, 0.0, 1)
    fs = C.FilamentSet([np.hstack([v, np.zeros((len(v), 1))])], [np.full(len(v), g)], np.ones(1), np.zeros(1, int),
                       np.array([[0.0, 0, 1]]))
    Lp = MU0 / (2 * np.pi) * (a * np.arcsinh(a / g) - np.hypot(a, g) + g)
    Mp = MU0 / (2 * np.pi) * (a * np.arcsinh(1.0) - np.hypot(a, a) + a)
    assert C.circuit_inductance(fs)[0, 0] == pytest.approx(4 * (Lp - Mp), rel=1e-8)


def test_polygon_circle_converges_to_maxwell():
    """Rounded-rectangle machinery with no straight part, polygonised, vs the exact circle."""
    R, g = 0.01, 1e-4
    ref = C.maxwell_mutual(R, R, g)
    errs = []
    for n_arc in (4, 8, 16, 32):
        v = C.outline_polygon(0.0, 0.0, R, n_arc)
        fs = C.FilamentSet([np.hstack([v, np.zeros((len(v), 1))])], [np.full(len(v), g)], np.ones(1),
                           np.zeros(1, int), np.array([[0.0, 0, 1]]))
        errs.append(abs(C.circuit_inductance(fs)[0, 0] - ref) / ref)
    assert errs[-1] < 2e-3 and all(e2 < e1 for e1, e2 in zip(errs, errs[1:]))


def test_cached_inductance_equals_plain_filament_sum():
    """Coil.inductance (shape/dz cache + quadrant symmetry) == plain sum over all segment pairs."""
    coil = C.RoundWireCoil("rectangular", bore_a=8e-3, bore_b=5e-3, corner_radius=1e-3, d_bare=0.5e-3, d_ins=0.55e-3,
                           turns_per_layer=3, n_layers=2, former_build=0.1e-3)
    fs = coil.filaments()
    assert coil.inductance() == pytest.approx(C.circuit_inductance(fs)[0, 0], rel=1e-10)
    foil = C.FoilCoil(bore_a=6e-3, bore_b=4e-3, corner_radius=0.5e-3, strip_width=2e-3, foil_thickness=0.2e-3,
                      insulation_thickness=0.05e-3, turns=3, n_sub=(1, 4))
    assert foil.inductance() == pytest.approx(C.circuit_inductance(foil.filaments())[0, 0], rel=1e-10)


def test_mutual_reciprocity_and_coaxial_fast_path():
    c1 = C.RoundWireCoil("circular", bore_radius=5e-3, d_bare=0.5e-3, d_ins=0.55e-3, turns_per_layer=4, n_layers=2)
    c2 = C.RoundWireCoil("circular", bore_radius=8e-3, d_bare=0.5e-3, d_ins=0.55e-3, turns_per_layer=3, n_layers=1,
                         position=(0, 0, 4e-3))
    m12, m21 = c1.mutual(c2), c2.mutual(c1)
    assert m12 == pytest.approx(m21, rel=1e-12)
    # same pair, slightly tilted so the polygon path is used: continuity within polygon error
    c3 = C.RoundWireCoil("circular", bore_radius=8e-3, d_bare=0.5e-3, d_ins=0.55e-3, turns_per_layer=3, n_layers=1,
                         position=(0, 0, 4e-3), rotation=Rotation.from_rotvec([1e-6, 0, 0]))
    assert c1.mutual(c3) == pytest.approx(m12, rel=1e-3)
    # passivity: k < 1
    assert abs(m12) < np.sqrt(c1.inductance() * c2.inductance())


def test_resistance_mass_heat_capacity():
    coil = C.RoundWireCoil("circular", bore_radius=5e-3, d_bare=1e-3, d_ins=1.1e-3, turns_per_layer=4, n_layers=3,
                           lead_length=0.05)
    length = sum(2 * np.pi * (5e-3 + 0.55e-3 + k * 1.1e-3) * 4 for k in range(3)) + 0.05
    area = np.pi * 0.25e-6
    assert coil.conductor_length() == pytest.approx(length, rel=1e-12)
    assert coil.resistance(293.15) == pytest.approx(RHO_CU_20 * length / area, rel=1e-12)
    assert coil.resistance(393.15) == pytest.approx(RHO_CU_20 * (1 + 100 * ALPHA_CU) * length / area, rel=1e-12)
    assert coil.mass() == pytest.approx(DENSITY_CU * length * area, rel=1e-12)
    assert coil.heat_capacity() == pytest.approx(coil.mass() * CP_CU, rel=1e-12)
    lump = coil.lumped(300.0)
    assert lump["R0"] == pytest.approx(coil.resistance(293.15)) and lump["T0"] == 300.0


def test_dowell_limits_and_skin_report():
    D = np.array([1e-3, 0.05, 0.3])
    for m in (1, 3, 10):
        assert np.allclose(C.dowell_factor(D, m), 1 + (5 * m * m - 1) * D**4 / 45, rtol=2e-4)
    # single layer = exact 1-D foil skin solution, high-frequency asymptote ~ D
    assert C.dowell_factor(30.0, 1) == pytest.approx(30.0, rel=1e-9)
    foil = C.reference_coil_r1()
    rep = foil.skin_report(100.0)
    assert rep["skin_depth_m"] == pytest.approx(np.sqrt(RHO_CU_20 / (np.pi * 100.0 * MU0)), rel=1e-12)
    assert rep["dowell_FR"] < 1.05
    with pytest.warns(UserWarning, match="Dowell factor"):
        foil.skin_report(20e3)


# ------------------------------------------------------------------------------------------- gate C2


def test_C2_loop_on_axis_field():
    R = 7e-3
    coil = C.RoundWireCoil("circular", bore_radius=R - 0.25e-3, d_bare=0.5e-3, d_ins=0.5e-3, turns_per_layer=1,
                           n_layers=1)
    z = np.linspace(-0.05, 0.05, 41)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    H = coil.field_per_amp(pts)
    ref = C.loop_axis_field(R, z)
    assert np.max(np.abs(H[:, 2] - ref) / ref) < 1e-8
    assert np.max(np.abs(H[:, :2])) < 1e-8 * ref.max()


def test_C2_finite_solenoid_on_axis_field():
    """Single-layer solenoid of N discrete turns (posed) vs the closed-form sum of loop fields."""
    R0, d, N = 5e-3, 0.5e-3, 12
    rot = Rotation.from_euler("xyz", [30, -20, 50], degrees=True)
    pos = np.array([1e-3, -2e-3, 3e-3])
    coil = C.RoundWireCoil("circular", bore_radius=R0, d_bare=d, d_ins=d, turns_per_layer=N, n_layers=1,
                           position=pos, rotation=rot)
    z = np.linspace(-0.03, 0.03, 31)
    pts = pos + rot.apply(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))
    Hb = rot.inv().apply(coil.field_per_amp(pts))
    zt = (np.arange(N) + 0.5) * d - N * d / 2
    ref = sum(C.loop_axis_field(R0 + d / 2, z - zk) for zk in zt)
    assert np.max(np.abs(Hb[:, 2] - ref) / ref) < 1e-8


def test_C2_square_sheet_on_axis_closed_form():
    """Finite-width square current sheet on its axis: H = 2/(pi w) [atan(Z/sqrt(2A^2+Z^2))]_(z-z_hi)^(z-z_lo)."""
    A, w = 0.01, 0.005
    v = C.outline_polygon(A, A, 0.0, 1)
    P1, P2 = v, np.roll(v, -1, axis=0)
    z = np.linspace(-0.03, 0.03, 25)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    H = C.strip_sheet_field(pts, P1, P2, -w / 2, w / 2)
    F = lambda Z: np.arctan(Z / np.sqrt(2 * A * A + Z * Z))  # noqa: E731
    ref = 2 / (np.pi * w) * (F(z + w / 2) - F(z - w / 2))
    assert np.max(np.abs(H[:, 2] - ref) / ref) < 1e-8


def test_C2_circular_sheet_on_axis_closed_form():
    """Circular foil turn (thin) on its axis vs the finite current-sheet solenoid formula."""
    R, w, t = 6e-3, 4e-3, 1e-6
    foil = C.FoilCoil(bore_a=2 * R, bore_b=2 * R, corner_radius=R, strip_width=w, foil_thickness=t,
                      insulation_thickness=0, turns=1, former_build=-t / 2)
    z = np.linspace(-0.02, 0.02, 21)
    H = foil.field_per_amp(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))
    ref = 1 / (2 * w) * ((z + w / 2) / np.hypot(R, z + w / 2) - (z - w / 2) / np.hypot(R, z - w / 2))
    # Gauss nodes across t shift the radius by O(t^2/R^2) ~ 1e-8
    assert np.max(np.abs(H[:, 2] - ref) / ref) < 1e-7


def test_strip_sheet_vs_magpylib_cuboid_off_axis():
    """Rectangular sheet = side wall of a uniformly polarised cuboid (independent magpylib kernel)."""
    a, b, w = 0.02, 0.012, 0.005
    v = C.outline_polygon(a / 2, b / 2, 0.0, 1)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.02, 0.02, size=(200, 3))
    ins = (np.abs(np.abs(pts[:, 0]) - a / 2) < 1e-4) | (np.abs(np.abs(pts[:, 1]) - b / 2) < 1e-4)
    pts = pts[~ins]
    H = C.strip_sheet_field(pts, v, np.roll(v, -1, axis=0), -w / 2, w / 2, current=1.0)
    ref = cuboid_field("B", pts, (a, b, w), (0, 0, MU0 / w)) / MU0
    assert np.max(np.linalg.norm(H - ref, axis=1)) < 1e-8 * np.max(np.linalg.norm(ref, axis=1))


def test_field_and_mutual_reciprocity():
    """Flux of the coil field through a tiny test loop = coil.mutual(test loop): magpylib field vs
    Neumann/Maxwell mutual (independent kernels)."""
    coil = C.reference_coil_r2(n_layers=3)
    r = 1e-4
    for p in ([0, 0, 0], [6e-3, 2e-3, 1e-3], [0, 0, 8e-3]):
        v = C.outline_polygon(0, 0, r, 64)
        loop = np.hstack([v, np.zeros((len(v), 1))]) + p
        fs = C.FilamentSet([loop], [np.full(len(v), 1e-6)], np.ones(1), np.zeros(1, int), np.array([[0, 0, 1.0]]))
        area = 0.5 * np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
        flux = MU0 * coil.field_per_amp(np.array([p], float))[0, 2] * area
        assert coil.mutual(fs)[0] == pytest.approx(flux, rel=2e-4)


# ------------------------------------------------------------------------------------------- gate C3


def test_C3_r1_foil_coil_vs_old_validated_model():
    """R1: L and R(20 C) vs `magnet_pulse_sim.py`. Its 16.84 uH is L_air = 16.7093 uH (re-run of
    make_model, radial_quadrature 6-96: 16.7093-16.7090 uH) plus the N52 reversible-recoil term
    mu0 V chi_eff <h>^2 (+0.80 %), which magsim keeps in the magnet model, not in the coil."""
    coil = C.reference_coil_r1()
    L, R = coil.inductance(), coil.resistance(293.15)
    assert abs(L - 16.7093e-6) / 16.7093e-6 < 1e-3  # air-core: achieved 2e-5
    assert abs(L - 16.84e-6) / 16.84e-6 < 1e-2  # documented value incl. recoil: 0.78 %
    assert abs(R - 38.6129e-3) / 38.6129e-3 < 1e-4
    rep = coil.geometry_report()
    assert rep["conductor_length_m"] == pytest.approx(2.23960, rel=1e-5)


@pytest.mark.slow
def test_C3_r1_offset_corners_variant():
    """Offset (rounded) foil corners shorten the conductor by sum (8 - 2 pi) d_k: L and R drop."""
    sharp, off = C.reference_coil_r1(), C.reference_coil_r1(corners="offset")
    assert off.resistance() < sharp.resistance() and off.inductance() < sharp.inductance()
    dk = 0.1e-3 + np.arange(22) * 0.25e-3
    assert sharp.conductor_length() - off.conductor_length() == pytest.approx((8 - 2 * np.pi) * dk.sum(), rel=1e-9)


def test_reference_coils_build():
    r2, r3 = C.reference_coil_r2(), C.reference_coil_r3()
    assert r2.n_turns == 100 and r3.n_turns == 126 and r3.is_circular and not r2.is_circular
    # R2 conductor length = old geometry.py without helix/crossovers (12.78301 m incl. 2 x 30 mm leads)
    assert r2.conductor_length() == pytest.approx(12.78301, rel=1e-6)
    assert r2.geometry_report()["helix_length_rel_error_max"] < 2e-4
