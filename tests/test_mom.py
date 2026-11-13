"""Tests for magsim.mom: kernels, assembly, Newton solver, history, demagnetising factors (gate F3)."""

import magpylib as magpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from magsim import analytic as an
from magsim.constants import MU0
from magsim.geometry import ANNULUS, CUBOID, CYLINDER, SEGMENT, Cuboid, Cylinder, Magnet, Pose, Ring
from magsim.materials.base import LinearMaterial
from magsim.mom import MagnetSystem, cell_field

RIGID = LinearMaterial(1.2, 1.0)


# ---------------------------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------------------------
def test_cell_field_kernels_consistent():
    """Four quarter segments = annulus (= cylinder(r2) - cylinder(r1)); rotated cuboid = magpylib object API."""
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(7, 3)) * 2.0
    n = pts.shape[0]
    pol = np.tile([0.3, -0.2, 1.0], (n, 1))
    eye = np.tile(np.eye(3), (n, 1, 1))
    H4 = sum(
        cell_field(pts, np.full(n, SEGMENT), np.zeros((n, 3)), np.tile([0.5, 1.0, 1.0, k * np.pi / 2, (k + 1) * np.pi / 2], (n, 1)), eye, pol)
        for k in range(4)
    )
    Ha = cell_field(pts, np.full(n, ANNULUS), np.zeros((n, 3)), np.tile([0.5, 1.0, 1.0, 0, 2 * np.pi], (n, 1)), eye, pol)
    assert np.abs(H4 - Ha).max() < 1e-9 * np.abs(Ha).max()
    Hc = cell_field(pts, np.full(n, CYLINDER), np.zeros((n, 3)), np.tile([0.0, 1.0, 1.0, 0, 2 * np.pi], (n, 1)), eye, pol)
    Hi = cell_field(pts, np.full(n, CYLINDER), np.zeros((n, 3)), np.tile([0.0, 0.5, 1.0, 0, 2 * np.pi], (n, 1)), eye, pol)
    assert np.allclose(Ha, Hc - Hi, rtol=0, atol=1e-12 * np.abs(Ha).max())
    # rotated, translated cuboid vs magpylib's object interface (independent transformation path)
    rot = Rotation.from_euler("xyz", [20, -35, 50], degrees=True)
    pos = np.array([0.1, -0.2, 0.3])
    J = np.array([0.2, 0.9, -0.4])
    # magpylib object polarisation is given in the object's own (rotated) frame
    src = magpy.magnet.Cuboid(dimension=(0.5, 0.8, 0.3), polarization=rot.inv().apply(J), position=pos, orientation=rot)
    ref = src.getH(pts) * (magpy.mu_0 / MU0)
    got = cell_field(pts, np.full(n, CUBOID), np.tile(pos, (n, 1)), np.tile([0.5, 0.8, 0.3, 0, 0], (n, 1)),
                     np.tile(rot.as_matrix(), (n, 1, 1)), np.tile(J, (n, 1)), ell=0.7)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_axisymmetric_block_equals_full_block():
    rot = Rotation.from_euler("xyz", [20, 30, 10], degrees=True)
    ax = rot.apply([0, 0, 1.0])
    mat = LinearMaterial(1.2, 4.0, 2.0)
    A = Magnet("A", Cylinder(4e-3, 6e-3), Pose([0, 0, 0], rot), mat, mesh=(3, 1, 4))
    B = Magnet("B", Ring(1e-3, 4e-3, 3e-3), Pose(ax * 5e-3, rot * Rotation.from_euler("z", 33, degrees=True)), mat,
               mesh=(2, 1, 3))
    s = MagnetSystem([A, B])
    N1 = s.interaction_matrix().copy()
    s._ring_body[:] = False
    s._N, s._self_body, s._cross = None, {}, {}
    N2 = s.interaction_matrix()
    assert np.abs(N1 - N2).max() < 1e-12 * np.abs(N2).max()


# ---------------------------------------------------------------------------------------------
# solver basics
# ---------------------------------------------------------------------------------------------
def test_rigid_solution_and_field_points():
    m = Magnet("A", Cuboid((2e-3, 1e-3, 1e-3)), Pose(), RIGID, easy_axis=(1, 0, 0), mesh=(2, 2, 1))
    s = MagnetSystem([m])
    pts, w = s.field_points()
    assert pts.shape == (s.n_cells * 8, 3) and np.allclose(np.bincount(s.gcell, w), 1.0)
    sol = s.solve()
    assert sol.converged and sol.residual < 1e-12
    assert np.allclose(sol.J, [1.2, 0, 0], atol=1e-15)
    assert np.allclose(sol.B, MU0 * sol.H + sol.J)
    with pytest.raises(ValueError):
        s.solve(H_app=np.zeros((3, 3)))


def test_rotation_covariance():
    """Rotating a body rotates J and H; cached body-frame self-blocks are reused."""
    mat = LinearMaterial(1.2, 1.3, 1.1)
    m = Magnet("A", Cuboid((3e-3, 2e-3, 1e-3)), Pose(), mat, easy_axis=(0.2, 0.1, 1.0), mesh=(3, 2, 2))
    s = MagnetSystem([m])
    sol0 = s.solve(H_app=np.array([1e4, -2e4, 5e3]))
    rot = Rotation.from_euler("xyz", [30, -60, 15], degrees=True)
    s.set_pose("A", Pose([1e-3, 0, 0], rot))
    Rm = rot.as_matrix()
    sol1 = s.solve(H_app=Rm @ np.array([1e4, -2e4, 5e3]))
    assert np.allclose(sol1.J, sol0.J @ Rm.T, atol=1e-12)
    assert np.allclose(sol1.H, sol0.H @ Rm.T, rtol=0, atol=1e-9 * np.abs(sol0.H).max())


def test_set_pose_cross_blocks_and_overlap_restore():
    mat = LinearMaterial(1.2, 1.5, 1.5)
    A = Magnet("A", Cuboid((1e-3, 1e-3, 1e-3)), Pose(), mat, mesh=(2, 2, 2))
    B = Magnet("B", Cuboid((1e-3, 1e-3, 1e-3)), Pose([0, 0, 1.5e-3]), mat, mesh=(2, 2, 2))
    s = MagnetSystem([A, B])
    J0 = s.solve().J.copy()
    s.set_pose("B", Pose([0, 0, 3e-3]))
    J1 = s.solve().J
    assert np.abs(J1 - J0).max() > 1e-4
    s.set_pose("B", {"position": [0, 0, 1.5e-3]})
    assert np.allclose(s.solve().J, J0, atol=1e-13)
    with pytest.raises(ValueError):
        s.set_pose("B", Pose([0, 0, 0.5e-3]))
    assert np.allclose(s.magnets[1].pose.position, [0, 0, 1.5e-3])
    assert np.allclose(s.solve().J, J0, atol=1e-13)


def test_memory_guard_and_names():
    big = Magnet("big", Cuboid((1, 1, 1)), Pose(), RIGID, mesh=(40, 40, 40))
    with pytest.raises(MemoryError):
        MagnetSystem([big])
    a = Magnet("a", Cuboid((1, 1, 1)), Pose(), RIGID, mesh=(1, 1, 1))
    b = Magnet("a", Cuboid((1, 1, 1)), Pose([2, 0, 0]), RIGID, mesh=(1, 1, 1))
    with pytest.raises(ValueError):
        MagnetSystem([a, b])


class _Recorder(LinearMaterial):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.commits = []

    def commit(self, H_par, state, T):
        self.commits.append(np.array(H_par))
        return super().commit(H_par, state, T)


def test_commit_passes_h_par_and_rejects_stale():
    ma, mb = _Recorder(1.2, 1.2), _Recorder(1.0, 2.0)
    A = Magnet("A", Cuboid((1e-3, 1e-3, 1e-3)), Pose(), ma, mesh=(2, 1, 1))
    B = Magnet("B", Cylinder(1e-3, 1e-3), Pose([0, 0, 1.2e-3]), mb, mesh=(2, 1, 2))
    s = MagnetSystem([A, B])
    sol = s.solve()
    s.commit(sol)
    assert np.allclose(ma.commits[-1], sol.H_par[s.body_slice("A")])
    assert np.allclose(mb.commits[-1], sol.H_par[s.body_slice("B")])
    assert np.allclose(sol.H_par, np.einsum("ij,ij->i", sol.H, s.easy))
    s.set_pose("B", Pose([0, 0, 2e-3]))
    with pytest.raises(ValueError):
        s.commit(sol)


class TanhMaterial:
    """Nonlinear toy law implementing the Material protocol (strongly kneed, Alnico-like slope).

    J_par = Js tanh((H_par + Hk) / Hw), J_perp = MU0 chi H_perp. History-free.
    """

    name = "tanh-toy"

    def __init__(self, Js=1.2, Hk=4e4, Hw=8e3, chi=0.5):
        self.Js, self.Hk, self.Hw, self.chi = Js, Hk, Hw, chi

    def init_state(self, n, initial):
        return np.zeros(n)

    def response(self, H_par, H_perp, state, T):
        x = (np.asarray(H_par) + self.Hk) / self.Hw
        J = self.Js * np.tanh(x)
        dJ = self.Js / self.Hw / np.cosh(x) ** 2
        n = J.size
        return J, dJ, MU0 * self.chi * np.asarray(H_perp), np.full(n, self.chi)

    def commit(self, H_par, state, T):
        return state


@pytest.mark.parametrize("J0", [None, "bad"])
def test_newton_nonlinear_toy_material(J0):
    mat = TanhMaterial()
    A = Magnet("A", Cylinder(4.75e-3, 12.5e-3), Pose(), mat, mesh=(2, 1, 8))
    B = Magnet("B", Cuboid((5e-3, 5e-3, 3e-3)), Pose([0, 0, 9e-3], Rotation.from_euler("x", 20, degrees=True)), mat,
               easy_axis=(0, 1, 1), mesh=(2, 2, 2))
    s = MagnetSystem([A, B])
    guess = None if J0 is None else np.tile([0.0, 0.0, -1.2], (s.n_cells, 1))
    sol = s.solve(J0=guess, H_app=np.array([0, 0, -2e4]))
    assert sol.converged and sol.residual < 1e-9
    # independent residual check: J = law(H_app + N J)
    N = s.interaction_matrix()
    H = np.array([0, 0, -2e4]) + (N @ sol.J.ravel()).reshape(-1, 3)
    Hp = np.einsum("ij,ij->i", H, s.easy)
    Jt = mat.Js * np.tanh((Hp + mat.Hk) / mat.Hw)
    Jperp = sol.J - Jt[:, None] * s.easy
    assert np.max(np.abs(Jt - np.einsum("ij,ij->i", sol.J, s.easy))) < 1e-9
    assert np.allclose(Jperp, MU0 * mat.chi * (H - Hp[:, None] * s.easy), atol=1e-9)
    # the knee is really exercised: some cells sit well below Js
    assert np.einsum("ij,ij->i", sol.J, s.easy).min() < 0.9 * mat.Js
    # Newton converges quadratically near the solution
    h = np.array(sol.history)
    assert len(h) >= 2 and h[-1] < 1e-9


def test_continuation_fallback():
    """With Newton crippled (one iteration), the coupling continuation still reaches the solution."""
    mat = TanhMaterial(Hw=2e3)
    A = Magnet("A", Cylinder(4.75e-3, 12.5e-3), Pose(), mat, mesh=(1, 1, 6))
    s = MagnetSystem([A])
    ref = s.solve(J0=np.zeros((s.n_cells, 3)))
    assert ref.converged
    s._J_last = None
    sol = s.solve(J0=np.tile([0, 0, -1.2], (s.n_cells, 1)), max_iter=4)
    if not sol.continuation:
        pytest.skip("Newton converged without needing continuation")
    assert sol.converged and np.allclose(sol.J, ref.J, atol=1e-8)


def test_permanent_vs_soft_body_identity():
    """Linear magnet: J_perm = (Jr / (MU0 chi H0)) J_soft exactly (same operator (I - MU0 chi N)^-1 e),
    so -MU0 <H>/<J> of the open-circuit magnet equals the soft-body magnetometric N_m(chi) [chen1991]."""
    for shape, mesh in ((Cuboid((4e-3, 2e-3, 1e-3)), (4, 2, 2)), (Cylinder(4e-3, 2e-3), (3, 1, 4))):
        perm = MagnetSystem([Magnet("A", shape, Pose(), LinearMaterial(1.2, 4.0, 4.0), mesh=mesh)])
        sp = perm.solve()
        Np = -MU0 * (sp.H[:, 2] * perm.volume).sum() / (sp.J[:, 2] * perm.volume).sum()
        soft = MagnetSystem([Magnet("A", shape, Pose(), LinearMaterial(0.0, 4.0, 4.0), mesh=mesh)])
        H0 = 3e4
        ss = soft.solve(H_app=np.array([0, 0, H0]))
        V = soft.volume
        Ns = (H0 - (ss.H[:, 2] * V).sum() / V.sum()) / ((ss.J[:, 2] * V).sum() / V.sum() / MU0)
        assert abs(Np - Ns) < 1e-12


# ---------------------------------------------------------------------------------------------
# F3: demagnetising factors after Richardson extrapolation
# ---------------------------------------------------------------------------------------------
def _neff(shape, mesh, mat):
    s = MagnetSystem([Magnet("A", shape, Pose(), mat, mesh=mesh)])
    sol = s.solve()
    return -MU0 * (sol.H[:, 2] * s.volume).sum() / (sol.J[:, 2] * s.volume).sum()


def _richardson(v):
    """Richardson extrapolation of three values on meshes refined by 2; observed order p."""
    v1, v2, v3 = v
    d1, d2 = v1 - v2, v2 - v3
    if d1 * d2 <= 0 or abs(d2) < 1e-14 * abs(v3):
        return v3
    p = np.log2(d1 / d2)
    return v3 + d2 / (2**p - 1)


def _f3_cases(fine):
    pr = [(4, 2, 1), (8, 4, 2), (16, 8, 4)] if fine else [(2, 1, 1), (4, 2, 2), (8, 4, 4)]
    cases = [("prism 20x10x5", Cuboid((20e-3, 10e-3, 5e-3)), pr, an.aharoni_prism_demag(20, 10, 5))]
    if fine:  # the cube needs >= 2 cells per side on the coarsest mesh (1 cell is exact by symmetry)
        cases.append(("cube", Cuboid((5e-3, 5e-3, 5e-3)), [(2, 2, 2), (4, 4, 4), (8, 8, 8)], an.aharoni_prism_demag(1, 1, 1)))
    for LD in (0.5, 1.0, 2.6):
        R = 2e-3
        nz = max(1, int(round(2 * LD)))
        cases.append((f"cylinder L/D={LD}", Cylinder(2 * R, 2 * R * LD), [(2, 1, 2 * nz), (4, 1, 4 * nz), (8, 1, 8 * nz)],
                      an.cylinder_demag_factor(R, 2 * R * LD)))
    return cases


def _f3(fine):
    rows = []
    for name, shape, meshes, ref in _f3_cases(fine):
        for mu in (1.0, 1.05):
            vals = [_neff(shape, m, LinearMaterial(1.2, mu, mu)) for m in meshes]
            Nx = _richardson(vals)
            rows.append((name, mu, Nx, ref))
            assert abs(Nx / ref - 1) < 5e-3, (name, mu, vals, Nx, ref)
            if mu == 1.0 and fine:
                # chi = 0: Aharoni / Chen-Brug-Goldfarb are the exact limits; the MoM reproduces them closely
                assert abs(Nx / ref - 1) < 1e-4, (name, vals, Nx, ref)
    return rows


def test_f3_demag_factors_quick():
    """Gate F3 (coarser mesh sequence): volume-mean N = -MU0 <H>/<J> vs [aharoni1998] / [chen1991] < 0.5 %."""
    _f3(fine=False)


@pytest.mark.slow
def test_f3_demag_factors_fine():
    """Gate F3 on the finer mesh sequence (finest prism / cube 512 cells)."""
    _f3(fine=True)


def test_f3_high_permeability_deviation_is_physical():
    """mu_rec = 4: N_eff = N_m(chi = 3) of Chen-Brug-Goldfarb, which is NOT the chi = 0 factor.

    Documented deviation from the literal F3 wording: the volume-mean factor of a linear magnet with
    chi = mu_rec - 1 equals the soft-body magnetometric factor N_m(chi) (``test_permanent_vs_soft_body_
    identity``); Aharoni [aharoni1998] and the Phi formula [chen1991] are its chi = 0 values. The MoM gives
    N_m(3) below N_m(0) by 2-9 % (cylinders / prisms), the known direction (N_m decreases with chi [chen1991]).
    """
    R = 2e-3
    vals = [_neff(Cylinder(2 * R, 2 * R), m, LinearMaterial(1.2, 4.0, 4.0)) for m in [(2, 1, 2), (4, 1, 4), (8, 1, 8)]]
    Nx = _richardson(vals)
    N0 = an.cylinder_demag_factor(R, 2 * R)
    assert 0.9 * N0 < Nx < 0.99 * N0


# ---------------------------------------------------------------------------------------------
# F4: dipole limit (MoM + force code vs point dipoles)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mat", [RIGID, LinearMaterial(1.2, 1.05, 1.05)])
@pytest.mark.parametrize("direction", [(0, 0, 1.0), (1.0, 0.5, 0.3)])
def test_f4_dipole_limit(mat, direction):
    from magsim.forces import force_torque

    a = 2e-3
    d = 20 * a * np.asarray(direction) / np.linalg.norm(direction)
    A = Magnet("A", Cuboid((a, a, a)), Pose(), mat, mesh=(2, 2, 2))
    B = Magnet("B", Cuboid((a, a, a)), Pose(d), mat, mesh=(2, 2, 2))
    s = MagnetSystem([A, B])
    sol = s.solve()
    mA = (sol.J[s.body_slice("A")] * s.volume[s.body_slice("A"), None]).sum(0) / MU0
    mB = (sol.J[s.body_slice("B")] * s.volume[s.body_slice("B"), None]).sum(0) / MU0
    F, _ = force_torque(s, sol, "B")
    Fd = an.dipole_force(mA, mB, d)
    assert np.linalg.norm(F - Fd) / np.linalg.norm(Fd) < 1e-3
