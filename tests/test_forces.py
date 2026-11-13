"""Force / torque gates F1, F2, F5, F6 (docs/SPEC.md) for magsim.forces."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from magsim import analytic as an
from magsim.constants import MU0
from magsim.forces import force_torque, force_torque_detail
from magsim.geometry import Cuboid, Cylinder, Magnet, Pose, Ring
from magsim.materials.base import LinearMaterial
from magsim.mom import MagnetSystem

RIGID = LinearMaterial(1.3, 1.0)
NDFEB = LinearMaterial(1.3, 1.05, 1.05)
D1 = np.array([20e-3, 10e-3, 5e-3])
D2 = np.array([16e-3, 8e-3, 4e-3])


def _cuboid_pair(off, mat=RIGID, meshes=((2, 2, 2), (2, 1, 2)), rot=None, center=np.zeros(3)):
    pa, pb = Pose(), Pose(off)
    if rot is not None:
        pa, pb = pa.rotated(rot, center), pb.rotated(rot, center)
    A = Magnet("A", Cuboid(D1), pa, mat, mesh=meshes[0])
    B = Magnet("B", Cuboid(D2), pb, mat, mesh=meshes[1])
    s = MagnetSystem([A, B])
    return s, s.solve()


# ---------------------------------------------------------------------------------------------
# F1: rigid aligned cuboids vs Akoun-Yonnet
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gapf", [0.0, 1e-4, 1e-2, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("lat", [(0.0, 0.0), (3e-3, -2e-3), (17e-3, 4e-3)])
def test_f1_rigid_cuboids_vs_akoun_yonnet(gapf, lat):
    size = D1.max()
    off = np.array([lat[0], lat[1], 0.5 * (D1[2] + D2[2]) + gapf * size])
    s, sol = _cuboid_pair(off)
    F, _ = force_torque(s, sol, "B")
    Fa = an.akoun_yonnet_force(1.3, 1.3, D1, D2, off)
    scale = np.linalg.norm(Fa)
    assert np.all(np.abs(F - Fa) < 1e-5 * scale), (F, Fa)
    assert np.linalg.norm(F - Fa) / scale < 1e-5


def test_f1_continuity_at_contact():
    off0 = np.array([3e-3, -2e-3, 0.5 * (D1[2] + D2[2])])
    s, sol = _cuboid_pair(off0)
    F0, _ = force_torque(s, sol, "B")
    for g in (1e-12, 1e-9, 1e-7):
        s, sol = _cuboid_pair(off0 + [0, 0, g * D1.max()])
        Fg, _ = force_torque(s, sol, "B")
        assert np.linalg.norm(Fg - F0) / np.linalg.norm(F0) < 10 * g + 1e-10


@pytest.mark.parametrize("gapf,lat", [(0.0, (3e-3, -2e-3)), (1e-2, (0, 0)), (0.1, (3e-3, -2e-3))])
def test_f1_quadrature_path_cross_check(gapf, lat):
    """The pure quadrature path (no closed form) reproduces Akoun-Yonnet too (refined rule)."""
    off = np.array([lat[0], lat[1], 0.5 * (D1[2] + D2[2]) + gapf * D1.max()])
    s, sol = _cuboid_pair(off)
    r = force_torque_detail(s, sol, "B", closed_form=False, level=1)
    Fa = an.akoun_yonnet_force(1.3, 1.3, D1, D2, off)
    assert np.linalg.norm(r.F - Fa) / np.linalg.norm(Fa) < 1e-6


def test_f1_rotated_frame():
    """Rotating the whole aligned pair rotates the force (closed-form path with rotated frames)."""
    rot = Rotation.from_euler("xyz", [25, -40, 70], degrees=True)
    off = np.array([3e-3, -2e-3, 0.5 * (D1[2] + D2[2]) + 1e-4])
    s, sol = _cuboid_pair(off, rot=rot, center=[1e-3, 2e-3, -5e-3])
    F, _ = force_torque(s, sol, "B")
    Fa = rot.apply(an.akoun_yonnet_force(1.3, 1.3, D1, D2, off))
    assert np.linalg.norm(F - Fa) / np.linalg.norm(Fa) < 1e-5


# ---------------------------------------------------------------------------------------------
# F2: rigid coaxial cylinders vs Lipschitz-Hankel
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gapf", [0.0, 0.01, 0.1, 1.0, 5.0])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 3)])
def test_f2_rigid_coaxial_cylinders(gapf, mesh):
    R, L1, L2 = 2.5e-3, 6e-3, 4e-3
    gap = gapf * R
    A = Magnet("A", Cylinder(2 * R, L1), Pose([0, 0, -L1 / 2]), LinearMaterial(1.2, 1.0), mesh=mesh)
    B = Magnet("B", Cylinder(2 * R, L2), Pose([0, 0, gap + L2 / 2]), LinearMaterial(1.2, 1.0), mesh=mesh)
    s = MagnetSystem([A, B])
    F, T = force_torque(s, s.solve(), "B")
    ref = an.coaxial_cylinder_force(1.2, 1.2, R, L1, L2, gap)
    assert abs(F[2] / ref - 1) < 1e-4
    assert np.linalg.norm(F[:2]) < 1e-9 * abs(ref) and np.linalg.norm(T) < 1e-9 * abs(ref) * L2


@pytest.mark.slow
def test_f2_segment_mesh():
    """Same gate with azimuthal segments (slow exact segment kernel, ~100 us per evaluation)."""
    R, L1, L2, gap = 2.5e-3, 6e-3, 4e-3, 0.1 * 2.5e-3
    A = Magnet("A", Cylinder(2 * R, L1), Pose([0, 0, -L1 / 2]), LinearMaterial(1.2, 1.0), mesh=(2, 4, 2))
    B = Magnet("B", Cylinder(2 * R, L2), Pose([0, 0, gap + L2 / 2]), LinearMaterial(1.2, 1.0), mesh=(2, 4, 2))
    s = MagnetSystem([A, B])
    F, _ = force_torque(s, s.solve(), "B")
    assert abs(F[2] / an.coaxial_cylinder_force(1.2, 1.2, R, L1, L2, gap) - 1) < 1e-4


# ---------------------------------------------------------------------------------------------
# F5: Newton's third law, isolated system, virtual work, frame invariance
# ---------------------------------------------------------------------------------------------
def _tilted_pair(mat, rot_all=None, center=np.zeros(3)):
    rot = Rotation.from_euler("xyz", [12, -7, 20], degrees=True)
    pa, pb = Pose(), Pose([1.5e-3, 0.8e-3, 7.5e-3], rot)
    if rot_all is not None:
        pa, pb = pa.rotated(rot_all, center), pb.rotated(rot_all, center)
    A = Magnet("A", Cuboid((10e-3, 6e-3, 4e-3)), pa, mat, mesh=(3, 2, 2))
    B = Magnet("B", Cuboid((8e-3, 5e-3, 3e-3)), pb, mat, easy_axis=(0.1, 0, 1), mesh=(3, 2, 2))
    s = MagnetSystem([A, B])
    return s, s.solve()


@pytest.mark.parametrize("mat", [RIGID, LinearMaterial(1.2, 1.05, 1.05), LinearMaterial(1.2, 4.0, 2.0)])
def test_f5_newton_third_law_and_isolated_system(mat):
    s, sol = _tilted_pair(mat)
    p = np.array([1e-3, -2e-3, 3e-3])
    FA, TA = force_torque(s, sol, "A", pivot=p)
    FB, TB = force_torque(s, sol, "B", pivot=p)
    assert np.linalg.norm(FA + FB) / np.linalg.norm(FB) < 1e-6
    assert np.linalg.norm(TA + TB) / np.linalg.norm(TB) < 1e-6


@pytest.mark.parametrize("gap", [0.0, 1e-5])
def test_f5_newton_third_law_aligned_linear_contact(gap):
    """Closed-form + perpendicular-charge quadrature, at contact, for a non-rigid pair."""
    off = np.array([3e-3, -2e-3, 0.5 * (D1[2] + D2[2]) + gap])
    s, sol = _cuboid_pair(off, mat=LinearMaterial(1.3, 1.05, 1.3), meshes=((3, 2, 2), (2, 2, 2)))
    p = np.zeros(3)
    FA, TA = force_torque(s, sol, "A", pivot=p)
    FB, TB = force_torque(s, sol, "B", pivot=p)
    assert np.linalg.norm(FA + FB) / np.linalg.norm(FB) < 1e-6
    assert np.linalg.norm(TA + TB) / np.linalg.norm(TB) < 1e-5


def test_isolated_body_has_no_force():
    s = MagnetSystem([Magnet("A", Cuboid(D1), Pose(), NDFEB, mesh=(2, 2, 2))])
    F, T = force_torque(s, s.solve(), "A")
    assert np.all(F == 0) and np.all(T == 0)


def _interaction_energy(rotB, pB, dimsA, dimsB, n_sub=3, q=8):
    """Rigid magnets: E_int = -integral_B J_B . H_A dV (composite Gauss over B; smooth at this gap)."""
    A = Magnet("A", Cuboid(dimsA), Pose(), RIGID, mesh=(1, 1, 1))
    B = Magnet("B", Cuboid(dimsB), Pose(pB, rotB), RIGID, mesh=(1, 1, 1))
    s = MagnetSystem([A, B])
    sol = s.solve()
    x, w = np.polynomial.legendre.leggauss(q)
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    gw = np.einsum("i,j,k->ijk", w, w, w).ravel() / 8
    c = (np.stack(np.meshgrid(*[np.arange(n_sub)] * 3, indexing="ij"), -1).reshape(-1, 3) + 0.5) / n_sub - 0.5
    loc = ((c[:, None, :] + g[None] / (2 * n_sub)) * dimsB).reshape(-1, 3)
    W = np.tile(gw, c.shape[0]) / c.shape[0] * np.prod(dimsB)
    H = s.field(Pose(pB, rotB).apply(loc), sol.J, bodies=["A"])
    JB = sol.J[s.body_slice("B")][0]
    return -(W * (H @ JB)).sum(), s, sol


def test_f5_torque_vs_virtual_work():
    """T = -dE/dtheta for small rotations of a tilted rigid magnet about its centroid (< 1e-3 rel).

    Also shows that the extra volume term sum V J x H_other listed in docs/SPEC.md would double count:
    with it the torque misses the virtual-work value by far more than the gate.
    """
    dimsA, dimsB = np.array([10e-3, 6e-3, 4e-3]), np.array([8e-3, 5e-3, 3e-3])
    rot0 = Rotation.from_euler("xyz", [12, -7, 20], degrees=True)
    pB = np.array([1.5e-3, 0.8e-3, 7.5e-3])
    _, s, sol = _interaction_energy(rot0, pB, dimsA, dimsB)
    _, T = force_torque(s, sol, "B")
    eps = 1e-4
    Tfd = np.zeros(3)
    for k in range(3):
        v = np.zeros(3)
        v[k] = eps
        Ep, _, _ = _interaction_energy(Rotation.from_rotvec(v) * rot0, pB, dimsA, dimsB)
        Em, _, _ = _interaction_energy(Rotation.from_rotvec(-v) * rot0, pB, dimsA, dimsB)
        Tfd[k] = -(Ep - Em) / (2 * eps)
    assert np.linalg.norm(T - Tfd) / np.linalg.norm(Tfd) < 1e-3
    # the SPEC's extra volume term (evaluated with the same quadrature as the energy) breaks it
    JB = sol.J[s.body_slice("B")][0]
    x, w = np.polynomial.legendre.leggauss(6)
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3) / 2
    gw = np.einsum("i,j,k->ijk", w, w, w).ravel() / 8
    H = s.field(Pose(pB, rot0).apply(g * dimsB), sol.J, bodies=["A"])
    extra = np.cross(JB, (gw[:, None] * H).sum(0)) * np.prod(dimsB)
    assert np.linalg.norm(T + extra - Tfd) / np.linalg.norm(Tfd) > 0.1


@pytest.mark.parametrize("mat", [RIGID, LinearMaterial(1.2, 1.5, 1.2)])
def test_f5_rotated_frame_invariance(mat):
    rot = Rotation.from_euler("xyz", [-30, 50, 110], degrees=True)
    c = np.array([2e-3, -1e-3, 4e-3])
    s0, sol0 = _tilted_pair(mat)
    s1, sol1 = _tilted_pair(mat, rot_all=rot, center=c)
    F0, T0 = force_torque(s0, sol0, "B")
    F1, T1 = force_torque(s1, sol1, "B")
    assert np.linalg.norm(F1 - rot.apply(F0)) / np.linalg.norm(F0) < 1e-6
    assert np.linalg.norm(T1 - rot.apply(T0)) / np.linalg.norm(T0) < 1e-6


def test_error_estimate_reported():
    s, sol = _tilted_pair(NDFEB)
    r = force_torque_detail(s, sol, "B", error_estimate=True)
    assert r.error_F is not None and r.error_F < 1e-6 * np.linalg.norm(r.F)
    assert r.error_T < 1e-6 * np.linalg.norm(r.T)
    assert r.n_nodes > 0 and r.n_pairs["gauss"] > 0


def test_ring_pair_symmetry():
    """Coaxial rings (inner charged surfaces present): axial force only, Newton's third law."""
    mat = LinearMaterial(1.2, 1.1, 1.1)
    A = Magnet("A", Ring(2e-3, 6e-3, 3e-3), Pose(), mat, mesh=(2, 1, 3))
    B = Magnet("B", Ring(3e-3, 5e-3, 2e-3), Pose([0, 0, 2.7e-3]), mat, mesh=(2, 1, 2))
    s = MagnetSystem([A, B])
    sol = s.solve()
    FA, _ = force_torque(s, sol, "A")
    FB, TB = force_torque(s, sol, "B")
    assert np.linalg.norm(FA + FB) / np.linalg.norm(FB) < 1e-6
    assert np.linalg.norm(FB[:2]) < 1e-9 * abs(FB[2]) and FB[2] < 0


# ---------------------------------------------------------------------------------------------
# F6: MoM vs the ported 1-D slab benchmark (linear-recoil Alnico-like rods)
# ---------------------------------------------------------------------------------------------
def _f6(mesh, n_slabs):
    R, L = 4.75e-3 / 2, 12.5e-3
    mat = LinearMaterial(1.2, 4.0)
    rod = an.CoaxialRodPair(R, L, mat, n_slabs=n_slabs)
    Joc_1d = rod.solve(200 * L)[0][:n_slabs].mean()
    F_1d = rod.force(0.1e-3)
    s1 = MagnetSystem([Magnet("A", Cylinder(2 * R, L), Pose(), mat, mesh=mesh)])
    sol1 = s1.solve()
    Joc = (sol1.J[:, 2] * s1.volume).sum() / s1.volume.sum()
    A = Magnet("A", Cylinder(2 * R, L), Pose([0, 0, -L / 2]), mat, mesh=mesh)
    B = Magnet("B", Cylinder(2 * R, L), Pose([0, 0, 0.1e-3 + L / 2]), mat, mesh=mesh)
    s2 = MagnetSystem([A, B])
    F, _ = force_torque(s2, s2.solve(), "B")
    return Joc / Joc_1d - 1, F[2] / F_1d - 1


def test_f6_mom_vs_1d_slab():
    """Open-circuit mean J and pair force at 0.1 mm: MoM vs the 1-D radially-uniform slab model.

    The 1-D model forces J to be radially uniform; with mu_rec = 4 the 3-D J concentrates near the
    rim at the pole faces, so a sub-percent to few-percent difference is expected (measured:
    +0.04 % / -0.5 % on this mesh, see the task report). Gate: < 2 % (docs/SPEC.md F6).
    """
    dJ, dF = _f6({"n": (3, 1, 12), "grading": (1.0, 1.0, 1.3)}, 48)
    assert abs(dJ) < 0.02 and abs(dF) < 0.02


@pytest.mark.slow
def test_f6_mom_vs_1d_slab_fine():
    dJ, dF = _f6({"n": (5, 1, 24), "grading": (1.0, 1.0, 1.2)}, 96)
    assert abs(dJ) < 0.02 and abs(dF) < 0.02
