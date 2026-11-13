"""Tests for magsim.geometry: poses, meshing, Gauss points, charge-face topology, overlap check."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from magsim.forces import _map
from magsim.geometry import (
    ANNULUS,
    CUBOID,
    CYLINDER,
    SEGMENT,
    Cuboid,
    Cylinder,
    Magnet,
    Pose,
    Ring,
    check_overlap,
    mesh_shape,
    perp_axes,
)
from magsim.materials.base import LinearMaterial

MAT = LinearMaterial(1.2, 1.0)


def test_pose_from_dict_and_inverse():
    p = Pose.from_dict({"position": [1e-3, 2e-3, -3e-3], "rotation": {"euler_xyz_deg": [10, 20, 30]}})
    ref = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    assert np.allclose(p.matrix, ref.as_matrix(), atol=1e-15)
    q = Pose.from_dict({"position": [0, 0, 0], "rotation": {"rotvec_deg": [0, 0, 90]}})
    assert np.allclose(q.apply([1.0, 0, 0]), [0, 1.0, 0], atol=1e-15)
    pts = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(p.to_body(p.apply(pts)), pts, atol=1e-14)
    assert Pose.from_dict(None).position.tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        Pose.from_dict({"rotation": {"quat": [0, 0, 0, 1]}})
    # rigid rotation of a pose about a centre
    R = Rotation.from_rotvec([0, 0, np.pi / 2])
    r = Pose([1.0, 0, 0]).rotated(R, center=[1.0, 1.0, 0])
    assert np.allclose(r.position, [2.0, 1.0, 0], atol=1e-15)


def test_perp_axes_orthonormal():
    e = np.random.default_rng(1).normal(size=(20, 3))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    e = np.vstack([e, [0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
    a, b = perp_axes(e)
    for u, v in ((a, b), (a, e), (b, e)):
        assert np.allclose((u * v).sum(1), 0, atol=1e-14)
    assert np.allclose(np.cross(a, b), e, atol=1e-14)


@pytest.mark.parametrize(
    "shape,mesh,kinds",
    [
        (Cuboid((20e-3, 10e-3, 5e-3)), {"n": (5, 3, 2), "grading": (1.5, 1.0, 2.0)}, {CUBOID}),
        (Cylinder(4.75e-3, 12.5e-3), {"n": (3, 1, 6), "grading": (1.3, 1, 1.2)}, {CYLINDER, ANNULUS}),
        (Cylinder(5e-3, 3e-3), (3, 4, 2), {CYLINDER, SEGMENT}),
        (Ring(2e-3, 6e-3, 3e-3), (2, 1, 3), {ANNULUS}),
        (Ring(2e-3, 6e-3, 3e-3), (2, 3, 2), {SEGMENT}),
    ],
)
def test_mesh_volumes_gauss_points(shape, mesh, kinds):
    bm = mesh_shape(shape, mesh)
    assert set(np.unique(bm.kind)) == kinds
    assert abs(bm.volume.sum() / shape.volume - 1) < 1e-13
    # weights sum to 1 per cell, every Gauss point lies inside the body and its own cell's sphere
    wsum = np.bincount(bm.gcell, bm.gw)
    assert np.allclose(wsum, 1.0, atol=1e-14)
    assert np.all(shape.sdf(bm.gp) < 0)
    d = np.linalg.norm(bm.gp - bm.bsphere_c[bm.gcell], axis=1)
    assert np.all(d <= bm.bsphere_r[bm.gcell])
    # Gauss-weighted point average of a cell = its centroid for cuboids and full rings (symmetric rules)
    gm = np.add.reduceat(bm.gp * bm.gw[:, None], np.searchsorted(bm.gcell, np.arange(bm.n_cells)), axis=0)
    full = bm.kind != SEGMENT
    assert np.allclose(gm[full], bm.centroid[full], atol=1e-12 * shape.size)


def test_graded_and_explicit_edges():
    bm = mesh_shape(Cuboid((1.0, 1.0, 1.0)), {"n": (4, 1, 1), "grading": (2.0, 1, 1)})
    xe = bm.grid["edges"][0]
    sizes = np.diff(xe)
    assert np.allclose(sizes / sizes[0], [1, 2, 2, 1])
    bm2 = mesh_shape(Cuboid((1.0, 1.0, 1.0)), {"n": (3, 1, 1), "edges": ([0, 0.1, 0.5, 1.0], None, None)})
    assert np.allclose(bm2.grid["edges"][0], [-0.5, -0.4, 0.0, 0.5])
    with pytest.raises(ValueError):
        mesh_shape(Cuboid((1.0, 1.0, 1.0)), {"n": (3, 1, 1), "edges": ([0, 0.6, 0.5, 1.0], None, None)})


def _face_moments(bm, J):
    """Charge and dipole moment of the net face charges for per-cell J (body frame), by GL quadrature."""
    f = bm.faces
    x, w = np.polynomial.legendre.leggauss(16)
    x, w = 0.5 * (x + 1), 0.5 * w
    X, Y = np.meshgrid(x, x, indexing="ij")
    W = np.outer(w, w).ravel()
    X, Y = X.ravel(), Y.ravel()
    Jm = np.where(f.minus[:, None] >= 0, J[np.maximum(f.minus, 0)], 0.0)
    Jp = np.where(f.plus[:, None] >= 0, J[np.maximum(f.plus, 0)], 0.0)
    dJ = Jm - Jp
    n = len(f)
    idx = np.repeat(np.arange(n), X.size)
    s = f.rng[idx, 0] + (f.rng[idx, 1] - f.rng[idx, 0]) * np.tile(X, n)
    t = f.rng[idx, 2] + (f.rng[idx, 3] - f.rng[idx, 2]) * np.tile(Y, n)
    P, nrm, jac = _map(f.kind[idx], f.o[idx], f.e1[idx], f.e2[idx], f.e3[idx], f.R[idx], s, t)
    wt = np.tile(W, n) * (f.rng[idx, 1] - f.rng[idx, 0]) * (f.rng[idx, 3] - f.rng[idx, 2]) * jac
    sig = (dJ[idx] * nrm).sum(1)
    return (sig * wt).sum(), (sig[:, None] * wt[:, None] * P).sum(0)


@pytest.mark.parametrize(
    "shape,mesh",
    [
        (Cuboid((3.0, 2.0, 1.0)), {"n": (3, 2, 2), "grading": (1.5, 1, 1)}),
        (Cylinder(2.0, 1.5), (3, 1, 3)),
        (Cylinder(2.0, 1.5), (3, 4, 2)),
        (Ring(1.0, 3.0, 1.0), (2, 1, 2)),
        (Ring(1.0, 3.0, 1.0), (2, 3, 2)),
    ],
)
def test_face_topology_charge_and_moment(shape, mesh):
    """Net face charges of any per-cell J: total charge 0 and dipole moment sum_k V_k J_k (exact)."""
    bm = mesh_shape(shape, mesh)
    J = np.random.default_rng(3).normal(size=(bm.n_cells, 3))
    q, m = _face_moments(bm, J)
    ref = (bm.volume[:, None] * J).sum(0)
    assert abs(q) < 1e-12 * np.abs(J).sum() * shape.size**2
    assert np.allclose(m, ref, rtol=0, atol=1e-11 * np.abs(ref).max())
    # uniform J: internal faces carry no charge
    Ju = np.tile([0.3, -0.2, 1.1], (bm.n_cells, 1))
    f = bm.faces
    internal = (f.minus >= 0) & (f.plus >= 0)
    dJ = Ju[np.maximum(f.minus, 0)] - Ju[np.maximum(f.plus, 0)]
    assert np.all(np.abs(dJ[internal]) == 0)


def test_overlap_check():
    a = Magnet("a", Cuboid((1e-3, 1e-3, 1e-3)), Pose(), MAT, mesh=(1, 1, 1))
    touching = Magnet("b", Cuboid((1e-3, 1e-3, 1e-3)), Pose([1e-3, 0.3e-3, 0]), MAT, mesh=(1, 1, 1))
    check_overlap([a, touching])
    over = Magnet("c", Cuboid((1e-3, 1e-3, 1e-3)), Pose([0.99e-3, 0, 0]), MAT, mesh=(1, 1, 1))
    with pytest.raises(ValueError):
        check_overlap([a, over])
    # rotated cube: corner reaches past a face (SAT), separated along a cross-product axis otherwise
    rot = Pose([1.15e-3, 0, 0], Rotation.from_euler("z", 45, degrees=True))
    with pytest.raises(ValueError):
        check_overlap([a, Magnet("d", Cuboid((1e-3, 1e-3, 1e-3)), rot, MAT, mesh=(1, 1, 1))])
    ok = Pose([0.5e-3 + np.sqrt(2) * 0.5e-3 + 1e-9, 0, 0], Rotation.from_euler("z", 45, degrees=True))
    check_overlap([a, Magnet("e", Cuboid((1e-3, 1e-3, 1e-3)), ok, MAT, mesh=(1, 1, 1))])
    # rod through a ring hole is fine (non-convex), rod crossing the ring wall is not
    ring = Magnet("r", Ring(3e-3, 6e-3, 2e-3), Pose(), MAT, mesh=(1, 1, 1))
    rod = Magnet("rod", Cylinder(2.9e-3, 10e-3), Pose(), MAT, mesh=(1, 1, 1))
    check_overlap([ring, rod])
    rod2 = Magnet("rod2", Cylinder(2e-3, 10e-3), Pose([2e-3, 0, 0]), MAT, mesh=(1, 1, 1))
    with pytest.raises(ValueError):
        check_overlap([ring, rod2])
    # crossed cylinders ("+")
    c1 = Magnet("c1", Cylinder(2e-3, 10e-3), Pose(), MAT, mesh=(1, 1, 1))
    c2 = Magnet("c2", Cylinder(2e-3, 10e-3), Pose([0, 0, 1e-3], Rotation.from_euler("x", 90, degrees=True)), MAT,
                mesh=(1, 1, 1))
    with pytest.raises(ValueError):
        check_overlap([c1, c2])
    c3 = Magnet("c3", Cylinder(2e-3, 4e-3), Pose([0, 0, 7e-3]), MAT, mesh=(1, 1, 1))  # end-to-end contact
    check_overlap([c1, c3])


def test_magnet_validation():
    with pytest.raises(ValueError):
        Magnet("x", Cuboid((1, 1, 1)), Pose(), None)
    with pytest.raises(ValueError):
        Magnet("x", Cuboid((1, 1, 1)), Pose(), MAT, easy_axis=(0, 0, 0))
    with pytest.raises(ValueError):
        Cuboid((1, -1, 1))
    with pytest.raises(ValueError):
        Ring(3, 2, 1)
    m = Magnet("x", Cylinder(2.0, 1.0), {"position": [0, 0, 1]}, MAT, easy_axis=(0, 0, 2), mesh={"n": (2, 1, 2), "q": 1})
    assert np.allclose(m.easy_axis, [0, 0, 1]) and m.n_cells == 4
