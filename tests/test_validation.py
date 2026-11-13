"""Validation tools: GCI [celik2008], CSV loader, fits, axisymmetric FEM (F6), magpylib cross-check."""

import math

import numpy as np
import pytest

from magsim import analytic as an
from magsim.materials.base import LinearMaterial
from magsim.validation.convergence import gci, representative_h, scale_mesh
from magsim.validation.experiment import e2_gap_readings, fit_amplitude, load_force_data


def test_gci_reproduces_celik_table1():
    """Celik et al. (2008) Table 1, column phi_1: p = 1.53, phi_ext = 6.1685, e_a = 1.5 %, e_ext = 1.71 %, GCI = 2.2 %."""
    g = gci([6.063, 5.972, 5.863], [1.0, 1.5, 2.0])
    assert g["p"] == pytest.approx(1.53, abs=0.01)
    assert g["phi_ext"] == pytest.approx(6.1685, abs=1e-3)
    assert g["e_ext21"] == pytest.approx(0.0171, abs=2e-4)
    assert g["gci21"] == pytest.approx(0.022, abs=5e-4)


@pytest.mark.parametrize("p", [1.0, 2.0, 0.7])
def test_gci_synthetic_known_order(p):
    h = np.array([1.0, 1.4, 2.1])
    phi = 3.0 + 0.2 * h**p
    g = gci(phi, h)
    assert g["p"] == pytest.approx(p, rel=1e-8)
    assert g["phi_ext"] == pytest.approx(3.0, rel=1e-10)
    assert g["s"] == 1.0


def test_mesh_helpers():
    assert scale_mesh([4, 2, 1], 3) == [12, 6, 3]
    assert scale_mesh([2, 1, 6], 1.5, fixed=(1,)) == [3, 1, 9]
    with pytest.raises(ValueError):
        scale_mesh([2, 1, 6], 1.5)
    assert representative_h(8.0, 8) == pytest.approx(1.0)
    assert representative_h(8.0, 2, dim=2) / representative_h(8.0, 8, dim=2) == pytest.approx(2.0)


def test_csv_loader():
    d = load_force_data()
    assert d.sigma_N == 0.25
    assert len(d.ndfeb) == 27 and len(d.alnico) == 27
    assert d.ndfeb.gap_mm.min() == 2 and d.ndfeb.gap_mm.max() == 10
    assert d.alnico.F_N.iloc[0] == 2.0 and d.alnico.gap_mm.max() == pytest.approx(0.4)
    assert any("3.63" in n for n in d.notes)  # mistyped stored average at 7 mm is flagged
    g = e2_gap_readings(np.array([0.0, 0.05, 0.4]))
    assert np.allclose(g["csv_mm"], [0, 5e-5, 4e-4]) and np.allclose(g["sheets"], [0, 1e-4, 8e-4])


def test_fit_amplitude_recovers_quadratic_model():
    rng = np.random.default_rng(1)
    gaps = np.linspace(1, 5, 9)
    model = lambda th: th**2 / gaps  # noqa: E731
    th_true = 0.9
    idx = np.repeat(np.arange(9), 3)
    y = model(th_true)[idx] + rng.normal(0, 0.01, idx.size)
    grid = np.array([0.85, 0.9, 0.95])
    fit = fit_amplitude(grid, np.array([model(t) for t in grid]), idx, y, 0.01, n_boot=300)
    assert fit["ci_jac"][0] < th_true < fit["ci_jac"][1]
    assert fit["ci_boot"][0] < th_true < fit["ci_boot"][1]
    assert fit["dof"] == 26 and 0.4 < fit["chi2_dof"] < 2.0


# --- F6 / F3 with the axisymmetric FEM ----------------------------------------------------------------
R, L = 4.75e-3 / 2, 12.5e-3


def test_fem_rigid_vs_closed_forms():
    """Rigid rods: FEM P2 pair force vs Lipschitz-Hankel [vokoun2009] and demag factor vs [chen1991].

    Tolerances from the measured h-convergence (h = 0.3 mm: 4.9e-4 force, 2.9e-4 demag).
    """
    from magsim.validation.axisym_fem import fem_rod_case

    r = fem_rod_case(LinearMaterial(1.2), R, L, 0.1e-3, 0.3e-3, slabs=(1, 1))
    assert r["F"] / an.coaxial_cylinder_force(1.2, 1.2, R, L, L, 0.1e-3) - 1 == pytest.approx(0, abs=1e-3)
    assert r["N_oc"] / an.cylinder_demag_factor(R, L) - 1 == pytest.approx(0, abs=1e-3)


def test_fem_vs_mom_mu_perp_one():
    """Gate F6 where both methods represent the same model (mu_perp = 1): linear mu_rec = 4 open circuit.

    MoM (8,1,24) 0.86127 vs FEM 0.86184 measured; full-ring MoM cannot carry radial J, so mu_perp > 1 is only
    compared in the report. Gate: < 2 % (SPEC F6), asserted at 0.5 %.
    """
    from magsim.geometry import Cylinder, Magnet, Pose
    from magsim.mom import MagnetSystem
    from magsim.validation.axisym_fem import fem_rod_case

    mat = LinearMaterial(1.2, 4.0, 1.0)
    s = MagnetSystem([Magnet("A", Cylinder(2 * R, L), Pose(), mat, mesh=(4, 1, 12))])
    sol = s.solve()
    J_mom = (sol.J[:, 2] * s.volume).sum() / s.volume.sum()
    J_fem = fem_rod_case(mat, R, L, None, 0.3e-3, slabs=(1, 1))["J_oc"]
    assert abs(J_mom / J_fem - 1) < 5e-3


@pytest.mark.slow
def test_fem_vs_mom_lng37_mu_perp_one():
    """Nonlinear LNG37 (mu_perp = 1), matched (8, 24) cells / slabs: open-circuit J within 1 % (measured 0.3 %)."""
    from magsim.geometry import Cylinder, Magnet, Pose
    from magsim.materials.library import get_material
    from magsim.mom import MagnetSystem
    from magsim.validation.axisym_fem import fem_rod_case

    mat = get_material("LNG37", mu_perp=1.0)
    s = MagnetSystem([Magnet("A", Cylinder(2 * R, L), Pose(), mat, mesh=(8, 1, 24))])
    sol = s.solve()
    J_mom = (sol.J[:, 2] * s.volume).sum() / s.volume.sum()
    J_fem = fem_rod_case(mat, R, L, None, 0.2e-3, slabs=(8, 24))["J_oc"]
    assert abs(J_mom / J_fem - 1) < 1e-2


@pytest.mark.slow
def test_magpylib_getft_crosscheck():
    """Rigid tilted pairs vs magpylib.getFT [ortner2020]: difference shrinks with the magpylib mesh."""
    from magsim.validation import magpylib_crosscheck

    df = magpylib_crosscheck((1, 2, 4, 8))
    for _, g in df.groupby("case"):
        assert np.all(np.diff(g["relF"].to_numpy()) < 0)
        assert g["relF"].iloc[-1] < 1e-4 and g["relF Richardson"].iloc[-1] < 2e-6
        assert g["relT"].iloc[-1] < 2e-4
