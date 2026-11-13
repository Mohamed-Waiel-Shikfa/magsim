import numpy as np
import pytest

from magsim.constants import MU0
from magsim.materials.base import LinearMaterial, Material


def test_satisfies_protocol():
    m: Material = LinearMaterial(1.2)
    assert m.name == "linear"


@pytest.mark.parametrize("initial, s", [("+sat", 1.0), ("-sat", -1.0), ("thermal", 0.0), (0.25, 0.25), (-1, -1.0)])
def test_init_state(initial, s):
    st = LinearMaterial(1.2).init_state(4, initial)
    np.testing.assert_array_equal(st.s, np.full(4, s))


@pytest.mark.parametrize("initial", ["sat", 1.5, -1.01])
def test_init_state_rejects(initial):
    with pytest.raises(ValueError):
        LinearMaterial(1.2).init_state(3, initial)


def test_response_values():
    Jr, mu_rec, mu_perp = 1.3, 1.05, 1.1
    m = LinearMaterial(Jr, mu_rec, mu_perp)
    st = m.init_state(3, "-sat")
    H_par = np.array([-1e5, 0.0, 2e5])
    H_perp = np.array([[1e4, -2e4], [0.0, 0.0], [3e3, 5e3]])
    J_par, dJ, J_perp, chi = m.response(H_par, H_perp, st, 293.15)
    np.testing.assert_allclose(J_par, -Jr + MU0 * 0.05 * H_par, rtol=1e-14)
    np.testing.assert_allclose(dJ, np.full(3, MU0 * 0.05), rtol=1e-14)
    np.testing.assert_allclose(J_perp, MU0 * 0.1 * H_perp, rtol=1e-14)
    np.testing.assert_allclose(chi, np.full(3, 0.1), rtol=1e-14)
    np.testing.assert_allclose(J_perp, MU0 * chi[:, None] * H_perp, rtol=1e-14)
    assert J_par.shape == dJ.shape == chi.shape == (3,) and J_perp.shape == (3, 2)


def test_rigid_magnet_ignores_field_and_temperature():
    m = LinearMaterial(1.4)
    st = m.init_state(2, "+sat")
    J_par, dJ, J_perp, chi = m.response(np.array([-5e5, 5e5]), np.ones((2, 2)) * 1e5, st, np.array([250.0, 400.0]))
    np.testing.assert_array_equal(J_par, [1.4, 1.4])
    np.testing.assert_array_equal(dJ, 0.0)
    np.testing.assert_array_equal(J_perp, 0.0)


def test_commit_is_history_free():
    m = LinearMaterial(1.2, 1.1)
    st = m.init_state(2, 0.5)
    st2 = m.commit(np.array([-1e6, 1e6]), st, 300.0)
    np.testing.assert_array_equal(st2.s, st.s)


def test_response_does_not_modify_state():
    m = LinearMaterial(1.2, 1.1)
    st = m.init_state(2, 0.5)
    before = st.s.copy()
    m.response(np.array([1e5, -1e5]), np.zeros((2, 2)), st, 293.15)
    np.testing.assert_array_equal(st.s, before)


def test_validation():
    with pytest.raises(ValueError):
        LinearMaterial(1.2, mu_rec=0.9)
    with pytest.raises(ValueError):
        LinearMaterial(1.2, mu_perp=np.nan)
    m = LinearMaterial(1.2)
    st = m.init_state(3, "+sat")
    with pytest.raises(ValueError, match="H_par"):
        m.response(np.zeros(2), np.zeros((3, 2)), st, 300.0)
    with pytest.raises(ValueError, match="H_perp"):
        m.response(np.zeros(3), np.zeros((3, 3)), st, 300.0)
    with pytest.raises(ValueError, match="finite"):
        m.response(np.array([0.0, np.inf, 0.0]), np.zeros((3, 2)), st, 300.0)
    with pytest.raises(ValueError, match="T must"):
        m.response(np.zeros(3), np.zeros((3, 2)), st, np.ones(2))
