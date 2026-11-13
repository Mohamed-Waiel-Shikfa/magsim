"""Material protocol (interface contract, docs/SPEC.md) and the linear reference material.

Conventions: polarisation J = MU0*M [T]; the law takes the field H [A/m] inside the material,
split into H_par = H.e (along the easy axis e) and H_perp (the two components orthogonal to e).
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from magsim.constants import MU0

State = Any  # opaque per-material object holding per-cell numpy arrays


class Material(Protocol):
    """Per-cell magnetic material law.

    All arrays are per cell: n cells, H_par (n,), H_perp (n, 2) in A/m, T scalar or (n,) in K.
    """

    name: str

    def init_state(self, n: int, initial: str | float) -> State:
        """Initial state for n cells.

        initial: "+sat" | "-sat" | "thermal" (demagnetised) | float J_irr/Js_irr in [-1, 1].
        """
        ...

    def response(
        self, H_par: np.ndarray, H_perp: np.ndarray, state: State, T: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Trial evaluation; does NOT modify state.

        Returns (J_par [T] (n,), dJ_dH_par [T m/A] (n,), J_perp [T] (n, 2), chi_perp [-] (n,)),
        with J_perp = MU0 * chi_perp * H_perp.
        """
        ...

    def commit(self, H_par: np.ndarray, state: State, T: float | np.ndarray) -> State:
        """Advance history along the monotone segment from the last committed H_par to H_par."""
        ...


@dataclass(frozen=True)
class LinearState:
    """State of LinearMaterial: per-cell remanence sign s in [-1, 1], shape (n,)."""

    s: np.ndarray


def _check_cells(H_par, H_perp, state: LinearState, T) -> tuple[np.ndarray, np.ndarray]:
    H_par = np.asarray(H_par, dtype=float)
    H_perp = np.asarray(H_perp, dtype=float)
    n = state.s.shape[0]
    if H_par.shape != (n,):
        raise ValueError(f"H_par must have shape ({n},), got {H_par.shape}")
    if H_perp.shape != (n, 2):
        raise ValueError(f"H_perp must have shape ({n}, 2), got {H_perp.shape}")
    if not (np.all(np.isfinite(H_par)) and np.all(np.isfinite(H_perp))):
        raise ValueError("H_par and H_perp must be finite")
    T = np.asarray(T, dtype=float)
    if T.shape not in ((), (n,)):
        raise ValueError(f"T must be a scalar or shape ({n},), got {T.shape}")
    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("T must be finite and positive [K]")
    return H_par, H_perp


class LinearMaterial:
    """History-free linear magnet: J_par = Jr*s + MU0*(mu_rec-1)*H_par, J_perp = MU0*(mu_perp-1)*H_perp.

    Jr: remanent polarisation [T]; mu_rec, mu_perp: relative recoil permeabilities along / across
    the easy axis (>= 1). mu_rec = mu_perp = 1 gives a rigid magnet. Temperature is ignored.
    """

    def __init__(self, Jr: float, mu_rec: float = 1.0, mu_perp: float = 1.0, name: str = "linear"):
        if not np.isfinite(Jr):
            raise ValueError(f"Jr must be finite, got {Jr}")
        for label, mu in (("mu_rec", mu_rec), ("mu_perp", mu_perp)):
            if not np.isfinite(mu) or mu < 1.0:
                raise ValueError(f"{label} must be finite and >= 1, got {mu}")
        self.Jr = float(Jr)
        self.mu_rec = float(mu_rec)
        self.mu_perp = float(mu_perp)
        self.name = name

    def init_state(self, n: int, initial: str | float) -> LinearState:
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError(f"n must be a non-negative integer, got {n!r}")
        if isinstance(initial, str):
            table = {"+sat": 1.0, "-sat": -1.0, "thermal": 0.0}
            if initial not in table:
                raise ValueError(f"initial must be one of {list(table)} or a float in [-1, 1], got {initial!r}")
            s = table[initial]
        else:
            s = float(initial)
            if not -1.0 <= s <= 1.0:
                raise ValueError(f"initial must lie in [-1, 1], got {initial}")
        return LinearState(s=np.full(int(n), s))

    def response(self, H_par, H_perp, state: LinearState, T):
        H_par, H_perp = _check_cells(H_par, H_perp, state, T)
        n = H_par.shape[0]
        J_par = self.Jr * state.s + MU0 * (self.mu_rec - 1.0) * H_par
        dJ_dH_par = np.full(n, MU0 * (self.mu_rec - 1.0))
        J_perp = MU0 * (self.mu_perp - 1.0) * H_perp
        chi_perp = np.full(n, self.mu_perp - 1.0)
        return J_par, dJ_dH_par, J_perp, chi_perp

    def commit(self, H_par, state: LinearState, T) -> LinearState:
        _check_cells(H_par, np.zeros((state.s.shape[0], 2)), state, T)
        return state
