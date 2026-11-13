"""Replay of the measured force data (E1 NdFeB blocks, E2 LNG37 rods) and single-parameter fits (gate F7).

Data: ``res_exp/force/mag_force_data.csv`` (byte copy of ``Mag Force Data.csv``): two side-by-side
column groups, "NIB" (E1: gap ``space (mm)`` 2..10 mm) and "Alnico" (E2: 0..0.4 mm), each with three
repeated readings ``force exp 1..3 (N)`` and their average; below them ``err (N)`` = 0.25 N, the stated
reading uncertainty. Readings are quantised (0.25 N steps for E1, 0.05 N for E2), so the quantisation
standard deviation alone is 0.25/sqrt(12) = 0.072 N for E1.
# ASSUMPTION: the stated +-0.25 N is used as the 1-sigma uncertainty of every individual reading (the CSV
# does not say whether it is a standard or an expanded uncertainty); chi^2/dof is reported with it.

Fits (one amplitude parameter theta, e.g. Br [T]): least squares over all N individual readings
(dof = N - 1). The force is evaluated by the full MoM + Preisach model on a grid theta_0 (1 +- 3 %),
interpolated per gap by the exact quadratic through the three points (F ~ theta^2 for a linear magnet);
the surrogate error at the optimum is checked by a direct model run and reported. Confidence intervals:
(a) linearised (Jacobian) with the residual variance s^2 = RSS/dof and Student t_{0.975, dof};
(b) residual bootstrap (resample residuals, refit, 2.5 / 97.5 % percentiles) (Efron & Tibshirani 1993, ch. 9).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from magsim.config import get_path, set_path

__all__ = ["DATA", "ForceData", "load_force_data", "fit_amplitude", "e1_fit", "e2_replay", "e2_gap_readings",
           "SHEET_THICKNESS"]

DATA = Path(__file__).resolve().parents[3] / "res_exp" / "force" / "mag_force_data.csv"
SHEET_THICKNESS = 0.10e-3  # m, spacer sheet of the old validation (SPEC open question)


@dataclass
class ForceData:
    """Long-format readings per experiment: columns gap_mm (CSV value), rep (1..3), F_N; sigma_N [N]."""

    ndfeb: pd.DataFrame
    alnico: pd.DataFrame
    sigma_N: float
    notes: list = field(default_factory=list)  # data-consistency remarks (e.g. a mistyped stored average)

    def group(self, which: str) -> pd.DataFrame:
        return {"ndfeb": self.ndfeb, "nib": self.ndfeb, "e1": self.ndfeb,
                "alnico": self.alnico, "e2": self.alnico}[which.lower()]


def load_force_data(path: str | Path = DATA) -> ForceData:
    """Parse the CSV layout described in the module docstring. Stored averages that disagree with the
    readings (beyond the 0.01 N rounding) are listed in ``notes``."""
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    hdr = next(i for i in range(len(raw)) if any("space (mm)" in c for c in raw.iloc[i]))
    labels = raw.iloc[hdr - 1].tolist()
    starts = [j for j, c in enumerate(raw.iloc[hdr]) if "space (mm)" in c]
    groups = {}
    notes: list[str] = []
    for j in starts:
        label = labels[j].strip().lower() or f"col{j}"
        cols = [c.strip() for c in raw.iloc[hdr, j:j + 5]]
        if not (cols[1].startswith("force exp 1") and cols[4].startswith("force exp avg")):
            raise ValueError(f"{path}: unexpected header {cols}")
        rows = []
        for i in range(hdr + 1, len(raw)):
            gap = raw.iloc[i, j].strip()
            if gap == "":
                break
            vals = [float(raw.iloc[i, j + k]) for k in (1, 2, 3)]
            avg = float(raw.iloc[i, j + 4])
            if abs(np.mean(vals) - avg) > 0.006:  # stored averages are rounded to 0.01 N
                notes.append(f"{label} gap {gap}: stored average {avg} != mean of the readings "
                             f"{np.mean(vals):.4f} (the individual readings are used)")
            rows += [{"gap_mm": float(gap), "rep": k + 1, "F_N": v} for k, v in enumerate(vals)]
        groups[label] = pd.DataFrame(rows)
    err_row = next(i for i in range(len(raw)) if raw.iloc[i, 0].strip().lower().startswith("err"))
    sigma = float(raw.iloc[err_row + 1, 0])
    if "nib" not in groups or "alnico" not in groups:
        raise ValueError(f"{path}: expected NIB and Alnico groups, got {sorted(groups)}")
    return ForceData(groups["nib"], groups["alnico"], sigma, notes)


# ---------------------------------------------------------------------------------------------
# generic single-amplitude least squares on a quadratic surrogate
# ---------------------------------------------------------------------------------------------
def _quad_coeffs(theta: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-column quadratic through (theta_k, F[k, :]) -> (3, m) polynomial coefficients (highest first)."""
    V = np.vander(theta, 3)
    return np.linalg.solve(V, F)


def _eval(c, th):
    return c[0] * th**2 + c[1] * th + c[2]


def _deriv(c, th):
    return 2 * c[0] * th + c[1]


def fit_amplitude(theta_grid, F_grid, idx, y, sigma, n_boot=2000, seed=0, bounds=None) -> dict:
    """Least-squares fit of one amplitude theta to readings y (measured at model column idx[j]).

    theta_grid (3,), F_grid (3, m) model values (same sign convention as y). Returns the estimate,
    linearised and bootstrap 95 % CIs, RMS, max |residual|, chi^2, dof, chi^2/dof (with sigma), residuals.
    """
    th = np.asarray(theta_grid, dtype=float)
    c = _quad_coeffs(th, np.asarray(F_grid, dtype=float))
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(y, dtype=float)
    lo, hi = bounds if bounds is not None else (0.5 * th.min(), 1.5 * th.max())

    def solve(yy):
        r = optimize.minimize_scalar(lambda t: float(np.sum((yy - _eval(c[:, idx], t)) ** 2)),
                                     bounds=(lo, hi), method="bounded", options={"xatol": 1e-12 * hi})
        return float(r.x)

    t_hat = solve(y)
    f = _eval(c[:, idx], t_hat)
    res = y - f
    n = y.size
    dof = n - 1
    rss = float(np.sum(res**2))
    Jv = _deriv(c[:, idx], t_hat)
    JtJ = float(Jv @ Jv)
    s2 = rss / dof
    tq = float(stats.t.ppf(0.975, dof))
    se = math.sqrt(s2 / JtJ)
    se_sigma = sigma / math.sqrt(JtJ)
    rng = np.random.default_rng(seed)
    boot = np.array([solve(f + rng.choice(res, size=n, replace=True)) for _ in range(n_boot)])
    return {
        "theta": t_hat, "se": se, "ci_jac": (t_hat - tq * se, t_hat + tq * se),
        "se_sigma": se_sigma, "ci_jac_sigma": (t_hat - 1.96 * se_sigma, t_hat + 1.96 * se_sigma),
        "ci_boot": tuple(float(v) for v in np.percentile(boot, [2.5, 97.5])), "boot_sd": float(boot.std(ddof=1)),
        "rms": math.sqrt(rss / n), "max_abs_res": float(np.max(np.abs(res))), "chi2": rss / sigma**2,
        "dof": dof, "chi2_dof": rss / sigma**2 / dof, "residuals": res, "fitted": f, "n": n,
        "surrogate": c, "t_quantile": tq,
    }


# ---------------------------------------------------------------------------------------------
# E1: NdFeB blocks, fit of Br
# ---------------------------------------------------------------------------------------------
def _e1_point(args):
    from magsim.force_sim import run_scenario

    cfg, fit_path, theta, gap_path, gap = args
    c = set_path(set_path(cfg, fit_path, float(theta)), gap_path, float(gap))
    out = run_scenario(c)
    return out["Fz"], out["converged"]


def model_grid(cfg, fit_path, thetas, gap_path, gaps_m, workers=4, sign=-1.0) -> np.ndarray:
    """sign * Fz for every (theta, gap): array (len(thetas), len(gaps))."""
    jobs = [(cfg, fit_path, t, gap_path, g) for t in thetas for g in gaps_m]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            res = list(pool.map(_e1_point, jobs))
    else:
        res = [_e1_point(j) for j in jobs]
    if not all(r[1] for r in res):
        raise RuntimeError("E1 model: a MoM solve did not converge")
    return sign * np.array([r[0] for r in res]).reshape(len(thetas), len(gaps_m))


def e1_fit(cfg: dict, data: ForceData, fit_path: str, gap_path: str = "layout.gap", theta0: float | None = None,
           workers: int = 4, n_boot: int = 2000, rel_step: float = 0.03) -> dict:
    """E1: fit one amplitude (``fit_path``, e.g. ``materials.block.Br``) to the NdFeB readings.

    Measured forces are attraction magnitudes; the model's Fz on the upper block is negative for attraction,
    so the fitted quantity is -Fz. Returns the ``fit_amplitude`` dict plus the per-gap table and checks.
    """
    d = data.ndfeb
    gaps_mm = np.unique(d["gap_mm"].to_numpy())
    gaps = gaps_mm * 1e-3
    idx = np.searchsorted(gaps_mm, d["gap_mm"].to_numpy())
    y = d["F_N"].to_numpy()
    th0 = float(theta0 if theta0 is not None else get_path(cfg, fit_path))
    F0 = model_grid(cfg, fit_path, [th0], gap_path, gaps, workers)[0]
    th1 = th0 * math.sqrt(float(np.sum(y * F0[idx]) / np.sum(F0[idx] ** 2)))  # F ~ theta^2 first guess
    grid = th1 * np.array([1 - rel_step, 1.0, 1 + rel_step])
    Fg = model_grid(cfg, fit_path, grid, gap_path, gaps, workers)
    fit = fit_amplitude(grid, Fg, idx, y, data.sigma_N, n_boot=n_boot)
    Fd = model_grid(cfg, fit_path, [fit["theta"]], gap_path, gaps, workers)[0]
    fit["surrogate_max_rel_err"] = float(np.max(np.abs(_eval(fit["surrogate"], fit["theta"]) / Fd - 1)))
    table = pd.DataFrame({"gap_mm": gaps_mm, "F_meas_mean": d.groupby("gap_mm")["F_N"].mean().to_numpy(),
                          "F_model": Fd, "F_model_theta0": F0})
    table["residual_mean"] = table["F_meas_mean"] - table["F_model"]
    fit.update(table=table, theta0=th0, fit_path=fit_path, grid=grid, readings=d.assign(F_fit=fit["fitted"],
               residual=fit["residuals"]))
    return fit


# ---------------------------------------------------------------------------------------------
# E2: LNG37 rods, protocol replay
# ---------------------------------------------------------------------------------------------
def e2_gap_readings(gap_mm: np.ndarray) -> dict[str, np.ndarray]:
    """The two readings of the E2 'space' column (SPEC open question) -> model gaps [m].

    ``csv_mm``: the value is the gap in mm. ``sheets``: the value / 0.05 is a count of 0.10 mm spacer sheets,
    i.e. the gap is twice the value (the old validation).
    """
    g = np.asarray(gap_mm, dtype=float)
    return {"csv_mm": g * 1e-3, "sheets": np.round(g / 0.05) * SHEET_THICKNESS}


def e2_replay(cfg: dict, gaps_m, n_rep: int = 3, body: str | None = None) -> pd.DataFrame:
    """Replay the E2 handling history on the scenario's magnets (history-dependent).

    Start: the scenario's initial state (+sat) -> ``isolate`` (open circuit) -> for each gap in the given
    (measured) order, n_rep x (``place`` at the gap = move + solve + commit, ``measure``, ``separate``).
    Returns one row per measurement: gap_m, rep, F (= -Fz, attraction positive), J_mean_<bodies>.
    """
    from magsim.force_sim import ForceScenario

    fs = ForceScenario(cfg)
    body = body or str(cfg["layout"]["body"])
    fs.isolate()
    rows = []
    for g in gaps_m:
        for k in range(n_rep):
            fs.move({"name": body, "gap": float(g)})
            out = fs.measure()
            row = {"gap_m": float(g), "rep": k + 1, "F": -out["Fz"], "converged": out["converged"]}
            row.update({kk: v for kk, v in out.items() if kk.startswith("J_mean_")})
            rows.append(row)
            fs.separate()
    df = pd.DataFrame(rows)
    df["all_converged"] = all(r["converged"] for r in fs.records)
    return df


def _e2_job(args):
    cfg, gaps, n_rep = args
    return e2_replay(cfg, gaps, n_rep)
