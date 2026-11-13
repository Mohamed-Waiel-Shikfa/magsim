"""Validation suite of the force simulator (task 5): experiment replay (gate F7), mesh convergence / GCI
[celik2008], axisymmetric FEM cross-check (gates F3, F6) and an independent magpylib force check.

``run_validation(cfg, out_dir)`` runs the checks listed in the scenario's ``validation.checks`` (each a name
or a one-key mapping ``{name: {params}}``; a ``scenario: <path>`` param runs the check on another scenario
file) and writes ``<out_dir>/validation.md`` plus CSV / figures. Without a ``validation`` block it runs a
mesh-convergence study of the scenario itself (meshes x 1, 1.5, 2, rounded).

Checks: ``e1_fit``, ``e2_replay``, ``convergence``, ``f6``, ``magpylib`` (parameters: see each ``check_*``).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from magsim.config import get_path, load_scenario, set_path
from magsim.constants import MU0

__all__ = ["run_validation", "check_e1", "check_e2", "check_convergence", "check_f6", "check_magpylib",
           "magpylib_crosscheck", "md_table"]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:.4g}") -> str:
    """GitHub markdown table of a DataFrame (no tabulate dependency)."""
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else fmt.get(c, default).format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _quad(c, x):
    return c[0] * x * x + c[1] * x + c[2]


def _style():
    import matplotlib.pyplot as plt

    from magsim import plotting

    plotting.use_style()
    return plt, plotting


def _strip(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in ("sweep", "sweeps", "validation", "outputs")}


# ---------------------------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------------------------
def check_e1(cfg, params, out_dir: Path, workers=4):
    """E1 Br fit. params: fit_path (e.g. materials.block.Br), theta0 [T], n_boot, data (CSV path),
    mesh_study: {base_n, factors} (GCI of Fz at the smallest gap and the fitted Br; used to correct the fitted
    value to the mesh limit and to add a discretisation uncertainty)."""
    from magsim.force_sim import build_material
    from magsim.validation.experiment import DATA, e1_fit, load_force_data, model_grid

    data = load_force_data(params.get("data", DATA))
    fit_path = params.get("fit_path", "materials.block.Br")
    base = _strip(cfg)
    t0 = time.time()
    fit = e1_fit(base, data, fit_path, theta0=params.get("theta0"), workers=workers,
                 n_boot=int(params.get("n_boot", 2000)))
    parent = fit_path.rsplit(".", 1)[0]
    mat_cfg = {k: v for k, v in get_path(base, parent).items() if k not in ("Br", "Br_scale")}
    Br_lib = build_material(mat_cfg).Br
    gaps = fit["table"]["gap_mm"].to_numpy() * 1e-3
    F_lib = model_grid(base, fit_path, [Br_lib], "layout.gap", gaps, workers)[0]
    fit["table"]["F_model_library"] = F_lib
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = params.get("mesh_study")
    disc_md, corr = [], None
    if ms:
        from magsim.validation.convergence import gci_table, mesh_study, scale_mesh

        c_fit = set_path(set_path(base, fit_path, fit["theta"]), "layout.gap", float(gaps.min()))
        mdf = mesh_study(c_fit, ms["base_n"], ms["factors"], workers=min(3, workers))
        gt = gci_table(mdf, ["Fz"]).iloc[0]
        mesh_now = [list((m.get("mesh") or {}).get("n", m.get("mesh"))) for m in base["magnets"]]
        hit = [k for k in ms["factors"] if [scale_mesh(n, k, (1,) if len(n) == 3 and n[1] == 1 and
                                                       base["magnets"][i]["shape"]["type"] != "cuboid" else ())
                                            for i, n in enumerate(ms["base_n"])] == mesh_now]
        if hit:
            phi = float(mdf.loc[mdf.k == hit[0], "Fz"].iloc[0])
            delta = phi / gt["phi_ext"] - 1.0  # relative mesh error of F at the fit mesh
            c = fit["surrogate"][:, 0]
            n_exp = float(fit["theta"] * (2 * c[0] * fit["theta"] + c[1]) / _quad(c, fit["theta"]))  # dlnF/dlnBr
            th_c = fit["theta"] * (1.0 + delta) ** (1.0 / n_exp)
            U = 1.25 * abs(delta) / n_exp * fit["theta"]  # GCI-style (Fs = 1.25) band on Br
            half = 0.5 * (fit["ci_jac"][1] - fit["ci_jac"][0])
            tot = math.hypot(half, U)
            corr = {"theta_c": th_c, "U": U, "ci": (th_c - tot, th_c + tot), "delta": delta, "n_exp": n_exp, "gci": gt}
            mdf.to_csv(out_dir / "e1_mesh_study.csv", index=False)
            disc_md = [
                "",
                f"Discretisation (Fz at {gaps.min() * 1e3:g} mm, fitted $B_r$; meshes {ms['base_n'][0]} x {ms['factors']}): "
                f"observed p = {gt['p']:.2f}, extrapolated Fz = {gt['phi_ext']:.4f} N, GCI$_{{21}}$ = {gt['gci21']:.2e}; "
                f"the fit mesh {mesh_now[0]} overestimates |F| by {delta:+.2%}. With dlnF/dlnBr = {n_exp:.3f}, the "
                f"mesh-limit value is **$B_r$ = {th_c:.4f} T**, discretisation band +-{U:.4f} T (Fs = 1.25), combined "
                f"95 % interval [{corr['ci'][0]:.4f}, {corr['ci'][1]:.4f}] T (Jacobian and discretisation in quadrature).",
            ]
    fit["readings"].to_csv(out_dir / "e1_readings.csv", index=False)
    fit["table"].to_csv(out_dir / "e1_table.csv", index=False)

    plt, plotting = _style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(plotting.IEEE_COL, 3.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    r = fit["readings"]
    tb = fit["table"]
    a1.errorbar(r["gap_mm"], r["F_N"], yerr=data.sigma_N, fmt="o", ms=2.5, color="0.25", elinewidth=0.6, capsize=1.5,
                label="measured (3 readings, $\\pm$0.25 N)")
    a1.plot(tb["gap_mm"], tb["F_model"], "-", color=plotting.OKABE_ITO[5],
            label=f"MoM, fitted $B_r$ = {fit['theta']:.3f} T")
    a1.plot(tb["gap_mm"], tb["F_model_library"], "--", color=plotting.OKABE_ITO[6],
            label=f"MoM, library N42 $B_r$ = {Br_lib:.2f} T")
    a1.set_ylabel("attraction $-F_z$ (N)")
    a1.legend(loc="upper right")
    a2.axhspan(-data.sigma_N, data.sigma_N, color="0.9", lw=0)
    a2.plot(r["gap_mm"], r["residual"], "o", ms=2.5, color=plotting.OKABE_ITO[5])
    a2.axhline(0, color="0.5", lw=0.5)
    a2.set_xlabel("gap (mm)")
    a2.set_ylabel("residual (N)")
    plotting.save(fig, out_dir / "e1_fit")
    plt.close(fig)

    lo, hi = fit["ci_jac"]
    blo, bhi = fit["ci_boot"]
    slo, shi = fit["ci_jac_sigma"]
    sig_q = 0.25 / math.sqrt(12)
    md = [
        "## E1: NdFeB block pair, fit of $B_r$ (gate F7)",
        "",
        f"Model: scenario `{cfg.get('name')}` (MoM + Preisach N42 with $B_r$ free; everything else library N42), "
        f"protocol isolate -> place -> measure, mesh {cfg['magnets'][0].get('mesh')} per block. "
        f"Fitted parameter: **{fit_path} only** (one amplitude). Data: {fit['n']} individual readings, 9 gaps x 3, "
        f"sigma = {data.sigma_N} N per reading (stated; quantisation alone would be {sig_q:.3f} N).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| fitted $B_r$ | **{fit['theta']:.4f} T** |",
        f"| 95 % CI, Jacobian, $s^2$ = RSS/dof, t(0.975, {fit['dof']}) = {fit['t_quantile']:.3f} | [{lo:.4f}, {hi:.4f}] T (se {fit['se']:.2e}) |",
        f"| 95 % CI, Jacobian with stated sigma | [{slo:.4f}, {shi:.4f}] T |",
        f"| 95 % CI, residual bootstrap ({int(params.get('n_boot', 2000))} resamples) | [{blo:.4f}, {bhi:.4f}] T |",
        f"| RMS residual | {fit['rms']:.3f} N |",
        f"| max abs residual | {fit['max_abs_res']:.3f} N |",
        f"| chi^2 / dof (sigma = {data.sigma_N} N, dof = {fit['dof']} = {fit['n']} - 1) | {fit['chi2_dof']:.3f} |",
        f"| chi^2 / dof with quantisation sigma {sig_q:.3f} N | {fit['chi2'] * (data.sigma_N / sig_q) ** 2 / fit['dof']:.2f} |",
        f"| library N42 $B_r$ (unfitted prediction) | {Br_lib:.3f} T |",
        f"| surrogate (quadratic in $B_r$) max rel. error at the optimum | {fit['surrogate_max_rel_err']:.1e} |",
        *disc_md,
        "",
        md_table(tb[["gap_mm", "F_meas_mean", "F_model", "residual_mean", "F_model_library"]],
                 {"gap_mm": "{:.0f}", "F_meas_mean": "{:.3f}", "F_model": "{:.3f}", "residual_mean": "{:+.3f}",
                  "F_model_library": "{:.2f}"}),
        "",
        *(f"Data note: {n}." for n in data.notes if n.startswith("nib")),
        "",
        f"Figure: `e1_fit.pdf`. Runtime {time.time() - t0:.0f} s.",
        "",
    ]
    ok = fit["surrogate_max_rel_err"] < 1e-3
    return "\n".join(md), ok


# ---------------------------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------------------------
def _e2_run(args):
    from magsim.validation.experiment import e2_replay

    cfg, gaps, n_rep = args
    return e2_replay(cfg, gaps, n_rep)


def _e2_fem(args):
    from magsim.force_sim import build_material
    from magsim.validation.axisym_fem import fem_replay

    mat_cfg, R, L, gaps, n_rep, h = args
    return fem_replay(build_material(mat_cfg), R, L, gaps, n_rep=n_rep, h=h)


def _e2_cfg(base, variant, scale, mat_key):
    return set_path(base, f"materials.{mat_key}", {"grade": variant, "Br_scale": float(scale)})


def check_e2(cfg, params, out_dir: Path, workers=4):
    """E2 replay. params: variants (grades), readings (csv_mm | sheets), n_rep, fit_scale, material (key)."""
    from magsim.validation.experiment import e2_gap_readings, fit_amplitude, load_force_data

    data = load_force_data(params.get("data") or _default_data())
    d = data.alnico
    variants = list(params.get("variants", ["LNG37", "LNG37-femm"]))
    readings = list(params.get("readings", ["csv_mm", "sheets"]))
    n_rep = int(params.get("n_rep", 3))
    mat_key = params.get("material", "rod")
    base = _strip(cfg)
    gaps_mm = list(dict.fromkeys(d["gap_mm"].tolist()))  # measured order
    y = d["F_N"].to_numpy()  # gap-major, rep 1..3: the replay order
    t0 = time.time()
    combos = [(v, rd) for v in variants for rd in readings]
    gapmap = {rd: e2_gap_readings(np.array(gaps_mm))[rd] for rd in readings}

    def run(jobs):
        args = [(_e2_cfg(base, v, s, mat_key), gapmap[rd], n_rep) for v, rd, s in jobs]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(args))) as pool:
                return list(pool.map(_e2_run, args))
        return [_e2_run(a) for a in args]

    pred = dict(zip(combos, run([(v, rd, 1.0) for v, rd in combos])))
    fits = {}
    if params.get("fit_scale", True):
        # Br_scale fit per combo on a quadratic surrogate over a +-5 % grid, re-centred until the optimum lies
        # inside the grid (F is far from ~s^2 for Alnico on the knee), then checked by a direct replay
        grids = {}
        for v, rd in combos:
            f = pred[(v, rd)]["F"].to_numpy()
            grids[(v, rd)] = math.sqrt(max(float(np.sum(y * f) / np.sum(f * f)), 1e-6)) * np.array([0.95, 1.0, 1.05])
        todo = list(combos)
        for _ in range(4):
            if not todo:
                break
            jobs = [(v, rd, s_) for v, rd in todo for s_ in grids[(v, rd)]]
            res = run(jobs)
            nxt = []
            for i, (v, rd) in enumerate(todo):
                g = grids[(v, rd)]
                Fg = np.array([r["F"].to_numpy() for r in res[3 * i:3 * i + 3]])
                ft = fit_amplitude(g, Fg, np.arange(y.size), y, data.sigma_N, n_boot=1000,
                                   bounds=(0.7 * g[0], 1.3 * g[-1]))
                fits[(v, rd)] = ft
                if not g[0] <= ft["theta"] <= g[-1]:
                    grids[(v, rd)] = ft["theta"] * np.array([0.95, 1.0, 1.05])
                    nxt.append((v, rd))
            todo = nxt
        direct = run([(v, rd, fits[(v, rd)]["theta"]) for v, rd in combos])
        for (v, rd), dr in zip(combos, direct):
            ft = fits[(v, rd)]
            ft["surrogate_err_N"] = float(np.max(np.abs(dr["F"].to_numpy() - ft["fitted"])))
            ft["inside_grid"] = bool(grids[(v, rd)][0] <= ft["theta"] <= grids[(v, rd)][-1])
    fem = {}
    if params.get("fem", False):
        sh = base["magnets"][0]["shape"]
        jobs = [({"grade": v}, 0.5 * sh["diameter"], sh["height"], gapmap[rd], n_rep, float(params.get("fem_h", 0.3e-3)))
                for v, rd in combos]
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            fem = dict(zip(combos, [pd.DataFrame(r) for r in pool.map(_e2_fem, jobs)]))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, long = [], []
    meas = d.groupby("gap_mm", sort=False)["F_N"]
    m_mean = meas.mean().to_numpy()
    m_dec = (d[d.rep == n_rep]["F_N"].to_numpy() - d[d.rep == 1]["F_N"].to_numpy()) / d[d.rep == 1]["F_N"].to_numpy()
    models = [("MoM", pred)] + ([("FEM", fem)] if fem else [])
    for model, src in models:
        for (v, rd), p in src.items():
            f = p["F"].to_numpy()
            res_ = y - f
            F1 = p[p.rep == 1]["F"].to_numpy()
            Fn = p[p.rep == n_rep]["F"].to_numpy()
            row = {"model": model, "material": v, "gap reading": rd, "F(0) pred": f[0], "F(0) meas": m_mean[0],
                   "RMS (N)": math.sqrt(np.mean(res_**2)), "max abs (N)": np.max(np.abs(res_)),
                   "chi2/dof (dof=%d)" % y.size: float(np.sum(res_**2) / data.sigma_N**2 / y.size),
                   "pred. decay rep1->%d" % n_rep: float(np.max(np.abs((Fn - F1) / F1))),
                   "J_B at contact (T)": float(p["J_mean_B"].iloc[0])}
            rows.append(row)
            long.append(p.assign(model=model, material=v, reading=rd, F_meas=y))
    tab = pd.DataFrame(rows)
    pd.concat(long).to_csv(out_dir / "e2_replay.csv", index=False)
    tab.to_csv(out_dir / "e2_summary.csv", index=False)
    ftab = None
    if fits:
        ftab = pd.DataFrame([{"material": v, "gap reading": rd, "fit Br_scale": ft["theta"],
                              "95% CI (Jac)": f"[{ft['ci_jac'][0]:.3f}, {ft['ci_jac'][1]:.3f}]",
                              "95% CI (boot)": f"[{ft['ci_boot'][0]:.3f}, {ft['ci_boot'][1]:.3f}]",
                              "RMS (N)": ft["rms"], "max abs (N)": ft["max_abs_res"],
                              "chi2/dof (dof=%d)" % ft["dof"]: ft["chi2_dof"],
                              "surrogate err (N)": ft["surrogate_err_N"]} for (v, rd), ft in fits.items()])
        ftab.to_csv(out_dir / "e2_scale_fit.csv", index=False)

    plt, plotting = _style()
    fig, axs = plt.subplots(len(readings), 1, figsize=(plotting.IEEE_COL, 1.75 * len(readings) + 0.5), sharey=True,
                            squeeze=False)
    for ax, rd in zip(axs[:, 0], readings):
        gm = gapmap[rd] * 1e3
        ax.errorbar(gm, m_mean, yerr=data.sigma_N, fmt="o", ms=2.5, color="0.2", elinewidth=0.6, capsize=1.5,
                    label="measured (mean of 3, $\\pm$0.25 N)", zorder=5)
        for i, v in enumerate(variants):
            col = plotting.OKABE_ITO[5 if i == 0 else 6]
            for model, src in models:
                fm = src[(v, rd)].groupby("gap_m", sort=False)["F"].mean().to_numpy()
                ax.plot(gm, fm, "-" if model == "MoM" else "--", color=col, label=f"{v}, {model}")
        ax.set_ylabel("attraction $-F_z$ (N)")
        ax.set_xlabel("model gap (mm): " + ("CSV value in mm" if rd == "csv_mm" else "2 x CSV value (0.10 mm sheets)"))
    axs[0, 0].legend(loc="upper right", fontsize=6)
    plotting.save(fig, out_dir / "e2_replay")
    plt.close(fig)

    md = [
        "## E2: LNG37 rod pair, history replay (gate F7)",
        "",
        f"Protocol: +sat -> isolate (open circuit, committed) -> for each gap in the measured order "
        f"({', '.join(f'{g:g}' for g in gaps_mm)} CSV units): {n_rep} x (place at the gap, measure, separate). "
        f"Mesh {cfg['magnets'][0].get('mesh')} per rod. Two readings of the gap column (SPEC open question): "
        "`csv_mm` = value in mm; `sheets` = value/0.05 sheets of 0.10 mm (gap = 2 x value). "
        "**No fitted parameter** in the prediction columns (pure prediction from the library grades: vendor `LNG37`, "
        "mu_rec 3.05, and `LNG37-femm`, mu_rec 2.02, fitted to the FEMM table); chi^2/dof there uses dof = N = "
        f"{y.size}. Models: MoM (full-ring mesh, which cannot carry radial polarisation, i.e. ignores mu_perp; "
        "see F6) and, when enabled, the axisymmetric FEM (`validation/axisym_fem.py`, P2, h = "
        f"{float(params.get('fem_h', 0.3e-3)) * 1e3:g} mm, (6, 36) irreversible slabs per rod, mu_perp included; "
        "contact evaluated at 1 um).",
        "",
        md_table(tab, {"F(0) pred": "{:.3f}", "F(0) meas": "{:.2f}", "RMS (N)": "{:.3f}", "max abs (N)": "{:.3f}",
                       "J_B at contact (T)": "{:.3f}", "pred. decay rep1->%d" % n_rep: "{:.1e}"}),
        "",
        *([f"Separate one-parameter fit (MoM): Br multiplied by `Br_scale`, Preisach shape and coercivity fixed; "
           f"dof = N - 1 = {y.size - 1}.", "",
           md_table(ftab, {"fit Br_scale": "{:.3f}", "RMS (N)": "{:.3f}", "max abs (N)": "{:.3f}",
                           "surrogate err (N)": "{:.1e}"}), ""] if ftab is not None else []),
        "Where the Jacobian and bootstrap intervals of `Br_scale` disagree, the scale is poorly identified: for "
        "Alnico on the knee of its demagnetisation curve the open-circuit operating point is pinned by the "
        "coercivity distribution, so the force responds weakly and nonlinearly to Br (a scaled Br also deepens "
        "the self-demagnetisation). The prediction (no fit) is the primary result.",
        "",
        f"Measured change from reading 1 to {n_rep} per gap (relative): "
        + ", ".join(f"{v:+.2f}" for v in m_dec)
        + f" (mean {np.mean(m_dec):+.3f}); the readings are quantised to 0.05 N, so changes below ~0.05/F are "
        "not resolved. Predicted change: the column `pred. decay` (max over gaps). Classical Preisach "
        "[mayergoyz1991] has wiping-out and congruency but no accommodation: after the first pull-off the "
        "operating point returns to the same open-circuit state, every repeated place/separate cycle retraces "
        "the same recoil (minor) loop, and the predicted force repeats exactly; excursions smaller than "
        "~2 Hc_min are reversible (no minor-loop width). Any systematic decay in the data would therefore be "
        "unmodelled accommodation / handling, not Preisach memory.",
        "",
        f"Figure: `e2_replay.pdf`. Runtime {time.time() - t0:.0f} s.",
        "",
    ]
    ok = all(bool(p["all_converged"].all()) for p in pred.values())
    return "\n".join(md), ok


def _default_data():
    from magsim.validation.experiment import DATA

    return DATA


# ---------------------------------------------------------------------------------------------
# mesh convergence
# ---------------------------------------------------------------------------------------------
def check_convergence(cfg, params, out_dir: Path, workers=4):
    """Mesh study. params: base_n (per magnet), factors (>= 3), outputs, set ({path: value}), label."""
    from magsim.validation.convergence import gci_table, mesh_study

    base = _strip(cfg)
    for p, v in (params.get("set") or {}).items():
        base = set_path(base, p, v)
    base_n = params.get("base_n") or [list((m.get("mesh") or {}).get("n", m.get("mesh"))) for m in base["magnets"]]
    factors = params.get("factors", [1, 2, 3])
    outs = params.get("outputs", ["Fz"])
    label = params.get("label") or f"{cfg.get('name')}" + (f" ({params['set']})" if params.get("set") else "")
    t0 = time.time()
    df = mesh_study(base, base_n, factors, workers=min(3, workers))
    gt = gci_table(df, outs)
    stem = "convergence_" + "".join(c if c.isalnum() else "_" for c in label).strip("_")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    gt.to_csv(out_dir / f"{stem}_gci.csv", index=False)

    plt, plotting = _style()
    fig, ax = plt.subplots(figsize=(plotting.IEEE_COL, 2.3))
    for i, o in enumerate(outs):
        g = gt[gt.output == o].iloc[0]
        ax.loglog(df["h"] * 1e3, np.abs(df[o] / g["phi_ext"] - 1), "o-", color=plotting.OKABE_ITO[(i + 5) % 8],
                  label=f"{o}: p = {g['p']:.2f}, GCI$_{{21}}$ = {100 * g['gci21']:.2g} %")
    ax.set_xlabel("representative cell size $h$ (mm)")
    ax.set_ylabel("$|\\phi/\\phi_{ext} - 1|$")
    ax.legend(loc="lower right")
    plotting.save(fig, out_dir / stem)
    plt.close(fig)
    show = df[["k", "mesh", "n_cells", "h", *outs]].copy()
    show["h"] = show["h"] * 1e3
    show = show.rename(columns={"h": "h (mm)"})
    md = [
        f"## Mesh convergence: {label}",
        "",
        "Systematic refinement (every cell count x k; n_phi = 1 of full-ring cylinders kept), representative "
        "h = (V/N)^(1/3) (full-ring meshes: (V/N)^(1/2) V^(-1/6), refined in r and z only) [celik2008]; GCI with "
        "Fs = 1.25 on the three finest meshes.",
        "",
        md_table(show, {"h (mm)": "{:.3f}", "n_cells": "{:.0f}", "k": "{:g}"}, default="{:.6g}"),
        "",
        md_table(gt[["output", "phi1", "phi_ext", "r21", "r32", "p", "s", "e_a21", "e_ext21", "gci21", "asymptotic"]],
                 {"phi1": "{:.6g}", "phi_ext": "{:.6g}", "r21": "{:.3f}", "r32": "{:.3f}", "p": "{:.2f}", "s": "{:+.0f}",
                  "e_a21": "{:.2e}", "e_ext21": "{:.2e}", "gci21": "{:.2e}", "asymptotic": "{:.3f}"}),
        "",
        f"Figure: `{stem}.pdf`. Runtime {time.time() - t0:.0f} s.",
        "",
    ]
    ok = bool(df["all_converged"].all()) if "all_converged" in df else True
    return "\n".join(md), ok


# ---------------------------------------------------------------------------------------------
# F6 (and F3 for chi = 3): MoM vs axisymmetric FEM vs 1-D slab model
# ---------------------------------------------------------------------------------------------
def _mom_rod(args):
    """MoM open circuit (+sat, isolate) and pair at gap (place, measure) for two identical rods."""
    from magsim.force_sim import ForceScenario, body_outputs

    mat_cfg, R, L, gap, n = args
    cfg = {"name": "f6", "materials": {"rod": mat_cfg}, "target": "B",
           "magnets": [{"name": "A", "shape": {"type": "cylinder", "diameter": 2 * R, "height": L},
                        "material": "rod", "pose": {"position": [0, 0, -L / 2]}, "mesh": {"n": list(n)}},
                       {"name": "B", "shape": {"type": "cylinder", "diameter": 2 * R, "height": L},
                        "material": "rod", "mesh": {"n": list(n)}}],
           "layout": {"body": "B", "reference": "A", "gap": gap}}
    fs = ForceScenario(cfg)
    s = fs.isolate()
    bo = body_outputs(fs.system, s)
    fs.place()
    out = fs.measure()
    V = fs.system.volume[fs.system.body_slice("A")]
    return {"J_oc": bo["J_mean_A"], "N_oc": -MU0 * bo["H_par_mean_A"] / bo["J_mean_A"], "F": out["Fz"],
            "cells": int(V.size), "h": float((V.sum() / V.size) ** (1 / 3))}


def _fem_rod(args):
    from magsim.force_sim import build_material
    from magsim.validation.axisym_fem import fem_rod_case

    mat_cfg, R, L, gap, h, slabs = args
    r = fem_rod_case(build_material(mat_cfg), R, L, gap, h, slabs=tuple(slabs))
    return {k: float(v) for k, v in r.items()}


def _onedim(mat_cfg, R, L, gap, n_slabs):
    from magsim.analytic import CoaxialRodPair
    from magsim.force_sim import build_material

    rod = CoaxialRodPair(R, L, build_material(mat_cfg), n_slabs=n_slabs)
    J_free, H_free = rod.solve(200 * L)
    rod.open_circuit()
    n = n_slabs
    return {"J_oc": float(J_free[:n].mean()), "N_oc": float(-MU0 * H_free[:n].mean() / J_free[:n].mean()),
            "F": rod.force(gap)}


def check_f6(cfg, params, out_dir: Path, workers=4):
    """Gate F6 (and F3 at chi = 3). params: radius, length, gap, layouts [(n_r, n_z)...] (MoM cells = FEM slabs),
    h (FEM P2 sizes, m), n_slabs_1d."""
    from magsim.analytic import coaxial_cylinder_force, cylinder_demag_factor
    from magsim.validation.convergence import gci

    R = float(params.get("radius", 4.75e-3 / 2))
    L = float(params.get("length", 12.5e-3))
    gap = float(params.get("gap", 0.1e-3))
    layouts = [tuple(x) for x in params.get("layouts", [(4, 12), (6, 18), (8, 24)])]
    hs = [float(x) for x in params.get("h", [0.4e-3, 0.3e-3, 0.2e-3])]
    n1d = int(params.get("n_slabs_1d", 96))
    cases = params.get("cases") or [
        ("rigid", {"type": "linear", "Jr": 1.2, "mu_rec": 1.0, "mu_perp": 1.0}),
        ("linear mu_rec = mu_perp = 4", {"type": "linear", "Jr": 1.2, "mu_rec": 4.0, "mu_perp": 4.0}),
        ("linear mu_rec = 4, mu_perp = 1", {"type": "linear", "Jr": 1.2, "mu_rec": 4.0, "mu_perp": 1.0}),
        ("LNG37 (library, mu_perp = mu_rec)", {"grade": "LNG37"}),
        ("LNG37, mu_perp = 1", {"grade": "LNG37", "mu_perp": 1.0}),
    ]
    cases = [(c[0], c[1]) if isinstance(c, (list, tuple)) else (c["label"], c["material"]) for c in cases]
    t0 = time.time()
    mom_jobs = [(m, R, L, gap, (nr, 1, nz)) for _, m in cases for nr, nz in layouts]
    fem_jobs = [(m, R, L, gap, hs[-1], lay) for _, m in cases for lay in layouts]
    fem_h_jobs = [(m, R, L, gap, h, layouts[-1]) for _, m in cases for h in hs[:-1]]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        mom = list(pool.map(_mom_rod, mom_jobs))
        fem = list(pool.map(_fem_rod, fem_jobs))
        fem_h = list(pool.map(_fem_rod, fem_h_jobs))
    nl, nh = len(layouts), len(hs) - 1
    rows, raw = [], []
    for ci, (label, m) in enumerate(cases):
        Mo = mom[ci * nl:(ci + 1) * nl]
        Fe = fem[ci * nl:(ci + 1) * nl]
        Fh = fem_h[ci * nh:(ci + 1) * nh] + [Fe[-1]]
        one = _onedim(m, R, L, gap, n1d)
        hcell = [x["h"] for x in Mo]
        rec = {"case": label}
        for q in ("J_oc", "F", "N_oc"):
            gm = gci([x[q] for x in Mo[::-1]], hcell[::-1])
            gf = gci([x[q] for x in Fe[::-1]], hcell[::-1])  # FEM slab layout = MoM cell layout
            gh = gci([x[q] for x in Fh[::-1]], hs[::-1])  # FEM h at the finest layout
            rec.update({f"{q} MoM": Mo[-1][q], f"{q} MoM ext": gm["phi_ext"], f"{q} MoM GCI": gm["gci21"],
                        f"{q} FEM": Fe[-1][q], f"{q} FEM ext": gf["phi_ext"], f"{q} FEM GCI": gf["gci21"],
                        f"{q} FEM h-GCI": gh["gci21"], f"{q} FEM h-p": gh["p"], f"{q} 1D": one[q]})
        if m.get("type") == "linear" and m.get("mu_rec", 1.0) == 1.0:
            rec["F analytic"] = coaxial_cylinder_force(m["Jr"], m["Jr"], R, L, L, gap)
            rec["N analytic"] = cylinder_demag_factor(R, L)
        rows.append(rec)
        for lay, a, b in zip(layouts, Mo, Fe):
            raw.append({"case": label, "layout": str(lay), **{f"MoM {k}": v for k, v in a.items()},
                        **{f"FEM {k}": v for k, v in b.items()}})
        for h, b in zip(hs, Fh):
            raw.append({"case": label, "layout": str(layouts[-1]), "FEM h": h, **{f"FEM {k}": v for k, v in b.items()}})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "f6_summary.csv", index=False)
    pd.DataFrame(raw).to_csv(out_dir / "f6_raw.csv", index=False)

    def rel(a, b):
        return a / b - 1.0

    t1 = pd.DataFrame({
        "case": df["case"],
        "J_oc MoM ext (T)": df["J_oc MoM ext"], "J_oc FEM ext (T)": df["J_oc FEM ext"], "J_oc 1-D (T)": df["J_oc 1D"],
        "MoM/FEM-1": rel(df["J_oc MoM ext"], df["J_oc FEM ext"]), "1D/FEM-1": rel(df["J_oc 1D"], df["J_oc FEM ext"]),
        "GCI MoM": df["J_oc MoM GCI"], "GCI FEM slabs": df["J_oc FEM GCI"], "FEM h-GCI": df["J_oc FEM h-GCI"],
    })
    t2 = pd.DataFrame({
        "case": df["case"],
        "F MoM ext (N)": df["F MoM ext"], "F FEM ext (N)": df["F FEM ext"], "F 1-D (N)": df["F 1D"],
        "MoM/FEM-1": rel(df["F MoM ext"], df["F FEM ext"]), "1D/FEM-1": rel(df["F 1D"], df["F FEM ext"]),
        "GCI MoM": df["F MoM GCI"], "GCI FEM slabs": df["F FEM GCI"], "FEM h-GCI": df["F FEM h-GCI"],
    })
    t3 = pd.DataFrame({"case": df["case"], "N MoM ext": df["N_oc MoM ext"], "N FEM ext": df["N_oc FEM ext"],
                       "N 1-D": df["N_oc 1D"], "MoM/FEM-1": rel(df["N_oc MoM ext"], df["N_oc FEM ext"])})
    rig = df.iloc[0]
    f_rig = rel(rig["F FEM"], rig["F analytic"])
    n_rig = rel(rig["N_oc FEM"], rig["N analytic"])
    fm = {c: "{:.5g}" for c in ("J_oc MoM ext (T)", "J_oc FEM ext (T)", "J_oc 1-D (T)", "F MoM ext (N)", "F FEM ext (N)",
                                "F 1-D (N)", "N MoM ext", "N FEM ext", "N 1-D")}
    fm.update({c: "{:+.2%}" for c in ("MoM/FEM-1", "1D/FEM-1")})
    fm.update({c: "{:.1e}" for c in ("GCI MoM", "GCI FEM slabs", "FEM h-GCI")})
    # agreement where both methods represent the same model (mu_perp = 1 cases)
    same = [i for i, (_, m) in enumerate(cases) if float(m.get("mu_perp", 0.0 if "grade" in m else 1.0)) == 1.0]
    worst = max(max(abs(t1["MoM/FEM-1"].iloc[i]), abs(t2["MoM/FEM-1"].iloc[i])) for i in same) if same else math.nan
    md = [
        "## F6 / F3: coaxial rods, MoM vs axisymmetric FEM vs 1-D slab model",
        "",
        f"Rods D{2 * R * 1e3:.2f} x {L * 1e3:.1f} mm, pair gap {gap * 1e3:g} mm. History for every method: +sat -> one rod "
        "alone (open circuit, committed) -> pair at the gap. MoM: cells (n_r, 1, n_z) in "
        f"{layouts} (full rings), Richardson / GCI over that sequence. FEM (`validation/axisym_fem.py`, P2, A_phi): "
        "irreversible part uniform over the same (n_r, n_z) slab grid (reversible part elementwise), extrapolated "
        f"over the same sequence at h = {hs[-1] * 1e3:g} mm; FEM h-GCI from h = {[h * 1e3 for h in hs]} mm at the finest "
        f"layout. 1-D: `analytic.CoaxialRodPair`, {n1d} slabs per rod (J radially uniform).",
        "",
        f"FEM self-check (rigid): pair force vs Lipschitz-Hankel {f_rig:+.1e}, open-circuit demag factor vs "
        f"Chen-Brug-Goldfarb {n_rig:+.1e} (finest layout, h = {hs[-1] * 1e3:g} mm).",
        "",
        "Open-circuit volume-mean J:",
        "",
        md_table(t1, fm),
        "",
        f"Force on the upper rod at {gap * 1e3:g} mm (negative = attraction):",
        "",
        md_table(t2, fm),
        "",
        "Open-circuit factor -MU0<H>/<J> (gate F3 for chi = 3: the 'linear mu_rec = mu_perp = 4' row is the "
        "isotropic chi = 3 factor, reference = FEM):",
        "",
        md_table(t3, fm),
        "",
        "**Explained discrepancy (MoM limitation, reported to the orchestrator):** a cylinder or ring meshed in "
        "full rings (n_phi = 1, the default) has one uniform polarisation VECTOR per ring, so it cannot carry a "
        "radial polarisation J_r e_r (its ring average is zero): the transverse susceptibility mu_perp - 1 is "
        "silently ignored. The MoM rows are therefore identical for mu_perp = 4 and mu_perp = 1, and they agree "
        "with the FEM when the FEM also has mu_perp = 1. With the library's Alnico mu_perp = mu_rec (an "
        "ASSUMPTION of the grade table), the full-ring MoM underestimates J_oc and the force. Segment meshes "
        "(n_phi > 1) recover it slowly. NdFeB (mu_perp ~ 1.05) is barely affected. "
        f"Worst MoM-FEM difference over the mu_perp = 1 cases: {worst:.2%} (gate < 2 %).",
        "",
        f"Runtime {time.time() - t0:.0f} s.",
        "",
    ]
    ok = abs(f_rig) < 2e-3 and abs(n_rig) < 1e-3 and (not same or worst < 0.02)
    return "\n".join(md), ok


# ---------------------------------------------------------------------------------------------
# magpylib.getFT cross-check (rigid magnets)
# ---------------------------------------------------------------------------------------------
def magpylib_crosscheck(levels=(1, 2, 4, 8)) -> pd.DataFrame:
    """Rigid tilted cuboid-on-cuboid and cylinder-on-cuboid: magsim charge-integration F, T vs
    ``magpylib.getFT`` (target cut into dipole cells, centroid pivot) [ortner2020].

    The magpylib target mesh is refined systematically: cuboid (10, 10, 5) x k divisions, cylinder
    (n_phi, n_r, n_h) = (12, 3, 4) x k, for k in ``levels`` (the dipole-cell error is O(h^2)). Returns rows
    (case, k, cells, relF = |F_mp - F_ms| / |F_ms|, relT = |T_mp - T_ms| / |T_ms|).
    """
    import magpylib as mp
    from scipy.spatial.transform import Rotation

    from magsim.forces import force_torque_detail
    from magsim.geometry import Cuboid, Cylinder, Magnet, Pose
    from magsim.materials.base import LinearMaterial
    from magsim.mom import MagnetSystem

    JA = np.array([0.0, 0.0, 1.2])
    pA = np.zeros(3)
    rot = Rotation.from_rotvec(np.deg2rad(25.0) * np.array([1.0, 1.0, 0.0]) / math.sqrt(2))
    pB = np.array([3e-3, -2e-3, 10e-3])
    JB = np.array([0.3, 0.0, 1.0])  # body frame
    cases = [("cuboid 10x10x5 on cuboid", "cuboid"), ("cylinder D8x6 on cuboid", "cylinder")]
    rows = []
    for label, kind in cases:
        A = Magnet("A", Cuboid((20e-3, 10e-3, 5e-3)), Pose(pA), LinearMaterial(1.2), mesh=(1, 1, 1))
        if kind == "cuboid":
            shB = Cuboid((10e-3, 10e-3, 5e-3))
            mpB = mp.magnet.Cuboid(dimension=(10e-3, 10e-3, 5e-3), polarization=JB, position=pB, orientation=rot)
        else:
            shB = Cylinder(8e-3, 6e-3)
            mpB = mp.magnet.Cylinder(dimension=(8e-3, 6e-3), polarization=JB, position=pB, orientation=rot)
        B = Magnet("B", shB, Pose(pB, rot), LinearMaterial(float(np.linalg.norm(JB))), easy_axis=JB, mesh=(1, 1, 1))
        s = MagnetSystem([A, B])
        sol = s.solve()
        fr = force_torque_detail(s, sol, "B", pivot=pB, quad_tol=1e-12, level=1)
        mpA = mp.magnet.Cuboid(dimension=(20e-3, 10e-3, 5e-3), polarization=JA, position=pA)
        eps = 1e-6 * 20e-3  # FD step for the field gradient: 1e-6 x source size (magpylib docs)
        prev = None
        for k in levels:
            div = tuple(int(k * v) for v in ((10, 10, 5) if kind == "cuboid" else (12, 3, 4)))
            F, T = mp.getFT(mpA, mpB, pivot=pB, eps=eps, meshing=div)
            F, T = np.asarray(F).reshape(3), np.asarray(T).reshape(3)
            row = {"case": label, "k": int(k), "cells": int(np.prod(div)), "F_magsim (N)": str(np.round(fr.F, 6)),
                   "relF": float(np.linalg.norm(F - fr.F) / np.linalg.norm(fr.F)),
                   "relT": float(np.linalg.norm(T - fr.T) / np.linalg.norm(fr.T)),
                   "relF Richardson": math.nan, "relT Richardson": math.nan}
            if prev is not None and k == 2 * prev[0]:  # Richardson with p = 2 on the magpylib sequence
                Fr, Tr = (4 * F - prev[1]) / 3, (4 * T - prev[2]) / 3
                row["relF Richardson"] = float(np.linalg.norm(Fr - fr.F) / np.linalg.norm(fr.F))
                row["relT Richardson"] = float(np.linalg.norm(Tr - fr.T) / np.linalg.norm(fr.T))
            rows.append(row)
            prev = (k, F, T)
    return pd.DataFrame(rows)


def check_magpylib(cfg, params, out_dir: Path, workers=4):
    """Independent force check vs magpylib.getFT. params: levels (mesh refinement factors k)."""
    t0 = time.time()
    df = magpylib_crosscheck(tuple(params.get("levels", (1, 2, 4, 8))))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "magpylib_crosscheck.csv", index=False)
    ok = True
    for _, g in df.groupby("case"):
        ok &= bool(np.all(np.diff(g["relF"].to_numpy()) < 0)) and float(g["relF"].iloc[-1]) < 1e-4
        ok &= float(g["relF Richardson"].iloc[-1]) < 1e-5
    md = [
        "## Independent check: magpylib.getFT (rigid magnets)",
        "",
        "Rigid tilted target (25 deg about (1,1,0), polarisation (0.3, 0, 1.0) T in its body frame) above a "
        "20 x 10 x 5 mm cuboid (1.2 T along z). magsim: exact-kernel charge integration (quad_tol 1e-12); "
        "magpylib 6 `getFT`: target cut into dipole cells (cuboid (10,10,5) x k, cylinder (n_phi, n_r, n_h) = "
        "(12,3,4) x k), field gradient by central differences with eps = 2e-8 m [ortner2020]. The difference "
        "must shrink as the magpylib mesh is refined (dipole-cell error O(h^2)); the Richardson columns "
        "extrapolate the magpylib pair (k/2, k) with p = 2.",
        "",
        md_table(df, {c: "{:.2e}" for c in ("relF", "relT", "relF Richardson", "relT Richardson")}),
        "",
        f"Runtime {time.time() - t0:.0f} s.",
        "",
    ]
    return "\n".join(md), ok


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------
CHECKS = {"e1_fit": check_e1, "e2_replay": check_e2, "convergence": check_convergence, "f6": check_f6,
          "magpylib": check_magpylib}


def run_validation(cfg: dict, out_dir: str | Path, workers: int = 4, scenario_path=None) -> bool:
    """Run the scenario's validation checks; write ``out_dir/validation.md``. True if every check passed."""
    from magsim.sweep import write_provenance

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val = cfg.get("validation")
    if val is None:
        base_n = [list((m.get("mesh") or {}).get("n", (m.get("mesh") or [2, 2, 2]))) for m in cfg["magnets"]]
        unit = [[max(1, int(round(v / 2))) for v in n] for n in base_n]
        checks = [("convergence", {"base_n": unit, "factors": [2, 3, 4]})]
    else:
        checks = []
        for c in val.get("checks", []):
            if isinstance(c, str):
                checks.append((c, {}))
            elif isinstance(c, dict) and len(c) == 1:
                k, v = next(iter(c.items()))
                checks.append((k, dict(v or {})))
            else:
                raise ValueError(f"validation.checks: bad entry {c!r}")
    parts = [f"# Validation report: {cfg.get('name')}", "",
             f"Scenario: `{scenario_path}`. Generated by `magsim force validate`; numbers below are computed, "
             "not copied. Provenance (git commit, versions) in `provenance.json`.", ""]
    status = []
    for name, params in checks:
        if name not in CHECKS:
            raise ValueError(f"unknown validation check {name!r} (have {sorted(CHECKS)})")
        c = load_scenario(params.pop("scenario")) if "scenario" in params else cfg
        md, ok = CHECKS[name](c, params, out_dir, workers)
        parts.append(md)
        status.append((name, params.get("label", c.get("name")), ok))
        print(f"[validate] {name} ({c.get('name')}): {'ok' if ok else 'FAILED'}", flush=True)
    summary = ["## Summary", "", "| check | case | status |", "|---|---|---|"]
    summary += [f"| {n} | {lab} | {'pass' if ok else 'FAIL'} |" for n, lab, ok in status]
    parts[4:4] = summary + [""]
    (out_dir / "validation.md").write_text("\n".join(parts), encoding="utf-8")
    write_provenance(out_dir, extra={"scenario": str(scenario_path), "checks": [c[0] for c in checks]})
    return all(ok for *_, ok in status)
