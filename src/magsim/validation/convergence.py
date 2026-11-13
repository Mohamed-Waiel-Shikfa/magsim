"""Mesh refinement, observed order, Richardson extrapolation and the Grid Convergence Index [celik2008].

Celik et al., "Procedure for estimation and reporting of uncertainty due to discretization in CFD
applications", J. Fluids Eng. 130, 078001 (2008), section 2 ("five-step procedure"):

    h = (V_total / N_cells)^(1/3)                         representative cell size (3-D)
    r21 = h2 / h1, r32 = h3 / h2   (h1 < h2 < h3: fine, medium, coarse)
    e21 = phi2 - phi1, e32 = phi3 - phi2, s = sign(e32 / e21)
    p = |ln|e32 / e21| + q(p)| / ln r21,  q(p) = ln((r21^p - s) / (r32^p - s))   (fixed-point iteration)
    phi_ext21 = (r21^p phi1 - phi2) / (r21^p - 1)
    e_a21 = |(phi1 - phi2) / phi1|,  e_ext21 = |(phi_ext21 - phi1) / phi_ext21|
    GCI_fine21 = 1.25 e_a21 / (r21^p - 1)

s < 0 flags oscillatory convergence (reported, p from |e32/e21| as in the paper). The ratio
GCI32 / (r21^p GCI21) ~ 1 indicates the asymptotic range.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from magsim.config import set_path

__all__ = ["gci", "representative_h", "scale_mesh", "mesh_study", "gci_table"]


def representative_h(volume: float, n_cells: int, dim: int = 3) -> float:
    """Representative cell size [m]: (V / N)^(1/3) [celik2008, eq. 1] for dim = 3. For a mesh refined in
    only ``dim`` < 3 directions (full-ring cylinders: r and z) h = (V / N)^(1/dim) V^(1/3 - 1/dim), which
    scales as 1/k under refinement by k (the length-unit factor V^(1/3 - 1/dim) cancels in every ratio)."""
    return float((volume / n_cells) ** (1.0 / dim) * volume ** (1.0 / 3.0 - 1.0 / dim))


def gci(phi, h, Fs: float = 1.25, p_fixed: float | None = None, max_iter: int = 200) -> dict:
    """Celik's GCI for three solutions ``phi`` = (phi1, phi2, phi3) on meshes ``h`` = (h1, h2, h3), h1 < h2 < h3.

    Returns p (observed order; ``p_fixed`` overrides it), phi_ext, e_a21, e_ext21, gci21 (fine-grid GCI,
    relative), gci32, asymptotic ratio, s (oscillation sign), r21, r32.
    """
    p1, p2, p3 = (float(v) for v in phi)
    h1, h2, h3 = (float(v) for v in h)
    if not h1 < h2 < h3:
        raise ValueError(f"need h1 < h2 < h3, got {h}")
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = p2 - p1, p3 - p2
    out = {"phi1": p1, "phi2": p2, "phi3": p3, "r21": r21, "r32": r32}
    if e21 == 0.0 or e32 == 0.0:
        out.update(p=math.nan, s=0.0, phi_ext=p1, e_a21=abs(e21 / p1) if p1 else 0.0, e_ext21=0.0,
                   gci21=0.0, gci32=0.0, asymptotic=math.nan)
        return out
    s = float(np.sign(e32 / e21))
    if p_fixed is not None:
        p = float(p_fixed)
    else:
        p, q = 1.0, 0.0
        for _ in range(max_iter):
            p_new = abs(math.log(abs(e32 / e21)) + q) / math.log(r21)
            q_arg = (r21**p_new - s) / (r32**p_new - s)
            q = math.log(q_arg) if q_arg > 0 else 0.0
            if abs(p_new - p) < 1e-12:
                p = p_new
                break
            p = p_new
    rp = r21**p
    phi_ext = (rp * p1 - p2) / (rp - 1.0)
    e_a21 = abs((p1 - p2) / p1)
    e_a32 = abs((p2 - p3) / p2)
    e_ext = abs((phi_ext - p1) / phi_ext)
    g21 = Fs * e_a21 / (rp - 1.0)
    g32 = Fs * e_a32 / (r32**p - 1.0)
    out.update(p=p, s=s, phi_ext=phi_ext, e_a21=e_a21, e_ext21=e_ext, gci21=g21, gci32=g32,
               asymptotic=g32 / (rp * g21) if g21 > 0 else math.nan)
    return out


def scale_mesh(n, k: float, fixed=()) -> list[int]:
    """Integer mesh counts n * k (axes in ``fixed`` kept); raises if not integral (refinement must be systematic)."""
    out = []
    for i, v in enumerate(n):
        if i in fixed:
            out.append(int(v))
            continue
        x = float(v) * float(k)
        if abs(x - round(x)) > 1e-9:
            raise ValueError(f"mesh {list(n)} x {k} is not integral")
        out.append(int(round(x)))
    return out


def _fixed_axes(cfg: dict, i: int, n) -> tuple:
    """Axes not refined: n_phi = 1 of a cylinder / ring (full rings stay full rings)."""
    t = cfg["magnets"][i]["shape"]["type"]
    return (1,) if t in ("cylinder", "ring") and int(n[1]) == 1 else ()


def _mesh_cfg(cfg: dict, base_n: list, k: float) -> dict:
    out = cfg
    for i, n in enumerate(base_n):
        mesh = cfg["magnets"][i].get("mesh") or {}
        mesh = dict(mesh) if isinstance(mesh, dict) else {}
        mesh["n"] = scale_mesh(n, k, _fixed_axes(cfg, i, n))
        out = set_path(out, f"magnets[{i}].mesh", mesh)
    return out


def _run(cfg):
    from magsim.force_sim import ForceScenario

    fs = ForceScenario(cfg)
    res = fs.run()
    res["n_cells"] = float(fs.system.n_cells)
    res["volume"] = float(fs.system.volume.sum())
    return res


def mesh_study(cfg: dict, base_n: list, factors, workers: int = 3) -> pd.DataFrame:
    """Run the scenario on meshes ``base_n[i] * k`` (every magnet i) for each k; one row per mesh with all
    outputs plus n_cells, volume and the representative h = (V/N)^(1/d), d = number of refined directions
    (2 for full-ring cylinder meshes, whose n_phi = 1 is kept; [celik2008] eq. 1 / 2)."""
    cfgs = [_mesh_cfg(cfg, base_n, k) for k in factors]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(cfgs))) as pool:
            rows = list(pool.map(_run, cfgs))
    else:
        rows = [_run(c) for c in cfgs]
    df = pd.DataFrame(rows)
    df.insert(0, "k", list(factors))
    fixed = [_fixed_axes(cfg, i, n) for i, n in enumerate(base_n)]
    df.insert(1, "mesh", [str([scale_mesh(n, k, f) for n, f in zip(base_n, fixed)]) for k in factors])
    dim = 2 if all(fixed) else 3
    df["h"] = [representative_h(v, int(n), dim) for v, n in zip(df["volume"], df["n_cells"])]
    return df


def gci_table(df: pd.DataFrame, outputs) -> pd.DataFrame:
    """GCI of each output from the three finest rows of a ``mesh_study`` frame (fine = smallest h)."""
    d = df.sort_values("h").iloc[:3]
    rows = []
    for o in outputs:
        g = gci(d[o].to_numpy(), d["h"].to_numpy())
        g["output"] = o
        rows.append(g)
    cols = ["output", "phi1", "phi2", "phi3", "r21", "r32", "p", "s", "phi_ext", "e_a21", "e_ext21", "gci21",
            "gci32", "asymptotic"]
    return pd.DataFrame(rows)[cols]
