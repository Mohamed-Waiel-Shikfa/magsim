"""Parameter sweeps: grid expansion over dotted paths, process-pool map, tidy DataFrame, provenance."""

import datetime
import importlib.metadata
import itertools
import json
import math
import platform
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from magsim.config import set_path

_PACKAGES = ["numpy", "scipy", "magpylib", "matplotlib", "pandas", "scikit-fem", "pyyaml"]


def axis_values(axis: dict) -> list:
    """Values of one axis: exactly one of `values: [...]`, `linspace: [a, b, n]`, `logspace: [a, b, n]`.

    logspace endpoints are the values themselves (np.geomspace), not exponents.
    """
    if "path" not in axis:
        raise ValueError(f"sweep axis needs a 'path': {axis}")
    kinds = [k for k in ("values", "linspace", "logspace") if k in axis]
    if len(kinds) != 1:
        raise ValueError(f"axis {axis['path']!r}: give exactly one of values / linspace / logspace")
    kind = kinds[0]
    spec = axis[kind]
    if kind == "values":
        if not isinstance(spec, list) or not spec:
            raise ValueError(f"axis {axis['path']!r}: values must be a non-empty list")
        return list(spec)
    if not isinstance(spec, list) or len(spec) != 3:
        raise ValueError(f"axis {axis['path']!r}: {kind} must be [start, stop, n]")
    a, b, n = float(spec[0]), float(spec[1]), int(spec[2])
    if n < 1:
        raise ValueError(f"axis {axis['path']!r}: n must be >= 1")
    grid = np.linspace(a, b, n) if kind == "linspace" else np.geomspace(a, b, n)
    return [float(v) for v in grid]


def expand_axes(axes: list[dict]) -> list[dict[str, Any]]:
    """Full Cartesian grid over the axes, first axis slowest: list of {path: value}."""
    grids = [axis_values(ax) for ax in axes]
    paths = [ax["path"] for ax in axes]
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate sweep paths: {paths}")
    return [dict(zip(paths, combo)) for combo in itertools.product(*grids)]


def _run_point(fn: Callable[[dict], dict], base_cfg: dict, point: dict) -> tuple[dict, str | None]:
    try:
        cfg = base_cfg
        for path, value in point.items():
            cfg = set_path(cfg, path, value)
        out = fn(cfg)
        return {k: float(v) for k, v in out.items()}, None
    except Exception as exc:  # one bad point must not abort the sweep
        return {}, f"{type(exc).__name__}: {exc}"


def run_sweep(
    base_cfg: dict,
    axes: list[dict],
    fn: Callable[[dict], dict[str, float]],
    workers: int = 4,
    out_dir: str | Path | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Evaluate fn(cfg) -> {output: float} on every grid point; return inputs + outputs, grid order.

    fn must be a picklable top-level function when workers > 1. A failing point gets NaN outputs
    and its message in the `error` column. With out_dir, writes <out_dir>[/<name>]/sweep.csv,
    scenario.yaml (base config) and provenance.json.
    """
    points = expand_axes(axes)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_point, [fn] * len(points), [base_cfg] * len(points), points))
    else:
        results = [_run_point(fn, base_cfg, p) for p in points]

    out_names: list[str] = []
    for outputs, _ in results:
        out_names += [k for k in outputs if k not in out_names]
    rows = []
    for point, (outputs, err) in zip(points, results):
        row = dict(point)
        row.update({k: outputs.get(k, math.nan) for k in out_names})
        row["error"] = err
        rows.append(row)
    df = pd.DataFrame(rows, columns=[*points[0].keys(), *out_names, "error"])

    if out_dir is not None:
        target = Path(out_dir) / name if name else Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        df.to_csv(target / "sweep.csv", index=False)
        with open(target / "scenario.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(base_cfg, f, sort_keys=False)
        write_provenance(target, extra={"axes": axes, "workers": workers})
    return df


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _ngspice_version() -> str | None:
    try:
        res = subprocess.run(["ngspice", "-v"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [l.strip(" *") for l in (res.stdout + res.stderr).splitlines()]
    return next((l for l in lines if "ngspice" in l.lower() and any(c.isdigit() for c in l)), None)


def write_provenance(out_dir: str | Path, extra: dict | None = None) -> Path:
    """Write out_dir/provenance.json: git commit + dirty flag, UTC time, Python/package/ngspice versions."""
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    status = _git("status", "--porcelain")
    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": versions,
        "ngspice": _ngspice_version(),
    }
    if extra:
        prov["extra"] = extra
    path = Path(out_dir) / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    return path
