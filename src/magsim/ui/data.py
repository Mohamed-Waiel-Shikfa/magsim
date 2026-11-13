"""Data helpers for the UI (no Streamlit imports): Hall traces, result-folder scans, environment check."""

from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from magsim.ui import RESULTS, ROOT

OUTPUT_SUFFIXES = {".png", ".csv", ".json", ".md"}


# ----------------------------------------------------------------------------------------------- Hall traces
def read_hall(path: Path) -> tuple[list[str], pd.DataFrame]:
    """Header lines (without '#') and the samples (t_s [s], adc, field_mT [mT]) of a Hall .csv(.gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    header: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                break
            header.append(line[1:].strip())
    try:
        df = pd.read_csv(path, comment="#", dtype={"t_s": float, "adc": float, "field_mT": float})
    except pd.errors.EmptyDataError:
        df = pd.DataFrame({"t_s": [], "adc": [], "field_mT": []})
    return header, df


def hall_stats(header: list[str], df: pd.DataFrame) -> dict:
    """n samples, duration [s], sample rate [Hz] (header `fs=` if given, else from the median step),
    peak / min field [mT]."""
    n = len(df)
    fs = None
    for h in header:
        if m := re.search(r"fs\s*=\s*([0-9.eE+-]+)\s*Hz", h):
            fs = float(m.group(1))
    if n < 2:
        return {"n": n, "duration_s": 0.0, "fs_Hz": fs, "peak_mT": float(df["field_mT"].max()) if n else None,
                "min_mT": float(df["field_mT"].min()) if n else None}
    t = df["t_s"].to_numpy()
    if fs is None:
        dt = float(np.median(np.diff(t)))
        fs = 1.0 / dt if dt > 0 else None
    return {"n": n, "duration_s": float(t[-1] - t[0]), "fs_Hz": fs,
            "peak_mT": float(df["field_mT"].max()), "min_mT": float(df["field_mT"].min())}


def envelope(t: np.ndarray, y: np.ndarray, bins: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max envelope of y over at most `bins` equal-count bins: (t_mid, y_min, y_max).
    With n <= bins every sample is its own bin (no decimation)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return t, y, y
    k = min(int(bins), n)
    starts = np.linspace(0, n, k + 1).astype(int)[:-1]
    starts = np.unique(starts)
    lo = np.minimum.reduceat(y, starts)
    hi = np.maximum.reduceat(y, starts)
    ends = np.append(starts[1:], n) - 1
    tm = 0.5 * (t[starts] + t[ends])
    return tm, lo, hi


# ----------------------------------------------------------------------------------------------- result folders
def output_folders(root: Path = RESULTS, exclude: tuple[str, ...] = ("ui",)) -> list[Path]:
    """Folders under `root` holding outputs (PNG / CSV / JSON / MD) directly or in a `figures/` subfolder.
    Top-level folders named in `exclude` and folders named `figures` are skipped. res_sim/paper first."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        parts = d.relative_to(root).parts
        if parts[0] in exclude or d.name == "figures" or any(p.startswith(".") for p in parts):
            continue
        files = [f for f in d.iterdir() if f.is_file() and f.suffix in OUTPUT_SUFFIXES]
        if files or any((d / "figures").glob("*.png")):
            found.append(d)
    return sorted(found, key=lambda p: (p.relative_to(root).parts[0] != "paper", str(p)))


def figures(folder: Path) -> list[Path]:
    """Every PNG under `folder` (recursive), sorted by subfolder then name."""
    return sorted(Path(folder).rglob("*.png"), key=lambda p: (len(p.relative_to(folder).parts), str(p)))


# ----------------------------------------------------------------------------------------------- environment
def _first_line(cmd: list[str], pattern: str | None = None) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (r.stdout + r.stderr).strip()
    if pattern:
        m = re.search(pattern, text)
        return m.group(0) if m else (text.splitlines() or [None])[0]
    return (text.splitlines() or [None])[0]


def environment() -> dict:
    """Paths and versions of ngspice and kicad-cli, the magsim version and the git commit of ROOT."""
    env = {}
    for tool, args, pat in (("ngspice", ["--version"], r"ngspice-\S+"), ("kicad-cli", ["version"], None)):
        path = shutil.which(tool)
        env[tool] = {"path": path, "version": _first_line([path, *args], pat) if path else None}
    try:
        env["magsim"] = metadata.version("magsim")
    except metadata.PackageNotFoundError:
        env["magsim"] = None
    commit = _first_line(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"])
    dirty = _first_line(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"])
    env["git"] = None if not commit else commit + (" (uncommitted changes)" if dirty else "")
    return env
