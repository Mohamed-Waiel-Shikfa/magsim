"""ngspice batch runner and `wrdata` parser (task 4) [ngspice manual v47].

`run_ngspice` writes the deck into `workdir`, runs ``ngspice -b`` there (so the deck's relative
`wrdata` file lands in `workdir`), checks the log for failures and returns the requested vectors as
numpy arrays keyed by the names given (plus "time"). Repeated time points (ngspice writes breakpoints
twice) are collapsed to the last value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_FILE = "magsim_wrdata.txt"
DECK_FILE = "magsim_deck.cir"
LOG_FILE = "magsim_ngspice.log"

# log lines that mean the run cannot be trusted
_FAIL = re.compile(
    r"timestep too small|singular matrix|simulation\(s\) aborted|\baborted\b|\bfatal\b|^\s*error\b|error:|"
    r"no such vector|could not find|unknown (?:subckt|model|device)|gmin stepping failed|source stepping failed",
    re.IGNORECASE | re.MULTILINE,
)

__all__ = ["NgspiceError", "control_block", "ngspice_available", "parse_wrdata", "run_ngspice"]


class NgspiceError(RuntimeError):
    """ngspice failed; the message carries the log tail."""


def ngspice_available(exe: str = "ngspice") -> bool:
    return shutil.which(exe) is not None


def control_block(vectors, save=(), output_file=OUTPUT_FILE) -> list[str]:
    """`.control` block: single time column with vector names, optional `save`, run, wrdata."""
    vectors = list(vectors)
    if not vectors:
        raise ValueError("need at least one vector to write")
    extra = list(save) + [v for v in vectors if v.startswith("@") and v not in save]
    lines = [".control", "set wr_singlescale", "set wr_vecnames", "option numdgt=12"]
    if extra:
        lines.append("save all " + " ".join(extra))
    lines += ["run", f"wrdata {output_file} " + " ".join(vectors), ".endc"]
    return lines


def parse_wrdata(path, vectors=None) -> dict[str, np.ndarray]:
    """Parse an ngspice `wrdata` file.

    Handles both layouts: `wr_singlescale` (time, v1, v2, ...) and the default time/value column
    pairs (time, v1, time, v2, ...). Columns map to `vectors` by position (names from the header if
    `vectors` is None). Repeated time points keep the last row.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    tokens = first.split()
    has_header = bool(tokens) and not _is_number(tokens[0])
    data = np.loadtxt(path, skiprows=1 if has_header else 0, ndmin=2)
    ncol = data.shape[1]
    if vectors is None:
        if not has_header:
            raise ValueError(f"{path}: no header; pass the vector names")
        names = tokens[1:]
        if len(names) >= 2 and all(s == tokens[0] for s in names[1::2]):  # pairs: time v1 time v2 ...
            names = names[0::2]
        vectors = names
    vectors = list(vectors)
    n = len(vectors)
    if ncol == n + 1:
        t, cols = data[:, 0], [data[:, 1 + i] for i in range(n)]
    elif ncol == 2 * n:
        t, cols = data[:, 0], [data[:, 2 * i + 1] for i in range(n)]
    else:
        raise ValueError(f"{path}: {ncol} columns do not match {n} vectors (single-scale or pairs layout)")
    keep = np.r_[np.diff(t) > 0, True]
    out = {"time": t[keep]}
    for name, c in zip(vectors, cols):
        out[name] = c[keep]
    return out


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def run_ngspice(deck: str, workdir, vectors=None, timeout: float = 600.0, exe: str = "ngspice",
                save=()) -> dict[str, np.ndarray]:
    """Run `deck` in batch mode in `workdir` and return {"time": t, vector: values}.

    If the deck has no `.control` block, one is appended (needs `vectors`). With a `.control` block
    (e.g. from `build_deck(..., vectors=...)`), `vectors` only names the wrdata columns (positional);
    None takes the names from the wrdata header. Raises NgspiceError with the log tail on a non-zero
    exit, a timeout, any failure message (e.g. "timestep too small") or a missing output file.
    """
    exe_path = shutil.which(exe)
    if exe_path is None:
        raise NgspiceError(f"{exe!r} not found on PATH")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if not re.search(r"^\s*\.control\b", deck, re.IGNORECASE | re.MULTILINE):
        if not vectors:
            raise ValueError("deck has no .control block: pass `vectors`")
        lines = [ln for ln in deck.rstrip().splitlines()]
        while lines and lines[-1].strip().lower() in ("", ".end"):
            lines.pop()
        deck = "\n".join(lines + control_block(vectors, save) + [".end"]) + "\n"
    out = workdir / OUTPUT_FILE
    if out.exists():
        out.unlink()
    (workdir / DECK_FILE).write_text(deck, encoding="utf-8")
    try:
        r = subprocess.run([exe_path, "-b", DECK_FILE], cwd=workdir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise NgspiceError(f"ngspice timed out after {timeout} s in {workdir}") from e
    log = (r.stdout or "") + (r.stderr or "")
    (workdir / LOG_FILE).write_text(log, encoding="utf-8")
    tail = "\n".join(log.splitlines()[-40:])
    m = _FAIL.search(log)
    if r.returncode != 0 or m is not None:
        why = f"exit code {r.returncode}" if r.returncode else f"log reports {m.group(0)!r}"
        raise NgspiceError(f"ngspice failed ({why}) in {workdir}; log tail:\n{tail}")
    if not out.exists() or out.stat().st_size == 0:
        raise NgspiceError(f"ngspice wrote no output ({out}); log tail:\n{tail}")
    return parse_wrdata(out, vectors)
