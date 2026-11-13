"""Schematic -> SPICE netlist, and ngspice deck assembly (task 4) [ngspice manual v47].

`export_spice` calls ``kicad-cli sch export netlist --format spice``. `build_deck` keeps every
component of the base netlist and adds, between the two `coil_nodes` (a, b):

    a --LMSC-- msc_1 --VMSCS(0 V sense)-- msc_2 --R(T)-- msc_3 --VMSCE(EMF)-- b

* coil current i_c = i(VMSCS), positive from a through the coil to b;
* R(T): behavioural source V = i_c R0 (1 + alpha (T - T_ref)) with T = v(msc_t) [K], or a plain
  resistor R(T0) when no thermal node is requested (C_th None);
* adiabatic thermal node: BMSCH injects i_c^2 R(T) [W] into CMSCT = C_th = m c_p [J/K]; node voltage
  = temperature in K, initial T0 (element ic + `.ic`), leak 1e12 ohm to a T0 source (time constant
  1e12 C_th s, i.e. adiabatic for any pulse);
* VMSCE is a PWL source carrying u_c(t) = d(lambda_ext)/dt [V] (the magnet's flux-linkage rate in
  the direction of positive coil current), so v(a) - v(b) = L di_c/dt + R(T) i_c + u_c;
* eddy loop k (0-based): LMSE{k} mse{k}_a->mse{k}_b, RMSE{k} ->mse{k}_c, VMSE{k} mse{k}_c->mse{k}_a
  carrying u_k(t) (loop current i(VMSE{k}), positive in the loop's own orientation), floating,
  referenced to ground through RMSEG{k} = 1 GOhm;
* couplings KMS{i}_{j} from the inductance matrix (index 0 = coil, 1..n = loops), after checking
  |k| < 1 and that the full matrix is positive definite.

Every generated element name starts with "<type letter>MS" and every generated node with "ms", so
they cannot collide with KiCad names (checked).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml

from magsim.circuit.ngspice import OUTPUT_FILE, control_block

T_REF = 293.15
DEFAULT_OPTIONS = {"method": "gear", "reltol": 1e-4}
LEAK_OHM = 1e12
EDDY_GROUND_OHM = 1e9

__all__ = ["build_deck", "coil_vectors", "export_spice", "load_circuit", "pwl"]


def export_spice(sch_path, out_path, kicad_cli: str = "kicad-cli", timeout: float = 120.0) -> Path:
    """Export a KiCad schematic to a SPICE netlist with ``kicad-cli sch export netlist --format spice``."""
    exe = shutil.which(kicad_cli)
    if exe is None:
        raise RuntimeError(f"{kicad_cli!r} not found on PATH")
    sch, out = Path(sch_path).resolve(), Path(out_path).resolve()
    if not sch.exists():
        raise FileNotFoundError(sch)
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([exe, "sch", "export", "netlist", "--format", "spice", "-o", str(out), str(sch)],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"kicad-cli failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}")
    return out


def coil_vectors(n_eddy: int = 0, thermal: bool = True) -> dict:
    """Standard vector names of the generated subcircuit (lower case, as ngspice reports them)."""
    v = {"i_coil": "i(vmscs)", "u_coil": "v(msc_3)-v(msc_2)"}
    if thermal:
        v["T_coil"] = "v(msc_t)"
    v["i_eddy"] = [f"i(vmse{k})" for k in range(n_eddy)]
    return v


def _num(x) -> str:
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"non-finite value {x!r} in deck")
    return f"{x:.12g}"


def pwl(t, v, per_line: int = 6) -> str:
    """SPICE PWL(...) text with '+' continuation lines; t [s] strictly increasing, v [V]."""
    t, v = np.asarray(t, float).ravel(), np.asarray(v, float).ravel()
    if len(t) != len(v) or len(t) < 1:
        raise ValueError("PWL needs equal-length, non-empty t and v")
    if np.any(np.diff(t) <= 0):
        raise ValueError("PWL times must be strictly increasing")
    pairs = [f"{_num(a)} {_num(b)}" for a, b in zip(t, v)]
    chunks = [" ".join(pairs[i : i + per_line]) for i in range(0, len(pairs), per_line)]
    return "PWL(" + "\n+ ".join(chunks) + ")"


def _coil_dict(coil) -> dict:
    c = dict(coil) if isinstance(coil, dict) else {k: getattr(coil, k) for k in ("L", "R0", "alpha", "T_ref", "T0", "C_th") if hasattr(coil, k)}
    if "L" not in c or "R0" not in c:
        raise ValueError("coil needs at least L [H] and R0 [ohm]")
    c.setdefault("alpha", 0.0)
    c.setdefault("T_ref", T_REF)
    c.setdefault("T0", c["T_ref"])
    c.setdefault("C_th", None)
    if not c["L"] > 0 or not c["R0"] > 0:
        raise ValueError("coil L and R0 must be > 0")
    if c["C_th"] is not None and not c["C_th"] > 0:
        raise ValueError("C_th must be > 0 (or None for no thermal node)")
    return c


def _split_base(text: str):
    """(title, body lines) of a SPICE netlist: first line is the title (SPICE rule); .end, .title and
    `.control` blocks are rejected/stripped; continuation lines are kept with their element."""
    lines = text.replace("\r\n", "\n").split("\n")
    title = lines[0].strip()
    if title.lower().startswith(".title"):
        title = title[6:].strip()
    elif title.startswith("*"):
        title = title.lstrip("*").strip()
    body = []
    for ln in lines[1:]:
        s = ln.strip().lower()
        if s == ".end" or s.startswith(".title"):
            continue
        if s.startswith(".control"):
            raise ValueError("base netlist contains a .control block; remove it (build_deck writes its own)")
        body.append(ln.rstrip())
    while body and not body[-1].strip():
        body.pop()
    return title, body


def _element_names(lines) -> set:
    names = set()
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith(("*", ".", "+")):
            names.add(s.split()[0].lower())
    return names


def _nodes_present(lines, nodes) -> set:
    toks = set()
    for ln in lines:
        s = ln.strip()
        if s and not s.startswith(("*", ".")):
            toks.update(t.lower() for t in s.split()[1:])
    return {n for n in nodes if n.lower() in toks or n == "0"}


def build_deck(
    base_netlist: str,
    coil_nodes,
    coil,
    eddy=None,
    emf_pwl=None,
    extra_lines=(),
    models=(),
    tran=None,
    ic=None,
    options=None,
    vectors=None,
    save=(),
    remove_refs=(),
    base_dir=None,
    title=None,
) -> str:
    """Assemble an ngspice deck (text) around `base_netlist`.

    coil_nodes : (a, b) node names; positive coil current flows a -> coil -> b
    coil : dict (or object with attributes) L [H], R0 [ohm] at T_ref [K] (default 293.15), alpha [1/K]
        (default 0), T0 [K] initial temperature (default T_ref), C_th [J/K] or None (no thermal node;
        R fixed at R0 (1 + alpha (T0 - T_ref))). `RoundWireCoil.lumped()` returns this dict.
    eddy : None or {"L": (n+1, n+1) inductance matrix [H] with index 0 = coil, "R": (n,) [ohm]}
    emf_pwl : {"coil" | "eddy{k}": (t [s], u [V])} series EMF sources (default 0 V)
    extra_lines : SPICE lines appended verbatim (bank, ESR, stimulus, ...)
    models : model files; `.include` with resolved absolute paths. A base `.include` of a file with
        the same basename is replaced by the given file; other relative includes resolve against
        `base_dir`.
    tran : {"tstop": s, "tstep": s, "tmax": s | None, "tstart": 0, "uic": False}
    ic : {node: V} -> `.ic V(node)=...` (the thermal node's T0 is added automatically)
    options : `.options` entries merged over {method: gear, reltol: 1e-4}
    vectors : wrdata vectors -> `.control` block writing `magsim_wrdata.txt` (see `run_ngspice`)
    save : extra `save` entries (e.g. "@d1[id]"); remove_refs : base elements to drop (e.g. a coil
        placeholder)
    """
    a, b = coil_nodes
    c = _coil_dict(coil)
    btitle, body = _split_base(base_netlist)

    # base includes
    model_paths = [Path(m).expanduser().resolve() for m in models]
    for m in model_paths:
        if not m.exists():
            raise FileNotFoundError(f"model file {m} not found")
    given = {m.name.lower() for m in model_paths}
    kept = []
    for ln in body:
        mm = re.match(r'^\s*\.(include|inc|lib)\s+"?([^"]+?)"?\s*$', ln, re.IGNORECASE)
        if mm:
            p = Path(mm.group(2))
            if p.name.lower() in given:
                continue
            if not p.is_absolute():
                if base_dir is None:
                    raise ValueError(f"relative include {ln.strip()!r} needs base_dir or a matching `models` entry")
                p = (Path(base_dir) / p).resolve()
            kept.append(f'.{mm.group(1).lower()} "{p}"')
            continue
        kept.append(ln)
    body = kept

    # drop elements on request (with their continuation lines)
    rm = {r.lower() for r in remove_refs}
    if rm:
        out, skip = [], False
        for ln in body:
            s = ln.strip()
            if s.startswith("+"):
                if not skip:
                    out.append(ln)
                continue
            skip = bool(s) and not s.startswith(("*", ".")) and s.split()[0].lower() in rm
            if not skip:
                out.append(ln)
        missing = rm - _element_names(body)
        if missing:
            raise ValueError(f"remove_refs not found in base netlist: {sorted(missing)}")
        body = out

    extra = [str(x) for x in extra_lines]
    names = _element_names(body + extra)
    clash = sorted(n for n in names if len(n) > 2 and n[1:3] == "ms")
    if clash:
        raise ValueError(f"element names reserved for magsim (second/third letters 'MS'): {clash}")
    present = _nodes_present(body + extra, (a, b))
    if len(present) != 2 and a != b:
        raise ValueError(f"coil node(s) {sorted({a, b} - present)} not found in the netlist")
    if a == b:
        raise ValueError("coil nodes must differ")
    for ln in body + extra:
        s = ln.strip().lower()
        if tran is not None and s.startswith(".tran"):
            raise ValueError("`.tran` given both in the netlist/extra lines and as `tran`")

    emf = dict(emf_pwl or {})
    n_e = 0
    if eddy is not None:
        Lm = np.asarray(eddy["L"], float)
        Re = np.asarray(eddy["R"], float).ravel()
        n_e = len(Re)
        if Lm.shape != (n_e + 1, n_e + 1):
            raise ValueError(f"eddy L must be ({n_e + 1}, {n_e + 1}) with index 0 = coil, got {Lm.shape}")
        if not np.allclose(Lm, Lm.T, rtol=1e-10, atol=0):
            raise ValueError("eddy L matrix is not symmetric")
        if abs(Lm[0, 0] - c["L"]) > 1e-9 * c["L"]:
            raise ValueError(f"eddy L[0,0] = {Lm[0, 0]:.6g} H differs from coil L = {c['L']:.6g} H")
        if np.any(Re <= 0):
            raise ValueError("eddy loop resistances must be > 0")
        try:
            np.linalg.cholesky(Lm)
        except np.linalg.LinAlgError as e:
            raise ValueError("inductance matrix (coil + eddy loops) is not positive definite") from e
        dg = np.sqrt(np.diag(Lm))
        K = Lm / np.outer(dg, dg)
        off = K[~np.eye(n_e + 1, dtype=bool)]
        if np.any(np.abs(off) >= 1.0):
            raise ValueError(f"coupling coefficient |k| >= 1 (max {np.max(np.abs(off)):.6f})")
    unknown = set(emf) - {"coil"} - {f"eddy{k}" for k in range(n_e)}
    if unknown:
        raise ValueError(f"emf_pwl keys not in the deck: {sorted(unknown)}")

    def src(key):
        return pwl(*emf[key]) if key in emf else "0"

    thermal = c["C_th"] is not None
    out = [f".title {title or ('magsim deck: ' + btitle)}",
           "* generated by magsim.circuit.netlist.build_deck (SI units)"]
    out += [f'.include "{m}"' for m in model_paths]
    out += ["* ---- base netlist", *body]
    out.append(f"* ---- magsim coil between {a} and {b} (positive current {a} -> {b})")
    out.append(f"LMSC {a} msc_1 {_num(c['L'])}")
    out.append("VMSCS msc_1 msc_2 0")
    rexpr = f"{_num(c['R0'])}*(1+({_num(c['alpha'])})*(V(msc_t)-{_num(c['T_ref'])}))"
    if thermal:
        out.append(f"BMSCR msc_2 msc_3 V=i(VMSCS)*{rexpr}")
    else:
        out.append(f"RMSC msc_2 msc_3 {_num(c['R0'] * (1 + c['alpha'] * (c['T0'] - c['T_ref'])))}")
    out.append(f"VMSCE msc_3 {b} {src('coil')}")
    if thermal:
        out += [
            "* adiabatic thermal node: v(msc_t) = coil temperature [K]",
            f"BMSCH 0 msc_t I=i(VMSCS)*i(VMSCS)*{rexpr}",
            f"CMSCT msc_t 0 {_num(c['C_th'])} ic={_num(c['T0'])}",
            f"RMSCT msc_t msc_tamb {_num(LEAK_OHM)}",
            f"VMSCT msc_tamb 0 {_num(c['T0'])}",
        ]
    if eddy is not None:
        out.append(f"* ---- {n_e} eddy loops (floating, {EDDY_GROUND_OHM:.0e} ohm to ground)")
        for k in range(n_e):
            out += [
                f"LMSE{k} mse{k}_a mse{k}_b {_num(Lm[k + 1, k + 1])}",
                f"RMSE{k} mse{k}_b mse{k}_c {_num(Re[k])}",
                f"VMSE{k} mse{k}_c mse{k}_a {src(f'eddy{k}')}",
                f"RMSEG{k} mse{k}_a 0 {_num(EDDY_GROUND_OHM)}",
            ]
        lname = ["LMSC"] + [f"LMSE{k}" for k in range(n_e)]
        for i in range(n_e + 1):
            for j in range(i + 1, n_e + 1):
                if abs(K[i, j]) > 1e-12:
                    out.append(f"KMS{i}_{j} {lname[i]} {lname[j]} {_num(K[i, j])}")
    if extra:
        out += ["* ---- extra lines", *extra]
    if tran is not None:
        t = dict(tran)
        line = f".tran {_num(t['tstep'])} {_num(t['tstop'])}"
        if t.get("tmax") is not None or t.get("tstart"):
            line += f" {_num(t.get('tstart', 0.0))}"
            if t.get("tmax") is not None:
                line += f" {_num(t['tmax'])}"
        if t.get("uic"):
            line += " uic"
        out.append(line)
    ics = {}
    if thermal:
        ics["msc_t"] = c["T0"]
    for node, val in (ic or {}).items():
        mm = re.match(r"^\s*v\((.+)\)\s*$", str(node), re.IGNORECASE)
        ics[mm.group(1) if mm else str(node)] = val
    if ics:
        out.append(".ic " + " ".join(f"V({n})={_num(v)}" for n, v in ics.items()))
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    out.append(".options " + " ".join(k if v is True else f"{k}={v}" for k, v in opts.items() if v is not False))
    if vectors:
        out += control_block(vectors, save, OUTPUT_FILE)
    out.append(".end")
    return "\n".join(out) + "\n"


def load_circuit(folder) -> dict:
    """Load a circuit snapshot folder (e.g. `res_exp/circuits/magnetiser/`): `extra.yaml` plus files.

    Returns a dict with `base_netlist` (text), `models` (absolute paths), `coil_nodes`, `extra_lines`,
    `tran`, `ic`, `options`, `vectors`, `save` and the remaining YAML keys unchanged.
    """
    folder = Path(folder).resolve()
    with open(folder / "extra.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = dict(cfg)
    out["base_netlist"] = (folder / cfg["netlist"]).read_text(encoding="utf-8")
    out["models"] = [str((folder / m).resolve()) for m in cfg.get("models", [])]
    out["coil_nodes"] = tuple(cfg["coil_nodes"])
    out.setdefault("extra_lines", [])
    out.setdefault("save", [])
    out["folder"] = str(folder)
    return out
