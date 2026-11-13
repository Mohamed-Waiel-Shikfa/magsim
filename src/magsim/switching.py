"""Switching simulator: coupled pulse-circuit / magnet transient by waveform relaxation (task 6).

Method (docs/SPEC.md "Switching simulator")
-------------------------------------------
* Circuit: any SPICE netlist (a board snapshot via `magsim.circuit.load_circuit`, or a `.cir` file) with
  the coil subcircuit of `magsim.circuit.build_deck` between the coil nodes: coil L from geometry, R(T)
  with an adiabatic thermal node [matula1979], the eddy-loop network of the magnet (R, L, K couplings)
  and one PWL EMF source per circuit [ngspice].
* Magnet: the volume-integral MoM of `magsim.mom` with the scalar Preisach law [mayergoyz1991],
  driven quasi-statically by H_app(x, t) = h_c(x) i_c(t) + sum_k h_k(x) i_k(t) (h in A/m per A,
  cell-averaged over the solver's Gauss points). Preisach is rate-independent, so the magnet is
  "marched" along the field path: committed at every sample of a path grid whose field change between
  samples is at most `field_tol` x the peak generated field (max over cells of |dH_app|), plus every
  local extremum of the coil and eddy currents, so each committed segment is monotone per cell.
* Back-EMF by reciprocity: lambda_c(t) = sum_cells V J(t) . h_c [Wb]; the PWL source of circuit c carries
  u_c = d lambda_c / dt [V], a voltage DROP in the direction of positive current (as L di/dt), so a
  magnet polarising along the coil field adds flux linkage like extra inductance (tested, gate S4).
* Coupling: Gauss-Jacobi waveform relaxation [lelarasmee1982]: circuit run with the EMFs of the
  previous magnet march (first: zero), then the magnet march on the new currents, until
  max|i_c^m - i_c^(m-1)| / max|i_c^m| < tol on a common time grid (default 1e-4, <= 15 iterations).
  Optional under-relaxation of the EMF (`coupling.relax`, default 1 = none).
* Field definitions (s_hat = target switching direction): H_gen = h_c i_c, H_app = H_gen + H_eddy,
  H_felt = Solution.H (total H in the material, = H_app + H_demag), B_in = MU0 H_felt + J. Each is
  reported as the maximum over time of s_hat . H at the centroid cell(s), of the cell-volume-weighted
  mean, and the minimum over cells of each cell's maximum (the weakest cell).
* Switched fraction <P> = <J_irr . s_hat> / Js_irr in [-1, 1] (Preisach state after the pulse,
  open circuit); relative remanence = final open-circuit <J . s_hat> / the value after an ideal
  saturating pulse (uniform 5 Hcj along s_hat, ramped up and down in 50 + 50 committed steps) on the
  same mesh from the same initial state.
* Energy audit: E0 = C V0^2 / 2; coil-port energy int v_port i_c dt = E_R(coil) + E_eddy + W_magnet +
  dE_L + residual (gate S3: |residual| < 0.5 % E0); W_magnet two ways: sum_c int i_c u_c dt (circuit
  side) and sum_cells V int H_app . dJ (magnet side); circuit-side losses = E0 - E_C(end) - E_port.

Scenario format (YAML, SI units; relative paths are relative to the magsim project root)
-------------------------------------------------------------------------------------
name: <str>
circuit:            # either a board snapshot ...
  board: res_exp/circuits/magnetiser            # folder read by load_circuit (extra.yaml)
  # ... or a plain netlist:
  netlist: scenarios/circuits/cap_discharge.cir
  # ... or a KiCad schematic, exported with kicad-cli on every run (parts marked "exclude from
  # simulation" are left out; add them with extra_lines):
  schematic: /path/to/board.kicad_sch
  coil_nodes: [a, b]                         # positive coil current a -> coil -> b
  models: [<lib>, ...]                       # replaces the board's models (its .include is dropped)
  extra_lines: [<SPICE line>, ...]           # replaces the board's extra lines
  extra_lines_add: [<SPICE line>, ...]       # appended
  element_values: {RESR1: 0.07, RG1: 470}    # rewrite the value of 2-terminal R/L/C elements [SI]
  ic: {cap_int: 450.0}                       # merged over the board's .ic [V]
  tran: {tstep, tstop, tmax, uic}            # merged over the board's .tran [s]
  options: {reltol: 1e-4}                    # merged over the board's .options
  bank: {element: C1, node: cap_int}         # bank capacitor (E0 = C V0^2 / 2, V0 = ic[node])
  scr: {Q1: "i(v.xq1.vsa)"}                  # thyristor current vectors for the stress report
  scr_ratings: {I2t_A2s: 6050, didt_A_per_s: 1.5e8}   # CLA100E1200HB p.2 (default)
  fire_time: 10.0e-6                         # [s], t_pk is reported relative to it
coil:
  reference: r1 | r2 | r3                    # magsim.coil.reference_coil_r*; overrides: {kwargs}
  type: foil | round_wire                    # or explicit FoilCoil / RoundWireCoil params: {...}
  pose: {position: [x, y, z], rotation: {...}}
  temperature: 293.15                        # initial coil temperature [K] (temperature_C allowed)
magnet: {name, shape: {type: cuboid, dims} | {type: cylinder, diameter, height} | {type: ring, d_inner,
         d_outer, height}, material: {grade, overrides} | {type: linear, Jr, mu_rec, mu_perp},
         pose, easy_axis, mesh, initial, temperature}
target_direction: [0, 0, 1]                  # global; default +easy axis
eddy: {enabled: true, n_r: 4, n_z: 2, resistivity: null}
      # resistivity default: the grade's rho_e (perpendicular to the easy axis) when the coil axis is
      # parallel to the easy axis (loops in the plane normal to it), else (rho_e + rho_e_par) / 2.
coupling: {tol: 1e-4, max_iter: 15, field_tol: 2.5e-3, grid_tol: 2e-3, n_background: 400, relax: 1.0}
outputs: {dir: res_sim/<name>, figures: true}
sweep: {axes: [{path, values | linspace | logspace}], workers: 4, plots: [...], heatmaps: [...]}
validation: {meshes: [[nx, ny, nz], ...], initial_guess_scale: 0.5}
"""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from magsim.circuit import build_deck, coil_vectors, load_circuit, run_ngspice
from magsim.circuit.ngspice import NgspiceError
from magsim.circuit.netlist import export_spice
from magsim.coil import FoilCoil, RoundWireCoil, reference_coil_r1, reference_coil_r2, reference_coil_r3
from magsim.config import load_scenario
from magsim.constants import MU0
from magsim.eddy import eddy_loops
from magsim.geometry import Cuboid, Cylinder, Magnet, Pose, Ring
from magsim.materials.base import LinearMaterial
from magsim.materials.library import get_material, grade_info
from magsim.materials.preisach import PreisachMaterial
from magsim.mom import MagnetSystem

__all__ = [
    "ScenarioError",
    "CLA100E1200HB_RATINGS",
    "prepare",
    "run_switching",
    "linear_fold_in",
    "sweep_point",
    "write_run",
    "cli_run",
    "cli_sweep",
    "cli_validate",
]

ROOT = Path(__file__).resolve().parents[2]

# Littelfuse/IXYS CLA100E1200HB, rev. 20210601d, p.2 (also `magnet_pulse_sim.py:31-59`): I^2t for fusing
# (10 ms half sine, TVJ = 45 degC) and the critical rate of rise of current (repetitive, IT = 300 A).
# These are 10 ms / repetitive ratings; a ~100 us kA pulse is outside their conditions (reported as ratios).
CLA100E1200HB_RATINGS = {"I2t_A2s": 6050.0, "didt_A_per_s": 150e6}

DEFAULT_COUPLING = {"tol": 1e-4, "max_iter": 15, "field_tol": 2.5e-3, "grid_tol": None, "n_background": 400,
                    "relax": 1.0, "max_grid": 12000, "extrema_rel": 1e-3, "freeze_tol": 1e-3, "split": "auto",
                    "eddy_grid_tol": 1e-2}
DEFAULT_EDDY = {"enabled": True, "n_r": 4, "n_z": 2, "resistivity": None, "n_q": 4}
_TOP_KEYS = {"name", "description", "circuit", "coil", "magnet", "target_direction", "eddy", "coupling",
             "outputs", "sweep", "validation"}
_CIRCUIT_KEYS = {"board", "netlist", "schematic", "coil_nodes", "models", "extra_lines", "extra_lines_add", "element_values",
                 "ic", "tran", "options", "bank", "scr", "scr_ratings", "fire_time", "save"}
_SPICE_SUFFIX = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "mil": 25.4e-6, "u": 1e-6, "n": 1e-9,
                 "p": 1e-12, "f": 1e-15}


class ScenarioError(ValueError):
    """Invalid switching scenario (message names the offending key)."""


# =====================================================================================================
# Scenario parsing
# =====================================================================================================
def _resolve(p) -> Path:
    """Absolute path; relative paths resolve against the project root, then the working directory."""
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        return p
    for base in (ROOT, Path.cwd()):
        if (base / p).exists():
            return (base / p).resolve()
    return (ROOT / p).resolve()


def spice_value(s) -> float:
    """SPICE number with scale suffix ('780u', '2.2k', '1meg', '70m', '1e-6') -> float."""
    if isinstance(s, (int, float)):
        return float(s)
    m = re.match(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)", str(s))
    if not m:
        raise ScenarioError(f"cannot parse SPICE value {s!r}")
    v, suf = float(m.group(1)), m.group(2).lower()
    for key in ("meg", "mil", "t", "g", "k", "m", "u", "n", "p", "f"):
        if suf.startswith(key):
            return v * _SPICE_SUFFIX[key]
    return v


def build_shape(d):
    """Cuboid / Cylinder / Ring from {type: cuboid, dims: [a,b,c]} | {type: cylinder, diameter, height} |
    {type: ring, d_inner, d_outer, height} [m]."""
    if not isinstance(d, dict) or "type" not in d:
        raise ScenarioError("magnet.shape needs a 'type' (cuboid | cylinder | ring)")
    t = d["type"]
    try:
        if t == "cuboid":
            return Cuboid(tuple(float(x) for x in d["dims"]))
        if t == "cylinder":
            return Cylinder(float(d["diameter"]), float(d["height"]))
        if t == "ring":
            return Ring(float(d["d_inner"]), float(d["d_outer"]), float(d["height"]))
    except KeyError as e:
        raise ScenarioError(f"magnet.shape ({t}) is missing {e.args[0]!r}") from e
    raise ScenarioError(f"magnet.shape.type must be cuboid | cylinder | ring, got {t!r}")


def build_material(d):
    """(material, grade name or None) from {grade, overrides} or {type: linear, Jr, mu_rec, mu_perp}."""
    if not isinstance(d, dict):
        raise ScenarioError("magnet.material must be a mapping")
    if "grade" in d:
        return get_material(str(d["grade"]), **(d.get("overrides") or {})), str(d["grade"])
    if d.get("type") == "linear":
        return LinearMaterial(float(d.get("Jr", 0.0)), float(d.get("mu_rec", 1.0)), float(d.get("mu_perp", 1.0))), None
    raise ScenarioError("magnet.material needs 'grade' or type: linear")


def build_magnet(d) -> tuple[Magnet, str | None]:
    """Magnet from the shared magnet schema (force simulator schema, docs/SPEC.md)."""
    if not isinstance(d, dict):
        raise ScenarioError("'magnet' must be a mapping")
    for key in ("shape", "material"):
        if key not in d:
            raise ScenarioError(f"magnet.{key} is required")
    mat, grade = build_material(d["material"])
    m = Magnet(d.get("name", "magnet"), build_shape(d["shape"]), Pose.from_dict(d.get("pose")), mat,
               easy_axis=d.get("easy_axis", (0.0, 0.0, 1.0)), mesh=d.get("mesh"),
               initial=d.get("initial", "+sat"), temperature=float(d.get("temperature", 293.15)))
    return m, grade


def build_coil(d):
    """Coil from {reference: r1|r2|r3, overrides} or {type: foil|round_wire, params}, plus pose."""
    if not isinstance(d, dict):
        raise ScenarioError("'coil' must be a mapping")
    pose = Pose.from_dict(d.get("pose"))
    kw = dict(position=pose.position, rotation=pose.rotation)
    ref = d.get("reference")
    if ref is not None:
        factories = {"r1": reference_coil_r1, "r2": reference_coil_r2, "r3": reference_coil_r3}
        if str(ref).lower() not in factories:
            raise ScenarioError(f"coil.reference must be r1 | r2 | r3, got {ref!r}")
        kw.update(d.get("overrides") or {})
        return factories[str(ref).lower()](**kw)
    params = dict(d.get("params") or {})
    params.update(kw)
    if d.get("type") == "foil":
        return FoilCoil(**params)
    if d.get("type") == "round_wire":
        return RoundWireCoil(**params)
    raise ScenarioError("coil needs 'reference' (r1|r2|r3) or type: foil | round_wire with params")


def _strip_includes(text: str, names) -> str:
    names = {n.lower() for n in names}
    out = []
    for ln in text.splitlines():
        mm = re.match(r'^\s*\.(include|inc|lib)\s+"?([^"]+?)"?\s*$', ln, re.IGNORECASE)
        if mm and Path(mm.group(2)).name.lower() in names:
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


def _set_element_values(lines: list[str], values: dict) -> tuple[list[str], set]:
    """Replace the value (4th token) of R/L/C elements named in `values`; returns (lines, names done)."""
    done, out = set(), []
    want = {str(k).lower(): v for k, v in values.items()}
    for ln in lines:
        tok = ln.split()
        if tok and tok[0].lower() in want:
            if tok[0][0].lower() not in "rlc" or len(tok) < 4:
                raise ScenarioError(f"element_values: {tok[0]} is not a 2-terminal R/L/C element line")
            tok[3] = f"{float(want[tok[0].lower()]):.12g}"
            ln = " ".join(tok)
            done.add(tok[0].lower())
        out.append(ln)
    return out, done


def build_circuit(d) -> dict:
    """Resolve the `circuit` block into build_deck arguments plus bank / device information."""
    if not isinstance(d, dict):
        raise ScenarioError("'circuit' must be a mapping")
    unknown = set(d) - _CIRCUIT_KEYS
    if unknown:
        raise ScenarioError(f"unknown circuit key(s) {sorted(unknown)}; allowed: {sorted(_CIRCUIT_KEYS)}")
    if sum(k in d for k in ("board", "netlist", "schematic")) != 1:
        raise ScenarioError("circuit needs exactly one of 'board' (snapshot folder), 'netlist' (.cir file) "
                            "or 'schematic' (.kicad_sch, exported with kicad-cli)")
    if "board" in d:
        b = load_circuit(_resolve(d["board"]))
        c = {"base_netlist": b["base_netlist"], "coil_nodes": tuple(b["coil_nodes"]), "models": list(b["models"]),
             "extra_lines": list(b["extra_lines"]), "tran": dict(b.get("tran") or {}), "ic": dict(b.get("ic") or {}),
             "options": dict(b.get("options") or {}), "fire_time": b.get("fire_time", 0.0), "base_dir": b["folder"]}
        default_models = [Path(m).name for m in b["models"]]
    else:
        key = "netlist" if "netlist" in d else "schematic"
        path = _resolve(d[key])
        if not path.exists():
            raise ScenarioError(f"circuit.{key} {path} not found")
        if "coil_nodes" not in d:
            raise ScenarioError(f"circuit.coil_nodes is required with circuit.{key}")
        if key == "schematic":
            # fresh `kicad-cli sch export netlist --format spice` on every run, so schematic edits are picked up
            text = export_spice(path, Path(tempfile.mkdtemp(prefix="magsim_sch_")) / (path.stem + ".cir")).read_text(
                encoding="utf-8")
        else:
            text = path.read_text(encoding="utf-8")
        c = {"base_netlist": text, "coil_nodes": None, "models": [], "extra_lines": [],
             "tran": {}, "ic": {}, "options": {}, "fire_time": 0.0, "base_dir": str(path.parent)}
        # `models` replaces the netlist's own model includes (as for a board)
        default_models = [Path(m.group(1)).name for m in
                          re.finditer(r'^\s*\.(?:include|inc|lib)\s+"?([^"\n]+?)"?\s*$', text, re.I | re.M)]
    if "coil_nodes" in d:
        c["coil_nodes"] = tuple(str(x) for x in d["coil_nodes"])
    if "models" in d:
        c["models"] = [str(_resolve(m)) for m in d["models"]]
        for m in c["models"]:
            if not Path(m).exists():
                raise ScenarioError(f"circuit.models: {m} not found")
        c["base_netlist"] = _strip_includes(c["base_netlist"], default_models)
    if "extra_lines" in d:
        c["extra_lines"] = [str(x) for x in d["extra_lines"]]
    c["extra_lines"] += [str(x) for x in d.get("extra_lines_add", [])]
    if d.get("element_values"):
        base_lines, done1 = _set_element_values(c["base_netlist"].splitlines(), d["element_values"])
        c["extra_lines"], done2 = _set_element_values(c["extra_lines"], d["element_values"])
        missing = {str(k).lower() for k in d["element_values"]} - done1 - done2
        if missing:
            raise ScenarioError(f"circuit.element_values: element(s) {sorted(missing)} not found")
        c["base_netlist"] = "\n".join(base_lines) + "\n"
    c["ic"].update({str(k): float(v) for k, v in (d.get("ic") or {}).items()})
    c["tran"].update(d.get("tran") or {})
    if "tstop" not in c["tran"] or "tstep" not in c["tran"]:
        raise ScenarioError("circuit.tran needs tstep and tstop [s]")
    c["options"].update(d.get("options") or {})
    if "fire_time" in d:
        c["fire_time"] = float(d["fire_time"])
    c["save"] = list(d.get("save", []))
    bank = d.get("bank")
    if not bank or "element" not in bank or "node" not in bank:
        raise ScenarioError("circuit.bank: {element: <capacitor name>, node: <its non-ground node>} is required")
    C = None
    for ln in c["base_netlist"].splitlines() + c["extra_lines"]:
        tok = ln.split()
        if tok and tok[0].lower() == str(bank["element"]).lower():
            C = spice_value(tok[3])
    if C is None:
        raise ScenarioError(f"circuit.bank.element {bank['element']!r} not found in the netlist / extra lines")
    node = str(bank["node"])
    if node not in c["ic"]:
        raise ScenarioError(f"circuit.ic has no initial voltage for the bank node {node!r}")
    c["bank"] = {"element": str(bank["element"]), "node": node, "C": C, "V0": c["ic"][node]}
    c["scr"] = {str(k): str(v) for k, v in (d.get("scr") or {}).items()}
    c["scr_ratings"] = {**CLA100E1200HB_RATINGS, **(d.get("scr_ratings") or {})}
    if c["coil_nodes"] is None or len(c["coil_nodes"]) != 2:
        raise ScenarioError("circuit.coil_nodes must name two nodes")
    return c


def _merge(defaults, given, where):
    given = dict(given or {})
    unknown = set(given) - set(defaults)
    if unknown:
        raise ScenarioError(f"unknown {where} key(s) {sorted(unknown)}; allowed: {sorted(defaults)}")
    return {**defaults, **given}


# =====================================================================================================
# Setup: magnet system, coil, eddy loops, per-amp fields, circuit
# =====================================================================================================
@dataclass
class Setup:
    """Everything a coupled run needs (built once per scenario by `prepare`)."""

    cfg: dict
    name: str
    magnet: Magnet
    grade: str | None
    system: MagnetSystem
    coil: object
    loops: object
    s_hat: np.ndarray
    h_pts: np.ndarray  # (n_pts, n_circ, 3) A/m per A at the solver's field points
    hbar: np.ndarray  # (n_cells, n_circ, 3) cell-averaged
    volume: np.ndarray
    L: np.ndarray  # (n_circ, n_circ) H, index 0 = coil
    R_loops: np.ndarray
    lumped: dict
    circuit: dict
    coupling: dict
    eddy: dict
    centroid_cells: np.ndarray
    states_pre: list = field(default_factory=list)
    J_pre: np.ndarray = None
    sol_pre: object = None
    Hcj: float = float("nan")
    timings: dict = field(default_factory=dict)
    dL_ref: np.ndarray = None  # (n_circ, n_circ) linear reference response folded into SPICE [H]
    split_info: dict = field(default_factory=dict)

    @property
    def n_circ(self) -> int:
        return self.hbar.shape[1]

    @property
    def n_loops(self) -> int:
        return self.n_circ - 1

    def circuit_names(self) -> list[str]:
        return ["coil"] + [f"eddy{k}" for k in range(self.n_loops)]
    robust: bool = False  # set after a 'Timestep too small' retry; later circuit runs reuse the robust deck


def _eddy_network(magnet: Magnet, coil, ecfg: dict, grade: str | None):
    """Eddy-loop network of the magnet threaded by the coil field; resistivity default from the grade."""
    a_coil = coil.rotation.apply([0.0, 0.0, 1.0])
    R = magnet.pose.matrix
    a_body = R.T @ a_coil
    e_body = magnet.easy_axis
    rho = ecfg.get("resistivity")
    if rho is None:
        if grade is None:
            raise ScenarioError("eddy.resistivity is required for a non-library material (or set eddy.enabled: false)")
        gi = grade_info(grade)
        aligned = abs(abs(float(a_body @ e_body)) - 1.0) < 1e-6
        rho = gi["rho_e"] if aligned else 0.5 * (gi["rho_e"] + gi["rho_e_par"])
    sh = magnet.shape
    pose = (magnet.pose.position, magnet.pose.rotation)
    if isinstance(sh, Cuboid):
        axis = int(np.argmax(np.abs(a_body)))
        if abs(abs(a_body[axis]) - 1.0) > 1e-6:
            warnings.warn("coil axis is not along a magnet body axis: eddy loops use the closest body axis", stacklevel=2)
        loops = eddy_loops("cuboid", sh.dims, pose, n_r=int(ecfg["n_r"]), n_z=int(ecfg["n_z"]), resistivity=float(rho),
                           axis=axis)
    else:
        if abs(abs(a_body[2]) - 1.0) > 1e-6:
            raise ScenarioError("eddy loops in a cylinder / ring need the coil axis along the body axis")
        dims = (sh.diameter, sh.height) if isinstance(sh, Cylinder) else (sh.d_outer, sh.d_inner, sh.height)
        loops = eddy_loops("cylinder" if isinstance(sh, Cylinder) else "ring", dims, pose, n_r=int(ecfg["n_r"]),
                           n_z=int(ecfg["n_z"]), resistivity=float(rho))
    return loops, float(rho)


def _irr_fraction(system: MagnetSystem) -> np.ndarray:
    """Per-cell irreversible fraction J_irr/Js_irr along the easy axis from the committed state, (n,)."""
    out = []
    for b, m in enumerate(system.magnets):
        st = system.states[b]
        if hasattr(st, "P_last"):  # PreisachState
            out.append(np.asarray(st.P_last, float))
        elif hasattr(st, "s"):  # LinearState: J_irr = Jr s
            out.append(np.asarray(st.s, float) if m.material.Jr != 0 else np.zeros(m.n_cells))
        else:
            raise TypeError(f"cannot read the irreversible fraction of {type(st).__name__}")
    return np.concatenate(out)


def _finite_field(fn, pts, size):
    """fn(pts) with non-finite entries re-evaluated at pts +/- 1e-6 size along a skewed direction and averaged.

    magpylib returns NaN within ~1e-8 size of a source edge (the field is log-singular there); this happens when a
    Gauss point lands on an eddy band's edge. The +/- average approximates the principal value.
    """
    h = np.asarray(fn(pts), float)
    bad = ~np.all(np.isfinite(h.reshape(len(pts), -1)), axis=1)
    if np.any(bad):
        d = 1e-6 * size * np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)]) / np.sqrt(6.0)
        h[bad] = 0.5 * (np.asarray(fn(pts[bad] + d), float) + np.asarray(fn(pts[bad] - d), float))
        if not np.all(np.isfinite(h)):
            raise FloatingPointError("field per amp is not finite even after the edge shift")
    return h


def prepare(cfg: dict, log=None) -> Setup:
    """Validate a scenario dict (temperatures already in K) and build the coupled model.

    Pre-pulse: the magnet is solved with H_app = 0 from its initial state and committed (open-circuit
    start), so the march of every WR iteration restarts from the same committed state.
    """
    if not isinstance(cfg, dict):
        raise ScenarioError("scenario must be a mapping")
    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise ScenarioError(f"unknown top-level key(s) {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}")
    for key in ("circuit", "coil", "magnet"):
        if key not in cfg:
            raise ScenarioError(f"scenario needs '{key}'")
    t0 = time.perf_counter()
    coupling = _merge(DEFAULT_COUPLING, cfg.get("coupling"), "coupling")
    if coupling["grid_tol"] is None:
        coupling["grid_tol"] = 0.8 * coupling["field_tol"]
    ecfg = _merge(DEFAULT_EDDY, cfg.get("eddy"), "eddy")
    circuit = build_circuit(cfg["circuit"])
    magnet, grade = build_magnet(cfg["magnet"])
    coil = build_coil(cfg["coil"])
    T0 = float((cfg["coil"] or {}).get("temperature", 293.15))
    system = MagnetSystem([magnet])
    e_glob = magnet.pose.matrix @ magnet.easy_axis
    s_hat = np.asarray(cfg.get("target_direction", e_glob), float).reshape(3)
    if np.linalg.norm(s_hat) == 0:
        raise ScenarioError("target_direction must be non-zero")
    s_hat = s_hat / np.linalg.norm(s_hat)
    pts, _ = system.field_points()
    h_c = _finite_field(coil.field_per_amp, pts, magnet.shape.size)
    loops, rho = (None, None)
    if ecfg["enabled"]:
        loops, rho = _eddy_network(magnet, coil, ecfg, grade)
        h_k = _finite_field(lambda q: loops.field_per_amp(q, n_q=int(ecfg["n_q"])), pts, magnet.shape.size)
        h_pts = np.concatenate([h_c[:, None, :], h_k], axis=1)
        L = loops.full_inductance(coil)
        R_loops = np.asarray(loops.R, float)
    else:
        h_pts = h_c[:, None, :]
        L = np.array([[coil.inductance()]])
        R_loops = np.zeros(0)
    n = system.n_cells
    nc = h_pts.shape[1]
    hbar = system.cell_average(h_pts.reshape(len(pts), -1)).reshape(n, nc, 3)
    ecfg["resistivity_used"] = rho
    d = np.linalg.norm(system.centroid - magnet.pose.position, axis=1)
    cc = np.flatnonzero(d <= d.min() * (1 + 1e-6) + 1e-12 * magnet.shape.size)
    Hcj = float("nan")
    if isinstance(magnet.material, PreisachMaterial):
        Hcj = float(magnet.material.characteristics(T=float(np.mean(magnet.temperature)))["Hcj"])
    st = Setup(cfg=cfg, name=str(cfg.get("name", "switch")), magnet=magnet, grade=grade, system=system, coil=coil,
               loops=loops, s_hat=s_hat, h_pts=h_pts, hbar=hbar, volume=system.volume.copy(), L=L, R_loops=R_loops,
               lumped=coil.lumped(T0=T0), circuit=circuit, coupling=coupling, eddy=ecfg, centroid_cells=cc, Hcj=Hcj)
    st.timings["prepare_fields_s"] = time.perf_counter() - t0
    sol = system.solve(H_app=None)
    system.commit(sol)
    st.states_pre = list(system.states)
    st.J_pre = sol.J.copy()
    st.sol_pre = sol
    st.dL_ref, st.split_info = reference_inductance(st, coupling["split"])
    st.timings["prepare_s"] = time.perf_counter() - t0
    if log:
        log(f"[prepare] {n} cells, {nc - 1} eddy loops, L = {coil.inductance() * 1e6:.4g} uH, "
            f"R0 = {st.lumped['R0'] * 1e3:.4g} mOhm, pre-pulse <P> = {_mean(_irr_fraction(system), st.volume):+.4f} "
            f"({st.timings['prepare_s']:.1f} s)")
    return st


def _mean(x, V):
    return float(np.sum(x * V) / np.sum(V))


# =====================================================================================================
# Circuit runs
# =====================================================================================================
def _vectors(setup: Setup):
    c = setup.circuit
    a, b = c["coil_nodes"]
    v = coil_vectors(setup.n_loops, thermal=True)
    if b == "0":
        vport = f"v({a})"
    elif a == "0":
        vport = f"-v({b})"
    else:
        vport = f"v({a})-v({b})"
    vecs = [v["i_coil"], v["u_coil"], v["T_coil"], *v["i_eddy"], vport, f"v({c['bank']['node']})",
            *c["scr"].values()]
    names = ["i_c", "u_R", "T_coil", *[f"i_e{k}" for k in range(setup.n_loops)], "v_port", "v_bank",
             *[f"scr_{k}" for k in c["scr"]]]
    return vecs, names


def make_deck(setup: Setup, emf: dict | None, L=None, lumped=None, robust=False) -> tuple[str, list[str]]:
    """ngspice deck text for EMF PWLs `emf` = {"coil" | "eddy<k>": (t [s], u [V])} (None = 0 V).

    robust: add a 1 GOhm shunt from every node to ground (ngspice `rshunt`; 0.45 uA at 450 V) and more Newton
    iterations per time point, the ngspice remedy for "Timestep too small" on a node left nearly floating
    when both bridge thyristors turn off; reltol is unchanged. Used only for a retry.
    """
    c = setup.circuit
    tran, options = c["tran"], c["options"]
    if robust:
        options = {**(options or {}), "rshunt": 1e9, "itl4": 100}
    vecs, names = _vectors(setup)
    L = setup.L + setup.dL_ref if L is None else L
    lumped = dict(setup.lumped if lumped is None else lumped)
    lumped["L"] = float(L[0, 0])
    eddy = {"L": L, "R": setup.R_loops} if setup.n_loops else None
    deck = build_deck(c["base_netlist"], c["coil_nodes"], lumped, eddy=eddy, emf_pwl=emf, extra_lines=c["extra_lines"],
                      models=c["models"], tran=tran, ic=c["ic"], options=options, vectors=vecs,
                      save=c["save"], base_dir=c["base_dir"], title=f"magsim switching: {setup.name}")
    return deck, names


def run_circuit(setup: Setup, emf: dict | None, workdir, L=None, lumped=None) -> dict:
    """One ngspice run; returns native-time arrays: time, I (n_circ, nt) [A], u_R, T_coil, v_port, v_bank, scr_*."""
    deck, names = make_deck(setup, emf, L=L, lumped=lumped, robust=setup.robust)
    try:
        d = run_ngspice(deck, workdir, names)
        d["retried"] = False
    except NgspiceError as err:
        if setup.robust or "timestep too small" not in str(err).lower():
            raise
        # one retry with node shunts: a nearly floating bridge node can stall ngspice's step control
        warnings.warn(f"ngspice 'Timestep too small' in {workdir}; retrying with rshunt = 1e9, itl4 = 100")
        setup.robust = True  # sticky: every later iterate uses the same deck, so WR compares like with like
        deck, names = make_deck(setup, emf, L=L, lumped=lumped, robust=True)
        d = run_ngspice(deck, Path(workdir) / "retry", names)
        d["retried"] = True
    d["I"] = np.vstack([d["i_c"]] + [d[f"i_e{k}"] for k in range(setup.n_loops)])
    d["deck"] = deck
    return d


# =====================================================================================================
# Field path grid and magnet march
# =====================================================================================================
def _href(setup: Setup, I) -> float:
    """Peak generated field (max over time and cells of |h_c i_c|) [A/m]."""
    return float(np.max(np.abs(I[0])) * np.max(np.linalg.norm(setup.hbar[:, 0, :], axis=1)))


def _extrema_times(t, y, rel):
    """Times of local extrema of y (sign changes of the increments) where |y| > rel max|y|."""
    y = np.asarray(y, float)
    ymax = float(np.max(np.abs(y))) if y.size else 0.0
    if y.size < 3 or ymax == 0.0:
        return np.zeros(0)
    d = np.diff(y)
    idx = np.flatnonzero(np.abs(d) > 1e-12 * ymax)
    if idx.size < 2:
        return np.zeros(0)
    sd = np.sign(d[idx])
    ch = np.flatnonzero(sd[1:] != sd[:-1])
    k = idx[ch] + 1
    k = k[np.abs(y[k]) > rel * ymax]
    return t[k]


def _arc_points(t, s, step, cap):
    """Times where the non-decreasing path length s(t) crosses multiples of step (at most cap points)."""
    if not s[-1] > 0 or not step > 0:
        return np.zeros(0)
    if s[-1] / step > cap:
        step = s[-1] / cap
    lev = np.arange(step, s[-1], step)
    j = np.clip(np.searchsorted(s, lev), 1, len(s) - 1)
    f = (lev - s[j - 1]) / np.where(s[j] > s[j - 1], s[j] - s[j - 1], 1.0)
    return t[j - 1] + f * (t[j] - t[j - 1])


def build_grid(setup: Setup, t, I) -> np.ndarray:
    """Common time grid [s]: uniform background + equal steps of the field-path length + equal steps of
    each eddy current's own path length.

    Field path s(t) = sum_c max_cells|h_c| int |di_c| (an upper bound of the max-cell field change): a
    point every grid_tol x the peak generated field. Eddy loop k: a point every eddy_grid_tol x max|i_k|
    of int |di_k|, which resolves the fast loop transients (time constants ~ us) that the EMF finite
    differences must follow even where their field is small (e.g. at switch-on).
    """
    cp = setup.coupling
    hmax = np.max(np.linalg.norm(setup.hbar, axis=2), axis=0)  # (n_circ,)
    dI = np.abs(np.diff(I, axis=1))
    s = np.concatenate([[0.0], np.cumsum((hmax[:, None] * dI).sum(axis=0))])
    Href = _href(setup, I)
    tg = [np.linspace(t[0], t[-1], int(cp["n_background"]))]
    if Href > 0:
        tg.append(_arc_points(t, s, cp["grid_tol"] * Href, cp["max_grid"]))
    for k in range(1, I.shape[0]):
        pk = float(np.max(np.abs(I[k])))
        if pk > 0:
            sk = np.concatenate([[0.0], np.cumsum(dI[k])])
            tg.append(_arc_points(t, sk, cp["eddy_grid_tol"] * pk, cp["max_grid"] // 4))
    return _merge_close(np.concatenate(tg), _min_dt(setup))


def _interp_rows(tq, t, Y):
    Y = np.atleast_2d(Y)
    return np.vstack([np.interp(tq, t, y) for y in Y])


def _min_dt(setup: Setup) -> float:
    """Minimum spacing of march / PWL samples [s]: 0.2 x tran tstep (PWL breakpoints closer than this make
    ngspice fail with 'timestep too small')."""
    tr = setup.circuit["tran"]
    return 0.2 * float(tr["tstep"])


def _merge_close(T, min_dt, keep=None) -> np.ndarray:
    """Sorted unique times with no two closer than min_dt; points in `keep` win over their neighbours."""
    T = np.unique(np.asarray(T, float))
    pri = np.zeros(T.size, bool) if keep is None or len(keep) == 0 else np.isin(T, keep)
    pri[[0, -1]] = True
    order = np.argsort(~pri, kind="stable")  # priority points first, then the rest in time order
    taken = []
    for i in order:
        if taken:
            j = np.searchsorted(taken, T[i])
            if (j < len(taken) and taken[j] - T[i] < min_dt) or (j > 0 and T[i] - taken[j - 1] < min_dt):
                continue
            taken.insert(j, T[i])
        else:
            taken.append(T[i])
    return np.asarray(taken)


def march_times(setup: Setup, tg, t, I) -> np.ndarray:
    """March samples: the common grid + current extrema (native times), refined so that the max-cell
    applied-field change between samples is <= field_tol x the peak generated field and each eddy current
    changes by <= eddy_grid_tol x its peak."""
    cp = setup.coupling
    ext = np.concatenate([_extrema_times(t, y, cp["extrema_rel"]) for y in I])
    T = _merge_close(np.concatenate([tg, ext]), _min_dt(setup), keep=ext)
    Href = _href(setup, I)
    if Href <= 0:
        return T
    lim = cp["field_tol"] * Href
    lim_k = cp["eddy_grid_tol"] * np.max(np.abs(I[1:]), axis=1) if I.shape[0] > 1 else np.zeros(0)
    for _ in range(3):
        Ia = _interp_rows(T, t, I)
        H = np.einsum("nck,ct->tnk", setup.hbar, Ia)
        ratio = np.max(np.linalg.norm(np.diff(H, axis=0), axis=2), axis=1) / lim
        if lim_k.size:
            ratio = np.maximum(ratio, np.max(np.abs(np.diff(Ia[1:], axis=1)) / np.maximum(lim_k, 1e-30)[:, None], axis=0))
        bad = np.flatnonzero(ratio > 1.0)
        if bad.size == 0:
            break
        new = []
        for j in bad:
            k = int(np.ceil(ratio[j]))
            new.append(T[j] + (T[j + 1] - T[j]) * np.arange(1, k) / k)
        T = _merge_close(np.concatenate([T, *new]), _min_dt(setup), keep=ext)
    return T


@dataclass
class March:
    """Magnet march along one current waveform (all per sample j = 0..ns-1 unless noted)."""

    t: np.ndarray
    I: np.ndarray  # (n_circ, ns)
    sHgen: np.ndarray  # (ns, n) s_hat . H_gen [A/m]
    sHapp: np.ndarray
    sHfelt: np.ndarray
    sB: np.ndarray  # [T]
    sJ: np.ndarray  # [T]
    P: np.ndarray  # (ns, n) irreversible fraction along the easy axis
    lam: np.ndarray  # (ns, n_circ) flux linkage [Wb]
    W_HdJ: float  # sum_cells V int H_app . dJ over the march + final open-circuit relaxation [J]
    J_oc: np.ndarray  # (n, 3) final open-circuit J [T]
    P_oc: np.ndarray  # (n,)
    sHfelt_oc: np.ndarray
    n_solves: int
    n_fail: int
    max_residual: float


def march(setup: Setup, T, I) -> March:
    """Quasi-static magnet march: solve + commit at every sample (rate-independent Preisach).

    Restarts from the committed pre-pulse state. A sample whose Newton solve does not converge is
    approached by recursive halving of the applied-field step (depth <= 4; failures are counted).
    """
    sysm = setup.system
    sysm.states = list(setup.states_pre)
    n, nc = sysm.n_cells, setup.n_circ
    ns = len(T)
    V = setup.volume
    s = setup.s_hat
    e_dot_s = sysm.easy @ s
    hb_s = np.einsum("nck,k->nc", setup.hbar, s)
    out = {k: np.zeros((ns, n)) for k in ("sHgen", "sHapp", "sHfelt", "sB", "sJ", "P")}
    lam = np.zeros((ns, nc))
    stats = {"solves": 0, "fail": 0, "res": 0.0, "W": 0.0}
    J_prev = setup.J_pre.copy()
    Happ_prev = np.zeros((n, 3))
    Hpts_prev = np.zeros((setup.h_pts.shape[0], 3))
    Href = max(_href(setup, I), 1e-30)
    sol = setup.sol_pre

    def advance(H_from, H_to, depth=0):
        nonlocal J_prev, Happ_prev
        sl = sysm.solve(H_app=H_to, J0=J_prev)
        stats["solves"] += 1
        if not sl.converged and depth < 4:
            mid = 0.5 * (H_from + H_to)
            advance(H_from, mid, depth + 1)
            return advance(mid, H_to, depth + 1)
        if not sl.converged:
            stats["fail"] += 1
        stats["res"] = max(stats["res"], float(sl.residual))
        sysm.commit(sl)
        stats["W"] += float(np.sum(V * np.einsum("nk,nk->n", 0.5 * (sl.H_app + Happ_prev), sl.J - J_prev)))
        J_prev, Happ_prev = sl.J.copy(), sl.H_app.copy()
        return sl

    for j in range(ns):
        Hpts = np.einsum("pck,c->pk", setup.h_pts, I[:, j])
        Hcells = np.einsum("nck,c->nk", setup.hbar, I[:, j])
        if j == 0 or np.max(np.abs(Hcells - Happ_prev)) > 1e-10 * Href:
            sol = advance(Hpts_prev, Hpts)
            Hpts_prev = Hpts
        out["sHgen"][j] = hb_s[:, 0] * I[0, j]
        out["sHapp"][j] = sol.H_app @ s
        out["sHfelt"][j] = sol.H @ s
        out["sB"][j] = sol.B @ s
        out["sJ"][j] = sol.J @ s
        out["P"][j] = _irr_fraction(sysm) * e_dot_s
        lam[j] = np.einsum("n,nk,nck->c", V, sol.J, setup.hbar)
    # pulse over: remove what is left of the applied field (open circuit)
    sl = advance(Hpts_prev, np.zeros_like(Hpts_prev))
    return March(t=np.asarray(T, float), I=I, lam=lam, W_HdJ=stats["W"], J_oc=sl.J.copy(),
                 P_oc=_irr_fraction(sysm) * e_dot_s, sHfelt_oc=sl.H @ s, n_solves=stats["solves"],
                 n_fail=stats["fail"], max_residual=stats["res"], **out)


def emf_from_march(setup: Setup, m: March) -> dict:
    """EMF PWLs u_c = d(lambda_c - sum_d dL_ref,cd i_d)/dt on the march samples [V] (second-order finite
    differences, no smoothing; dL_ref = 0 without WR splitting)."""
    names = setup.circuit_names()
    if len(m.t) < 3:
        return {k: (m.t, np.zeros_like(m.t)) for k in names}
    u = np.gradient(m.lam - m.I.T @ setup.dL_ref.T, m.t, axis=0)
    return {k: (m.t.copy(), u[:, c].copy()) for c, k in enumerate(names)}


def _resample_emf(emf: dict, t) -> dict:
    return {k: (t.copy(), np.interp(t, tt, uu)) for k, (tt, uu) in emf.items()}


def _emf_at(emf: dict | None, key: str, t) -> np.ndarray:
    if not emf or key not in emf:
        return np.zeros_like(t)
    tt, uu = emf[key]
    return np.interp(t, tt, uu)


# =====================================================================================================
# Waveform relaxation
# =====================================================================================================
def ideal_reference(setup: Setup, n_steps: int = 50) -> dict:
    """Open-circuit <J.s_hat> after an ideal saturating pulse: uniform 5 Hcj along s_hat, ramped up and down
    in n_steps committed steps each, from the scenario's initial state on the same mesh (Preisach only)."""
    if not math.isfinite(setup.Hcj):
        return {"J_s_mean_T": float("nan"), "P_mean": float("nan")}
    sysm = setup.system
    saved = list(sysm.states)
    sysm.states = [m.material.init_state(m.n_cells, m.initial) for m in sysm.magnets]
    amp = np.concatenate([np.linspace(0, 1, n_steps + 1), np.linspace(1, 0, n_steps + 1)[1:]]) * 5.0 * setup.Hcj
    J = None
    sol = None
    for a in amp:
        sol = sysm.solve(H_app=a * setup.s_hat, J0=J)
        sysm.commit(sol)
        J = sol.J
    e_dot_s = sysm.easy @ setup.s_hat
    out = {"J_s_mean_T": _mean(sol.J @ setup.s_hat, setup.volume),
           "P_mean": _mean(_irr_fraction(sysm) * e_dot_s, setup.volume), "converged": bool(sol.converged)}
    sysm.states = saved
    return out


def _max_field_step(setup: Setup, I) -> float:
    """Largest max-cell |dH_app| between consecutive march samples, relative to the peak generated field."""
    H = np.einsum("nck,ct->tnk", setup.hbar, I)
    if H.shape[0] < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(H, axis=0), axis=2)) / max(_href(setup, I), 1e-30))


def _w(w: dict, key: str) -> float:
    return w["coil"] if key == "coil" else w["eddy"]


def run_switching(cfg: dict, workdir=None, initial_emf=None, setup: Setup | None = None, log=None):
    """Coupled transient of a scenario dict. Returns (summary dict, waveforms dict).

    initial_emf : None (zero EMF in the first circuit run) or {"coil" | "eddy<k>": (t, u)} [s, V].
    """
    st = setup or prepare(cfg, log=log)
    cp = st.coupling
    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="magsim_switch_")
        workdir = tmp.name
    workdir = Path(workdir)
    t0 = time.perf_counter()
    emf = copy.deepcopy(initial_emf) if initial_emf else None  # EMF applied in the next circuit run
    history = []
    tg = I_prev = None
    m = circ = emf_run = None
    converged = False
    T_frozen = None
    t_circ = t_march = 0.0
    rl = cp["relax"]
    w = {"coil": float(rl.get("coil", 1.0)), "eddy": float(rl.get("eddy", 1.0))} if isinstance(rl, dict) else \
        {"coil": float(rl), "eddy": float(rl)}
    try:
        for it in range(1, int(cp["max_iter"]) + 1):
            ta = time.perf_counter()
            emf_run = emf
            circ = run_circuit(st, emf_run, workdir / f"it{it:02d}")
            t_circ += time.perf_counter() - ta
            if tg is None:
                tg = build_grid(st, circ["time"], circ["I"])
            Ig = _interp_rows(tg, circ["time"], circ["I"])
            res = res_all = float("nan")
            per = np.full(st.n_circ, np.nan)
            t_res = float("nan")
            if I_prev is not None:
                scale = max(float(np.max(np.abs(Ig[0]))), 1e-30)
                res = float(np.max(np.abs(Ig[0] - I_prev[0])) / scale)
                res_all = float(np.max(np.abs(Ig - I_prev)) / scale)
                per = np.max(np.abs(Ig - I_prev), axis=1) / scale
                t_res = float(tg[np.argmax(np.max(np.abs(Ig - I_prev), axis=0))])
            tb = time.perf_counter()
            # march samples: rebuilt (current extrema + field_tol refinement) until the iterates settle, then
            # frozen, so the discrete march is a continuous map of the currents and WR converges to its fixed point
            if T_frozen is None:
                T = march_times(st, tg, circ["time"], circ["I"])
                if math.isfinite(res) and res < cp["freeze_tol"]:
                    T_frozen = T
            else:
                T = T_frozen
            Im = _interp_rows(T, circ["time"], circ["I"])
            m = march(st, T, Im)
            t_march += time.perf_counter() - tb
            step = _max_field_step(st, Im)
            history.append({"iteration": it, "residual": res, "residual_all": res_all, "march_samples": len(T),
                            "samples_frozen": T_frozen is not None, "max_field_step_rel": step,
                            "newton_failures": m.n_fail, "I_pk_A": float(np.max(np.abs(circ["I"][0]))),
                            "residual_per_circuit": " ".join(f"{x:.2e}" for x in per), "t_max_change_s": t_res,
                            "time_s": time.perf_counter() - ta})
            if log:
                log(f"[WR {it:2d}] residual {res:.3e} (all circuits {res_all:.3e}), I_pk {history[-1]['I_pk_A']:.2f} A, "
                    f"{len(T)} samples{' (frozen)' if T_frozen is not None else ''}, max field step {step:.2e}, "
                    f"{m.n_fail} Newton failures, {history[-1]['time_s']:.1f} s")
            if I_prev is not None and res < cp["tol"]:
                converged = True
                break
            emf = emf_from_march(st, m)
            if emf_run and (w["coil"] != 1.0 or w["eddy"] != 1.0):  # under-relaxation u <- w u_new + (1 - w) u_old
                old = _resample_emf(emf_run, emf["coil"][0])
                emf = {k: (tt, _w(w, k) * uu + (1.0 - _w(w, k)) * old[k][1]) if k in old else (tt, uu)
                       for k, (tt, uu) in emf.items()}
            I_prev = Ig
        summary, waves = _postprocess(st, circ, m, emf_run, tg, history, converged)
    finally:
        if tmp is not None:
            tmp.cleanup()
    summary["time_circuit_s"] = t_circ
    summary["time_march_s"] = t_march
    summary["time_total_s"] = time.perf_counter() - t0 + st.timings.get("prepare_s", 0.0)
    waves["emf_used"] = emf_run
    waves["deck"] = circ["deck"]
    return summary, waves


# =====================================================================================================
# Outputs: metrics, energy audit
# =====================================================================================================
def _field_stats(Q, V, cc) -> tuple[dict, dict]:
    """Max over time of the centroid / volume-mean series, and the weakest cell's peak (min over cells)."""
    cen = Q[:, cc].mean(axis=1)
    mean = Q @ V / V.sum()
    cellmax = Q.max(axis=0)
    w = int(np.argmin(cellmax))
    return ({"centroid": float(cen.max()), "mean": float(mean.max()), "weakest": float(cellmax[w]), "weakest_cell": w},
            {"centroid": cen, "mean": mean, "weakest": Q[:, w]})


def _postprocess(st: Setup, circ: dict, m: March, emf_used, tg, history, converged):
    c = st.circuit
    t = circ["time"]
    I = circ["I"]
    ic = I[0]
    V = st.volume
    S: dict = {"name": st.name, "n_cells": int(st.system.n_cells), "n_loops": int(st.n_loops),
               "L_coil_H": float(st.L[0, 0]), "R_coil_ohm": float(st.lumped["R0"]),
               "C_th_coil_J_per_K": float(st.lumped["C_th"]), "Hcj_A_per_m": st.Hcj,
               "eddy_resistivity_ohm_m": st.eddy.get("resistivity_used") or float("nan"),
               "wr_split": st.split_info.get("split"), "dL_ref_coil_H": float(st.dL_ref[0, 0]),
               "mu_lo": st.split_info.get("mu_lo", float("nan")), "mu_hi": st.split_info.get("mu_hi", float("nan"))}
    # ---- circuit / device stress
    k = int(np.argmax(np.abs(ic)))
    didt = np.diff(ic) / np.diff(t)
    S.update(I_pk_A=float(ic[k]), t_pk_s=float(t[k] - c["fire_time"]), didt_max_A_per_s=float(np.max(np.abs(didt))),
             I2t_coil_A2s=float(np.trapezoid(ic**2, t)), I_end_A=float(ic[-1]),
             V_bank_end_V=float(circ["v_bank"][-1]), V_bank_min_V=float(np.min(circ["v_bank"])),
             T_coil_peak_K=float(np.max(circ["T_coil"])), T_coil_end_K=float(circ["T_coil"][-1]),
             dT_coil_K=float(circ["T_coil"][-1] - circ["T_coil"][0]))
    if st.n_loops:
        S["I_eddy_pk_A"] = float(np.max(np.abs(I[1:])))
    rat = c["scr_ratings"]
    for name in c["scr"]:
        ia = circ[f"scr_{name}"]
        S[f"I_pk_{name}_A"] = float(np.max(ia))
        S[f"I2t_{name}_A2s"] = float(np.trapezoid(ia**2, t))
        S[f"didt_max_{name}_A_per_s"] = float(np.max(np.diff(ia) / np.diff(t)))
        S[f"I2t_{name}_ratio"] = S[f"I2t_{name}_A2s"] / rat["I2t_A2s"]
        S[f"didt_{name}_ratio"] = S[f"didt_max_{name}_A_per_s"] / rat["didt_A_per_s"]
    if abs(S["I_end_A"]) > 0.01 * abs(S["I_pk_A"]):
        warnings.warn(f"coil current at tstop is {S['I_end_A']:.3g} A (> 1 % of peak): extend circuit.tran.tstop",
                      stacklevel=2)
    # ---- fields (max over time of s_hat . H on the march samples)
    cc = st.centroid_cells
    fstats, series = {}, {}
    for key, Q in (("H_gen", m.sHgen), ("H_app", m.sHapp), ("H_felt", m.sHfelt), ("B_in", m.sB)):
        fs, se = _field_stats(Q, V, cc)
        fstats[key], series[key] = fs, se
        for loc in ("centroid", "mean", "weakest"):
            if key == "B_in":
                S[f"{key}_{loc}_T"] = fs[loc]
            else:
                S[f"{key}_{loc}_A_per_m"] = fs[loc]
                S[f"{key}_{loc}_T"] = MU0 * fs[loc]
        S[f"{key}_weakest_cell"] = fs["weakest_cell"]
    wJ = int(np.argmin(m.sJ[-1]))
    series["J"] = {"mean": m.sJ @ V / V.sum(), "weakest": m.sJ[:, wJ], "weakest_cell": wJ}
    series["P_mean"] = m.P @ V / V.sum()
    # ---- magnetisation state
    S["switched_fraction_pre"] = float(series["P_mean"][0])
    S["switched_fraction"] = _mean(m.P_oc, V)
    S["J_s_mean_pre_T"] = _mean(st.J_pre @ st.s_hat, V)
    S["J_s_mean_final_T"] = _mean(m.J_oc @ st.s_hat, V)
    S["J_s_min_final_T"] = float(np.min(m.J_oc @ st.s_hat))
    ref = ideal_reference(st)
    S["J_s_mean_ideal_T"] = ref["J_s_mean_T"]
    S["switched_fraction_ideal"] = ref["P_mean"]
    S["relative_remanence"] = (S["J_s_mean_final_T"] / ref["J_s_mean_T"]) if ref["J_s_mean_T"] else float("nan")
    # ---- energy audit (native ngspice time base of the final circuit run)
    names = st.circuit_names()
    U = np.vstack([_emf_at(emf_used, k, t) for k in names])
    C, V0 = c["bank"]["C"], c["bank"]["V0"]
    E0 = 0.5 * C * V0**2
    E_C = 0.5 * C * circ["v_bank"][-1] ** 2
    E_port = float(np.trapezoid(circ["v_port"] * ic, t))
    E_R = float(np.trapezoid(-circ["u_R"] * ic, t))
    E_eddy = float(sum(np.trapezoid(I[1 + j] ** 2 * st.R_loops[j], t) for j in range(st.n_loops)))
    # magnet work through the circuits: EMF sources + the reference response carried by L + dL_ref
    W1 = float(np.trapezoid(np.sum(I * U, axis=0), t)
               + 0.5 * I[:, -1] @ st.dL_ref @ I[:, -1] - 0.5 * I[:, 0] @ st.dL_ref @ I[:, 0])
    dEL = float(0.5 * I[:, -1] @ st.L @ I[:, -1] - 0.5 * I[:, 0] @ st.L @ I[:, 0])
    resid = E_port - (E_R + E_eddy + W1 + dEL)
    S.update(E0_J=E0, E_C_end_J=E_C, E_port_J=E_port, E_R_coil_J=E_R, E_eddy_J=E_eddy, W_magnet_emf_J=W1,
             W_magnet_HdJ_J=m.W_HdJ, W_magnet_diff_J=W1 - m.W_HdJ, dE_L_J=dEL, E_residual_J=resid,
             E_residual_rel=resid / E0 if E0 else float("nan"), E_circuit_losses_J=E0 - E_C - E_port,
             E_R_coil_thermal_J=float(st.lumped["C_th"] * (circ["T_coil"][-1] - circ["T_coil"][0])))
    # ---- WR / numerics
    last = history[-1]
    S.update(wr_iterations=len(history), wr_residual=last["residual"], wr_converged=bool(converged),
             wr_tol=float(st.coupling["tol"]), march_samples=len(m.t), newton_failures=int(m.n_fail),
             newton_max_residual_T=float(m.max_residual))
    S["wr_history"] = history
    # ---- waveforms
    Ig = _interp_rows(tg, t, I)
    circ_w = {"t": tg, "i_coil": Ig[0], "v_bank": np.interp(tg, t, circ["v_bank"]),
              "T_coil": np.interp(tg, t, circ["T_coil"]), "v_port": np.interp(tg, t, circ["v_port"]),
              "u_coil": _emf_at(emf_used, "coil", tg)}
    for j in range(st.n_loops):
        circ_w[f"i_eddy{j}"] = Ig[1 + j]
        circ_w[f"u_eddy{j}"] = _emf_at(emf_used, f"eddy{j}", tg)
    for name in c["scr"]:
        circ_w[f"i_{name}"] = np.interp(tg, t, circ[f"scr_{name}"])
    mag_w = {"t": m.t, "i_coil": m.I[0], "lambda_coil": m.lam[:, 0], "P_mean": series["P_mean"],
             "J_mean": series["J"]["mean"], "J_weakest": series["J"]["weakest"]}
    for key in ("H_gen", "H_app", "H_felt", "B_in"):
        for loc in ("centroid", "mean", "weakest"):
            mag_w[f"{key}_{loc}"] = series[key][loc]
    cells = {"x": st.system.centroid[:, 0], "y": st.system.centroid[:, 1], "z": st.system.centroid[:, 2],
             "volume": V, "peak_H_felt": m.sHfelt.max(axis=0), "peak_H_app": m.sHapp.max(axis=0),
             "peak_H_gen": m.sHgen.max(axis=0), "final_J": m.J_oc @ st.s_hat, "final_P": m.P_oc,
             "pre_J": st.J_pre @ st.s_hat}
    waves = {"circuit": circ_w, "magnet": mag_w, "cells": cells, "native": {k: circ[k] for k in circ if k != "deck"},
             "wr": history}
    return S, waves


# =====================================================================================================
# Linear magnet: inductance fold-in (gate S4)
# =====================================================================================================
def linear_fold_in(setup: Setup, material=None) -> dict:
    """Magnet response folded into the inductance matrix: dL_cd = sum_cells V J^(d) . h_c, J^(d) = MoM
    solution for 1 A in circuit d minus the zero-field solution [H] (exact for a history-free linear
    material, reciprocity [lelarasmee1982 coupling]). `material`: evaluate with this (linear) material
    instead of the magnet's own (swapped in temporarily on the same mesh). Returns dL (raw), its
    symmetric part and the relative asymmetry (collocation MoM is not exactly symmetric)."""
    sysm = setup.system
    mag = sysm.magnets[0]
    saved_mat, saved_states = mag.material, list(sysm.states)
    try:
        if material is not None:
            mag.material = material
            sysm.states = [material.init_state(mag.n_cells, 0.0)]
        J0 = sysm.solve(H_app=None).J
        nc = setup.n_circ
        dL = np.zeros((nc, nc))
        for d in range(nc):
            Jd = sysm.solve(H_app=setup.h_pts[:, d, :], J0=J0).J - J0
            dL[:, d] = np.einsum("n,nk,nck->c", setup.volume, Jd, setup.hbar)
    finally:
        mag.material, sysm.states = saved_mat, saved_states
    sym = 0.5 * (dL + dL.T)
    asym = float(np.max(np.abs(dL - dL.T)) / max(np.max(np.abs(sym)), 1e-300))
    return {"dL": dL, "dL_sym": sym, "asymmetry": asym}


MU_HI_CAP = 1e3


def material_mu_range(material, T) -> tuple[float, float, float]:
    """(mu_lo, mu_hi, mu_perp): reversible recoil permeability, 1 + the steepest major-loop slope
    max dJ/dH / MU0 (capped at MU_HI_CAP), and the transverse permeability. For a Preisach material every
    minor-loop slope lies in [mu_lo, mu_hi]; a linear material has mu_lo = mu_hi = mu_rec."""
    if isinstance(material, PreisachMaterial):
        Hc = float(material.characteristics(T=T)["Hcj"])
        H = np.linspace(-4 * Hc, 4 * Hc, 4001)
        slope = max(float(np.max(material.major_loop(H, br, T=T)[1])) for br in ("ascending", "descending"))
        return float(material.mu_rec), float(min(1.0 + slope / MU0, MU_HI_CAP)), float(material.mu_perp)
    if isinstance(material, LinearMaterial):
        return material.mu_rec, material.mu_rec, material.mu_perp
    raise TypeError(f"no permeability range for {type(material).__name__}")


def reference_inductance(setup: Setup, split) -> tuple[np.ndarray, dict]:
    """Linear reference response dL_ref [H] folded into the SPICE inductance matrix (WR splitting).

    split = "none": 0. A number mu: linear magnet (mu along the easy axis, the material's mu_perp across).
    "auto": dL_ref = (dL(mu_lo) + dL(mu_hi)) / 2 with (mu_lo, mu_hi) = `material_mu_range`. The circuit
    then carries L + dL_ref and the EMF sources only u = d(lambda - dL_ref i)/dt, so (scalar picture)
    the WR error gain is (dL_inc - dL_ref)/(L + dL_ref); with dL_inc between dL(mu_lo) and dL(mu_hi) its
    magnitude is < 1/2 (dL_hi - dL_lo)/(L + dL_ref) < 1, i.e. the iteration contracts even when the
    magnet's incremental inductance exceeds the coil's own (Alnico rod) - plain Gauss-Jacobi WR diverges
    there. The fixed point (the coupled solution) does not depend on dL_ref.
    """
    nc = setup.n_circ
    if split in (None, "none", False, 0):
        return np.zeros((nc, nc)), {"split": "none"}
    T = float(np.mean(setup.magnet.temperature))
    mu_lo, mu_hi, mu_perp = material_mu_range(setup.magnet.material, T)
    if split == "auto":
        lo = linear_fold_in(setup, LinearMaterial(0.0, mu_lo, mu_perp))["dL_sym"]
        hi = linear_fold_in(setup, LinearMaterial(0.0, mu_hi, mu_perp))["dL_sym"] if mu_hi != mu_lo else lo
        dL = 0.5 * (lo + hi)
        info = {"split": "auto", "mu_lo": mu_lo, "mu_hi": mu_hi, "mu_perp": mu_perp,
                "dL_lo_coil_H": float(lo[0, 0]), "dL_hi_coil_H": float(hi[0, 0])}
    else:
        mu = float(split)
        dL = linear_fold_in(setup, LinearMaterial(0.0, mu, mu_perp))["dL_sym"]
        info = {"split": "mu", "mu_ref": mu, "mu_perp": mu_perp}
    info["dL_ref_coil_H"] = float(dL[0, 0])
    try:
        np.linalg.cholesky(setup.L + dL)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("L + dL_ref is not positive definite") from e
    return dL, info


# =====================================================================================================
# Writing results, plots
# =====================================================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_run(setup: Setup, summary: dict, waves: dict, out_dir, figures: bool = True) -> list[Path]:
    """Write summary.json, waveforms CSVs, cell map CSV, WR history, deck, provenance and figures."""
    import pandas as pd

    from magsim.sweep import write_provenance

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    with open(out / "scenario.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(_jsonable(setup.cfg), f, sort_keys=False)
    for key, fname in (("circuit", "waveforms_circuit.csv"), ("magnet", "waveforms_magnet.csv"),
                       ("cells", "cells.csv")):
        pd.DataFrame(waves[key]).to_csv(out / fname, index=False)
        files.append(out / fname)
    pd.DataFrame(waves["wr"]).to_csv(out / "wr_history.csv", index=False)
    (out / "deck_final.cir").write_text(waves["deck"], encoding="utf-8")
    write_provenance(out, extra={"scenario": setup.name})
    files += [out / "summary.json", out / "wr_history.csv"]
    if figures:
        files += plot_run(setup, summary, waves, out / "figures")
    return files


def _legend_above(ax, ncol=3):
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol, borderaxespad=0.0, handlelength=1.5,
              columnspacing=1.0)


def plot_run(setup: Setup, S: dict, W: dict, fig_dir) -> list[Path]:
    """Standard figures of one run (PDF + PNG): waveforms, field histories, J, cell maps, WR residual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from magsim import plotting

    plotting.use_style()
    fig_dir = Path(fig_dir)
    files = []
    cw, mw = W["circuit"], W["magnet"]
    tus = cw["t"] * 1e6
    # 1) current, bank voltage, coil temperature
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(plotting.IEEE_COL, 3.6))
    axs[0].plot(tus, cw["i_coil"] * 1e-3, label="coil")
    if setup.n_loops:
        tot = sum(cw[f"i_eddy{j}"] for j in range(setup.n_loops))
        axs[0].plot(tus, tot * 1e-3, label="eddy loops (sum)")
        _legend_above(axs[0], 2)
    axs[0].set_ylabel("current [kA]")
    axs[1].plot(tus, cw["v_bank"])
    axs[1].set_ylabel("bank voltage [V]")
    axs[2].plot(tus, cw["T_coil"] - 273.15)
    axs[2].set_ylabel("coil temp. [°C]")
    axs[2].set_xlabel("time [µs]")
    files += plotting.save(fig, fig_dir / "circuit_waveforms")
    plt.close(fig)
    # 2) field histories with the Hcj line
    tm = mw["t"] * 1e6
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(plotting.IEEE_PAGE, 2.3))
    labels = {"H_gen": r"$H_\mathrm{gen}$", "H_app": r"$H_\mathrm{app}$", "H_felt": r"$H_\mathrm{felt}$"}
    for ax, loc, title in zip(axs, ("centroid", "mean", "weakest"), ("centroid cell", "volume mean", "weakest cell")):
        for key in ("H_gen", "H_app", "H_felt"):
            ax.plot(tm, mw[f"{key}_{loc}"] * 1e-6, label=labels[key])
        if math.isfinite(setup.Hcj):
            ax.axhline(setup.Hcj * 1e-6, color="0.5", ls="--", lw=0.7, label=r"$H_\mathrm{cJ}$")
        ax.set_xlabel("time [µs]")
        ax.text(0.98, 0.04, title, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    axs[0].set_ylabel(r"$\hat s\cdot H$ [MA/m]")
    _legend_above(axs[0], 4)
    files += plotting.save(fig, fig_dir / "field_histories")
    plt.close(fig)
    # 3) J mean / weakest
    fig, ax = plt.subplots()
    ax.plot(tm, mw["J_mean"], label="volume mean")
    ax.plot(tm, mw["J_weakest"], label="weakest cell (final)")
    ax.set_xlabel("time [µs]")
    ax.set_ylabel(r"$\hat s\cdot J$ [T]")
    _legend_above(ax, 2)
    files += plotting.save(fig, fig_dir / "J_history")
    plt.close(fig)
    # 4) cell maps
    files += _plot_maps(setup, W["cells"], fig_dir / "cell_maps", plt, plotting)
    # 5) WR residual
    hist = [h for h in W["wr"] if math.isfinite(h["residual"])]
    if hist:
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL, 2.2))
        ax.semilogy([h["iteration"] for h in hist], [h["residual"] for h in hist], marker="o",
                    label=r"$\max|\Delta i_c|/\max|i_c|$")
        ax.axhline(setup.coupling["tol"], color="0.5", ls="--", lw=0.7, label="tolerance")
        ax.set_xlabel("WR iteration")
        ax.set_ylabel("residual [-]")
        _legend_above(ax, 2)
        files += plotting.save(fig, fig_dir / "wr_residual")
        plt.close(fig)
    return files


def _plot_maps(setup: Setup, cells, stem, plt, plotting) -> list[Path]:
    bm = setup.magnet.body_mesh
    g = bm.grid
    Hp = np.asarray(cells["peak_H_felt"]) * 1e-6
    Jf = np.asarray(cells["final_J"])
    Js = float(np.max(np.abs(Jf))) or 1.0
    hcj = setup.Hcj * 1e-6 if math.isfinite(setup.Hcj) else None
    if g["type"] == "cuboid":
        xe, ye, ze = (np.asarray(e) * 1e3 for e in g["edges"])
        idx = g["idx"]
        kz, jy = idx.shape[2] // 2, idx.shape[1] // 2
        panels = [("x [mm]", "y [mm]", xe, ye, idx[:, :, kz]), ("x [mm]", "z [mm]", xe, ze, idx[:, jy, :])]
    else:
        re_, ze = np.asarray(g["r_edges"]) * 1e3, np.asarray(g["z_edges"]) * 1e3
        panels = [("r [mm]", "z [mm]", re_, ze, g["idx"][:, 0, :])]
    fig, axs = plt.subplots(2, len(panels), figsize=(plotting.IEEE_COL * (1.0 if len(panels) == 1 else 2.0), 4.0),
                            squeeze=False)
    for col, (xl, yl, a, b, I2) in enumerate(panels):
        for row, (vals, cmap, lim, lab) in enumerate((
                (Hp, "viridis", None, r"peak $\hat s\cdot H_\mathrm{felt}$ [MA/m]"),
                (Jf, "RdBu_r", (-Js, Js), r"final $\hat s\cdot J$ [T]"))):
            ax = axs[row, col]
            Z = vals[I2].T
            kw = {"vmin": lim[0], "vmax": lim[1]} if lim else {}
            pm = ax.pcolormesh(a, b, Z, cmap=cmap, shading="flat", **kw)
            if row == 0 and hcj is not None and Z.min() < hcj < Z.max() and min(Z.shape) > 1:
                ac, bc = 0.5 * (a[1:] + a[:-1]), 0.5 * (b[1:] + b[:-1])
                cs = ax.contour(ac, bc, Z, levels=[hcj], colors="white", linewidths=0.7)
                ax.clabel(cs, fmt={hcj: r"$H_\mathrm{cJ}$"}, fontsize=6)
            ax.set_aspect("equal")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            cb = fig.colorbar(pm, ax=ax, shrink=0.9)
            cb.set_label(lab)
            cb.outline.set_linewidth(0.6)
    files = plotting.save(fig, stem)
    plt.close(fig)
    return list(files)


# =====================================================================================================
# Sweeps and CLI
# =====================================================================================================
def _scalars(summary: dict) -> dict:
    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float, bool, np.floating, np.integer))}


def sweep_point(cfg: dict) -> dict:
    """Sweep worker: one coupled run -> scalar outputs (top-level, picklable)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary, _ = run_switching(cfg)
    return _scalars(summary)


def _out_dir(cfg, out_dir, suffix=""):
    if out_dir is not None:
        return Path(out_dir)
    o = (cfg.get("outputs") or {}).get("dir")
    return _resolve(o) if o else ROOT / "res_sim" / (str(cfg.get("name", "switch")) + suffix)


def _print_summary(S: dict, out=print):
    out(f"  I_pk = {S['I_pk_A']:.1f} A at {S['t_pk_s'] * 1e6:.1f} us after firing, di/dt max = "
        f"{S['didt_max_A_per_s'] * 1e-6:.2f} A/us, V_bank end = {S['V_bank_end_V']:.2f} V, coil dT = {S['dT_coil_K']:.2f} K")
    for key in ("H_gen", "H_app", "H_felt"):
        out(f"  {key:6s} peak s.H: centroid {S[f'{key}_centroid_A_per_m'] * 1e-6:.4f}  mean {S[f'{key}_mean_A_per_m'] * 1e-6:.4f}"
            f"  weakest {S[f'{key}_weakest_A_per_m'] * 1e-6:.4f} MA/m")
    out(f"  switched fraction <P> = {S['switched_fraction']:+.4f} (pre {S['switched_fraction_pre']:+.4f}), "
        f"relative remanence = {S['relative_remanence']:.4f}")
    out(f"  energy: E0 {S['E0_J']:.2f} J, port {S['E_port_J']:.3f}, R_coil {S['E_R_coil_J']:.3f}, eddy {S['E_eddy_J']:.4f}, "
        f"W_mag {S['W_magnet_emf_J']:.4f} (HdJ {S['W_magnet_HdJ_J']:.4f}), residual {S['E_residual_rel'] * 100:.4f} % E0")
    out(f"  WR: {S['wr_iterations']} iterations, residual {S['wr_residual']:.2e}, converged {S['wr_converged']}")


def cli_run(scenario, out_dir=None, workers=None) -> int:
    """`magsim switch run SCENARIO`: one coupled run, CSV + figures in res_sim/<name>/ (or --out)."""
    cfg = load_scenario(scenario)
    st = prepare(cfg, log=print)
    S, W = run_switching(cfg, setup=st, log=print)
    out = _out_dir(cfg, out_dir)
    write_run(st, S, W, out, figures=(cfg.get("outputs") or {}).get("figures", True))
    print(f"magsim switch run {cfg.get('name')}: results in {out}")
    _print_summary(S)
    return 0 if S["wr_converged"] else 1


def _plot_sweep(cfg, df, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from magsim import plotting

    plotting.use_style()
    sw = cfg["sweep"]
    axes = [a["path"] for a in sw["axes"]]
    labels = {**{"switched_fraction": "switched fraction ⟨P⟩ [-]", "relative_remanence": "relative remanence [-]",
                 "H_felt_weakest_A_per_m": "weakest-cell peak felt field [A/m]", "I_pk_A": "peak coil current [A]",
                 "dT_coil_K": "coil temperature rise [K]"}, **(sw.get("labels") or {})}
    files = []
    ok = df[df["error"].isna()] if "error" in df else df
    for p in sw.get("plots") or ([{"x": axes[0], "y": ["switched_fraction", "relative_remanence"]}] if len(axes) == 1 else []):
        ys = p["y"] if isinstance(p["y"], list) else [p["y"]]
        for y in ys:
            fig, ax = plt.subplots()
            plotting.line(ok, p["x"], y, hue=p.get("hue"), xlabel=p.get("xlabel", labels.get(p["x"], p["x"])),
                          ylabel=labels.get(y, y), ax=ax)
            files += plotting.save(fig, out / "figures" / f"sweep_{y}")
            plt.close(fig)
    for h in sw.get("heatmaps") or []:
        fig, ax = plt.subplots(figsize=(plotting.IEEE_COL, 2.6))
        plotting.heatmap(ok, h["x"], h["y"], h["z"], ax=ax, xlabel=h.get("xlabel", h["x"]), ylabel=h.get("ylabel", h["y"]),
                         zlabel=h.get("zlabel", labels.get(h["z"], h["z"])), levels=h.get("levels"))
        files += plotting.save(fig, out / "figures" / f"heatmap_{h['z']}")
        plt.close(fig)
    return files


def cli_sweep(scenario, out_dir=None, workers=None) -> int:
    """`magsim switch sweep SCENARIO`: grid over `sweep.axes`, sweep.csv + plots in res_sim/<name>/."""
    from magsim.sweep import run_sweep

    cfg = load_scenario(scenario)
    if not cfg.get("sweep") or not cfg["sweep"].get("axes"):
        raise ScenarioError("scenario has no sweep.axes")
    out = _out_dir(cfg, out_dir)
    nw = int(workers if workers is not None else cfg["sweep"].get("workers", 4))
    prepare(cfg)  # fail fast on a bad base scenario
    base = {k: v for k, v in cfg.items() if k != "sweep"}
    df = run_sweep(base, cfg["sweep"]["axes"], sweep_point, workers=nw, out_dir=out)
    _plot_sweep(cfg, df, out)
    cols = [a["path"] for a in cfg["sweep"]["axes"]] + [c for c in ("I_pk_A", "switched_fraction", "relative_remanence",
                                                                    "H_felt_weakest_A_per_m", "wr_iterations") if c in df]
    print(df[cols].to_string(index=False))
    bad = df["error"].notna().sum() if "error" in df else 0
    print(f"magsim switch sweep {cfg.get('name')}: {len(df)} points ({bad} failed), results in {out}")
    return 0 if bad == 0 else 1


def _richardson(h, f):
    """Celik et al. 2008 [celik2008] three-grid estimate (h fine -> coarse): observed order p, extrapolated value,
    fine-grid GCI. h: representative cell sizes (h1 < h2 < h3), f: values. NaN when not monotone."""
    h1, h2, h3 = h
    f1, f2, f3 = f
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return {"p": float("nan"), "f_ext": float("nan"), "gci_fine": float("nan")}
    s = np.sign(e32 / e21)
    p = 1.0
    for _ in range(100):
        q = np.log((r21**p - s) / (r32**p - s))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    f_ext = (r21**p * f1 - f2) / (r21**p - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r21**p - 1) if f1 else float("nan")
    return {"p": float(p), "f_ext": float(f_ext), "gci_fine": float(gci)}


MESH_KEYS = ("I_pk_A", "H_felt_centroid_A_per_m", "H_felt_mean_A_per_m", "H_felt_weakest_A_per_m",
             "switched_fraction", "relative_remanence", "W_magnet_emf_J")


def mesh_convergence(cfg: dict, meshes, log=None) -> dict:
    """Key outputs on >= 3 magnet meshes (coarse -> fine), with a Celik three-grid estimate on the finest three."""
    rows = []
    for n in meshes:
        c = copy.deepcopy(cfg)
        mesh = dict(c["magnet"].get("mesh") or {})
        mesh["n"] = list(n)
        c["magnet"]["mesh"] = mesh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S, _ = run_switching(c, log=log)
        vol = build_shape(c["magnet"]["shape"]).volume
        rows.append({"mesh": "x".join(map(str, n)), "n_cells": S["n_cells"], "h_m": (vol / S["n_cells"]) ** (1 / 3),
                     **{k: S[k] for k in MESH_KEYS}, "wr_iterations": S["wr_iterations"], "time_s": S["time_total_s"]})
        if log:
            log(f"[mesh {rows[-1]['mesh']}] " + ", ".join(f"{k}={rows[-1][k]:.6g}" for k in MESH_KEYS))
    rows.sort(key=lambda r: -r["h_m"])
    est = {}
    if len(rows) >= 3:
        fine3 = rows[-3:][::-1]
        for k in MESH_KEYS:
            est[k] = _richardson([r["h_m"] for r in fine3], [r[k] for r in fine3])
    return {"rows": rows, "richardson": est}


def initial_guess_check(cfg: dict, base_summary: dict, base_waves: dict, scale: float = 0.5, log=None) -> dict:
    """Gate S2 (initial guess): rerun with the initial EMF = scale x the converged EMF and compare i_c(t)."""
    emf0 = {k: (tt, scale * uu) for k, (tt, uu) in base_waves["emf_used"].items()}
    S2, W2 = run_switching(cfg, initial_emf=emf0, log=log)
    t = base_waves["circuit"]["t"]
    i1 = base_waves["circuit"]["i_coil"]
    i2 = np.interp(t, W2["circuit"]["t"], W2["circuit"]["i_coil"])
    diff = float(np.max(np.abs(i1 - i2)) / np.max(np.abs(i1)))
    return {"scale": scale, "max_rel_diff": diff, "iterations": S2["wr_iterations"], "converged": S2["wr_converged"],
            "switched_fraction": S2["switched_fraction"], "switched_fraction_base": base_summary["switched_fraction"]}


def linear_magnet_check(cfg: dict, mu: float = 4.0, log=None) -> dict:
    """Gate S4: same scenario with a purely reversible linear magnet (Jr = 0, mu_rec = mu_perp = mu): WR
    result vs an all-SPICE run with the magnet folded into the inductance matrix; plus the sign test."""
    c = copy.deepcopy(cfg)
    c["magnet"]["material"] = {"type": "linear", "Jr": 0.0, "mu_rec": mu, "mu_perp": mu}
    c["magnet"]["initial"] = 0.0
    c.setdefault("coupling", {})["split"] = "none"  # the whole magnet response goes through the EMF sources
    # without splitting, plain Gauss-Jacobi WR on a mu = 4 body settles into a 2-cycle; damp the EMF update
    c["coupling"]["relax"] = 0.5
    # coil coupling only: with split = none, plain Gauss-Jacobi WR diverges on eddy loops inside a mu = 4 body
    # (strong loop-magnet coupling); S4 with eddy loops is covered on a small case in tests/test_switching.py
    c["eddy"] = {"enabled": False}
    st = prepare(c)
    S, W = run_switching(c, setup=st, log=log)
    fold = linear_fold_in(st)
    with tempfile.TemporaryDirectory(prefix="magsim_s4_") as tmp:
        d = run_circuit(st, None, Path(tmp) / "fold", L=st.L + fold["dL_sym"])
        d0 = run_circuit(st, None, Path(tmp) / "nomag", L=st.L)
    t = W["circuit"]["t"]
    i_wr = W["circuit"]["i_coil"]
    i_sp = np.interp(t, d["time"], d["I"][0])
    i_0 = np.interp(t, d0["time"], d0["I"][0])
    scale = float(np.max(np.abs(i_sp)))
    return {"max_rel_diff": float(np.max(np.abs(i_wr - i_sp)) / scale), "dL_coil_H": float(fold["dL"][0, 0]),
            "dL_over_L": float(fold["dL"][0, 0] / st.L[0, 0]), "dL_asymmetry": fold["asymmetry"],
            "no_magnet_rel_diff": float(np.max(np.abs(i_0 - i_sp)) / scale), "wr_iterations": S["wr_iterations"],
            "wr_converged": S["wr_converged"], "t_pk_wr_s": S["t_pk_s"],
            "t_pk_nomag_s": float(d0["time"][np.argmax(np.abs(d0["I"][0]))] - st.circuit["fire_time"])}


def cli_validate(scenario, out_dir=None, workers=None) -> int:
    """`magsim switch validate SCENARIO`: gates S2-S5 (+ mesh convergence if `validation.meshes`) -> validation.md."""
    cfg = load_scenario(scenario)
    out = _out_dir(cfg, out_dir)
    vcfg = cfg.get("validation") or {}
    st = prepare(cfg, log=print)
    S, W = run_switching(cfg, setup=st, log=print)
    files = write_run(st, S, W, out)
    _print_summary(S)
    tol = st.coupling["tol"]
    print("[S2] initial-guess independence run")
    ig = initial_guess_check(cfg, S, W, scale=float(vcfg.get("initial_guess_scale", 0.5)), log=print)
    print("[S4] linear magnet (mu = 4) vs inductance fold-in")
    lin = linear_magnet_check(cfg, log=print)
    mc = None
    if vcfg.get("meshes"):
        print("[mesh] convergence")
        mc = mesh_convergence(cfg, vcfg["meshes"], log=print)
    s5 = all(Path(f).exists() for f in files) and any(str(f).endswith(".png") for f in files)
    gates = [
        ("S2", "WR converges < tol within max_iter", f"{S['wr_iterations']} iterations, residual {S['wr_residual']:.2e} "
         f"(tol {tol:g})", bool(S["wr_converged"])),
        ("S2", "independent of the initial EMF guess", f"zero vs {ig['scale']} x converged EMF: max|di|/max|i| = "
         f"{ig['max_rel_diff']:.2e} ({ig['iterations']} iterations)", ig["converged"] and ig["max_rel_diff"] < 10 * tol),
        ("S3", "energy residual < 0.5 % of E0", f"{S['E_residual_rel'] * 100:.4f} % (E0 = {S['E0_J']:.3f} J)",
         abs(S["E_residual_rel"]) < 5e-3),
        ("S4", "linear magnet: WR = all-SPICE fold-in, < 0.5 %", f"max|di|/max|i| = {lin['max_rel_diff']:.2e} "
         f"(dL/L = {lin['dL_over_L']:+.4f}, no-magnet difference {lin['no_magnet_rel_diff']:.2e})",
         lin["max_rel_diff"] < 5e-3 and lin["wr_converged"]),
        ("S4", "sign: magnet along the coil field adds inductance", f"dL_coil = {lin['dL_coil_H'] * 1e9:+.3f} nH, "
         f"t_pk {lin['t_pk_nomag_s'] * 1e6:.2f} -> {lin['t_pk_wr_s'] * 1e6:.2f} us", lin["dL_coil_H"] > 0),
        ("S5", "end-to-end run writes CSV + figures", f"{len(files)} files in {out}", s5),
    ]
    lines = [f"# Validation: {cfg.get('name')}", "", f"Scenario `{scenario}`; results in `{out}`.", "",
             "| gate | check | value | pass |", "|---|---|---|---|"]
    lines += [f"| {g} | {c} | {v} | {'yes' if p else 'NO'} |" for g, c, v, p in gates]
    lines += ["", "## Key outputs", "", "```"]
    _print_summary(S, out=lines.append)
    lines += ["```", "", "## WR history", "", "| iteration | residual | samples | Newton failures |", "|---|---|---|---|"]
    lines += [f"| {h['iteration']} | {h['residual']:.3e} | {h['march_samples']} | {h['newton_failures']} |" for h in S["wr_history"]]
    if mc:
        lines += ["", "## Mesh convergence", "", "| mesh | cells | h [mm] | " + " | ".join(MESH_KEYS) + " |",
                  "|" + "---|" * (3 + len(MESH_KEYS))]
        for r in mc["rows"]:
            lines.append(f"| {r['mesh']} | {r['n_cells']} | {r['h_m'] * 1e3:.3f} | " + " | ".join(f"{r[k]:.6g}" for k in MESH_KEYS) + " |")
        if mc["richardson"]:
            lines.append("| Richardson (finest 3) | | | " + " | ".join(
                f"{mc['richardson'][k]['f_ext']:.6g} (p={mc['richardson'][k]['p']:.2f}, GCI={mc['richardson'][k]['gci_fine'] * 100:.2g} %)"
                for k in MESH_KEYS) + " |")
    (out / "validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(_jsonable({"gates": gates, "initial_guess": ig, "linear": lin,
                                                               "mesh": mc}), indent=2), encoding="utf-8")
    for g, c_, v, p in gates:
        print(f"[{g}] {'PASS' if p else 'FAIL'}  {c_}: {v}")
    print(f"validation report: {out / 'validation.md'}")
    return 0 if all(p for *_, p in gates) else 1
