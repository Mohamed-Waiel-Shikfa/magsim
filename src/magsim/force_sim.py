"""Force simulator app (task 5): scenarios, magnetisation-history protocols, outputs, sweeps, plots, CLI.

Method: nonlinear volume-integral MoM (``magsim.mom``, [chubar1998, chadebec2006, carpentier2013]) with
per-cell scalar Preisach / linear materials (``magsim.materials``, [mayergoyz1991]); force and torque by
magnetic-charge integration (``magsim.forces``, [akoun1984, allag2009]).

Scenario format (YAML, SI units; ``*temperature_C`` keys are converted to K on load)
-------------------------------------------------------------------------------------

.. code-block:: yaml

    name: r2_n42_pair                 # res_sim/<name>/ (default: file stem)
    temperature_C: 20                 # default magnet temperature (optional, default 293.15 K)
    materials:                        # optional named materials, referenced by string from magnets
      block: {grade: N42}             #   library grade + overrides (Br, mu_rec, mu_perp, Hcj, ...), or
      soft:  {type: linear, Jr: 1.2, mu_rec: 4.0, mu_perp: 4.0}   # LinearMaterial (mu_perp defaults to mu_rec)
    magnets:
      - name: A
        shape: {type: cuboid, dims: [0.02, 0.01, 0.005]}  # | {type: cylinder, diameter, height}
                                                          # | {type: ring, d_inner, d_outer, height}
        material: block                 # name from `materials`, or an inline material mapping
        pose: {position: [0, 0, -0.0025], rotation: {euler_xyz_deg: [0, 0, 0]}}  # or rotvec_deg
        easy_axis: [0, 0, 1]            # body frame (default +z)
        mesh: {n: [6, 3, 2], grading: 1.0, q: 2}             # or [nx, ny, nz]
        initial: +sat                   # +sat | -sat | thermal | float in [-1, 1]
        temperature_C: 20               # optional per magnet
      - {name: B, ...}
    layout:                           # optional: places `body` relative to `reference` (overrides its pose)
      body: B
      reference: A
      gap: 2.0e-3                     # [m] clearance between the planes z = max_z(A) and z = min_z(B)
      offset: [0.0, 0.0]              # [m] (x, y) of B's centroid minus A's centroid
      tilt_deg: 0.0                   # [deg] rotation of B about `tilt_axis` (global), after B's pose rotation
      tilt_axis: [0, 1, 0]
    target: B                         # body whose force / torque is reported (from all other bodies)
    pivot: centroid                   # or [x, y, z] in m (global)
    protocol: [isolate, place, measure]          # default when absent (see below)
    outputs: [Fz, J_mean, converged]  # optional subset (default: all, see below)
    solver: {tol: 1.0e-9, max_iter: 60}          # optional MagnetSystem.solve options
    force:  {quad_tol: 1.0e-8}                   # optional force_torque_detail options
    sweep:  {axes: [{path: layout.gap, linspace: [2.0e-3, 1.0e-2, 9]}], plot: {y: [Fz]}}
    sweeps: [{name: gap, axes: [...]}, ...]      # several named sweeps (res_sim/<name>/<sweep name>/)

``layout`` stacks along global +z: ``gap`` is the distance between A's highest point and B's lowest
point (support functions of the posed shapes), so any gap >= 0 is overlap-free at any tilt / offset.

Protocol (ordered; every step except ``measure``'s report solves the self-consistent state at the current
poses and COMMITS it into the material history, so the order matters for Alnico):

- ``isolate``: all bodies moved 1e3 x (largest body size) apart along x (rotations kept), solved, committed:
  the open-circuit excursion after magnetising (neighbour fields ~1e-9 of the self field).
- ``place``: every body set to its scenario pose (with ``layout``), solved, committed.
- ``move: {name, position: [..] | rotation: {..} | delta: [..] | gap: g}``: one body moved (absolute
  position [m], absolute rotation, relative translation [m], or re-layout at gap g [m] for the layout body),
  solved, committed.
- ``measure``: solved at the current poses, outputs recorded, committed.
- ``separate``: the target moved 1e3 x size away from the volume centroid of the other bodies (pull-off),
  solved, committed.

Outputs (scalars): ``Fx, Fy, Fz`` [N] and ``Tx, Ty, Tz`` [N m] on the target about the pivot; per body
``J_mean_<name>`` (volume mean of J.e, T), ``J_min_<name>`` (minimum over cells of J.e, T),
``H_par_mean_<name>`` (volume mean of H.e, A/m); solver ``iterations``, ``residual`` [T] and ``converged``
(1/0) of the measure solve, and ``all_converged`` over every protocol solve. With several ``measure`` steps
the k-th one's outputs are also given with suffix ``_m<k>`` (1-based); the unsuffixed values are the last.
In ``outputs:``, a per-body name without ``_<body>`` selects it for every body.
"""

from __future__ import annotations

import json
import math
import re
import sys
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from magsim.config import load_scenario, set_path
from magsim.geometry import CUBOID, CYLINDER, SEGMENT, Cuboid, Cylinder, Magnet, Pose, Ring
from magsim.materials.base import LinearMaterial
from magsim.materials.library import get_material, list_grades
from magsim.mom import MagnetSystem, Solution
from magsim.sweep import run_sweep, write_provenance

__all__ = [
    "ScenarioError", "ForceScenario", "run_scenario", "run_protocol", "build_magnets", "scenario_poses",
    "body_outputs", "plot_sweep", "plot_j_map", "cli_run", "cli_sweep", "cli_validate", "FORCE_KEYS",
    "DEFAULT_PROTOCOL", "FAR_FACTOR",
]

FORCE_KEYS = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
BODY_KEYS = ("J_mean", "J_min", "H_par_mean")
SOLVER_KEYS = ("iterations", "residual", "converged", "all_converged")
DEFAULT_PROTOCOL = ("isolate", "place", "measure")
FAR_FACTOR = 1e3  # isolation / pull-off distance in units of the largest body size
_STEPS = {"isolate", "place", "move", "measure", "separate"}
_TOP_KEYS = {"name", "description", "temperature", "materials", "magnets", "layout", "target", "pivot",
             "protocol", "outputs", "solver", "force", "sweep", "sweeps", "validation", "plot", "notes"}
RESULTS = Path("res_sim")


class ScenarioError(ValueError):
    """Malformed scenario (message names the offending key)."""


# ---------------------------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------------------------
def _need(d: dict, key: str, where: str):
    if not isinstance(d, dict) or key not in d:
        raise ScenarioError(f"{where}: missing key {key!r}")
    return d[key]


def _num(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ScenarioError(f"{where}: expected a finite number, got {v!r}")
    return float(v)


def _vec(v, n: int, where: str) -> np.ndarray:
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise ScenarioError(f"{where}: expected a list of {n} numbers, got {v!r}")
    return np.array([_num(x, f"{where}[{i}]") for i, x in enumerate(v)])


def build_shape(d: dict, where: str = "shape"):
    """Cuboid / Cylinder / Ring from ``{type: cuboid, dims}`` | ``{type: cylinder, diameter, height}`` |
    ``{type: ring, d_inner, d_outer, height}`` (m)."""
    kind = _need(d, "type", where)
    try:
        if kind == "cuboid":
            return Cuboid(tuple(_vec(_need(d, "dims", where), 3, f"{where}.dims")))
        if kind == "cylinder":
            return Cylinder(_num(_need(d, "diameter", where), f"{where}.diameter"),
                            _num(_need(d, "height", where), f"{where}.height"))
        if kind == "ring":
            return Ring(_num(_need(d, "d_inner", where), f"{where}.d_inner"),
                        _num(_need(d, "d_outer", where), f"{where}.d_outer"),
                        _num(_need(d, "height", where), f"{where}.height"))
    except ScenarioError:
        raise
    except (ValueError, TypeError) as exc:
        raise ScenarioError(f"{where}: {exc}") from exc
    raise ScenarioError(f"{where}.type: expected cuboid | cylinder | ring, got {kind!r}")


def build_material(d, where: str = "material", named: dict | None = None):
    """Material from a mapping (``{grade: <library name>, <overrides>}`` or ``{type: linear, Jr, mu_rec,
    mu_perp}``) or a string naming an entry of the scenario's ``materials``.

    Grade overrides are those of ``magsim.materials.library.get_material`` plus ``Br_scale`` (multiplies the
    library Br). Linear: ``mu_rec`` defaults to 1 (rigid) and ``mu_perp`` to ``mu_rec`` (isotropic).
    """
    if isinstance(d, str):
        if not named or d not in named:
            raise ScenarioError(f"{where}: material {d!r} not defined under 'materials' (have {sorted(named or {})})")
        return build_material(named[d], f"materials.{d}", None)
    if not isinstance(d, dict):
        raise ScenarioError(f"{where}: expected a mapping or a material name, got {d!r}")
    if d.get("type", "grade") == "linear":
        extra = set(d) - {"type", "Jr", "mu_rec", "mu_perp", "name"}
        if extra:
            raise ScenarioError(f"{where}: unknown key(s) {sorted(extra)} for a linear material")
        mu = _num(d.get("mu_rec", 1.0), f"{where}.mu_rec")
        try:
            return LinearMaterial(_num(_need(d, "Jr", where), f"{where}.Jr"), mu,
                                  _num(d.get("mu_perp", mu), f"{where}.mu_perp"), name=str(d.get("name", "linear")))
        except ValueError as exc:
            raise ScenarioError(f"{where}: {exc}") from exc
    if "type" in d and d["type"] != "grade":
        raise ScenarioError(f"{where}.type: expected 'linear' or a 'grade' entry, got {d['type']!r}")
    grade = _need(d, "grade", where)
    if grade not in list_grades():
        raise ScenarioError(f"{where}.grade: unknown grade {grade!r}; see magsim.materials.library.list_grades()")
    over = {k: v for k, v in d.items() if k not in ("grade", "type", "Br_scale")}
    try:
        if "Br_scale" in d:
            if "Br" in over:
                raise ScenarioError(f"{where}: give Br or Br_scale, not both")
            over["Br"] = get_material(grade).Br * _num(d["Br_scale"], f"{where}.Br_scale")
        return get_material(grade, **over)
    except ScenarioError:
        raise
    except (ValueError, TypeError, KeyError) as exc:
        raise ScenarioError(f"{where}: {exc}") from exc


def build_magnets(cfg: dict) -> list[Magnet]:
    """Magnets of a scenario at their (pre-layout) scenario poses."""
    mags = _need(cfg, "magnets", "scenario")
    if not isinstance(mags, list) or not mags:
        raise ScenarioError("magnets: expected a non-empty list")
    T0 = _num(cfg.get("temperature", 293.15), "temperature")
    named = cfg.get("materials") or {}
    out, names = [], set()
    for i, m in enumerate(mags):
        w = f"magnets[{i}]"
        if not isinstance(m, dict):
            raise ScenarioError(f"{w}: expected a mapping")
        extra = set(m) - {"name", "shape", "material", "pose", "easy_axis", "mesh", "initial", "temperature"}
        if extra:
            raise ScenarioError(f"{w}: unknown key(s) {sorted(extra)}")
        name = str(_need(m, "name", w))
        if name in names:
            raise ScenarioError(f"{w}.name: duplicate magnet name {name!r}")
        names.add(name)
        shape = build_shape(_need(m, "shape", w), f"{w}.shape")
        mat = build_material(_need(m, "material", w), f"{w}.material", named)
        try:
            pose = Pose.from_dict(m.get("pose"))
            out.append(Magnet(name, shape, pose, mat, easy_axis=m.get("easy_axis", (0.0, 0.0, 1.0)),
                              mesh=m.get("mesh"), initial=m.get("initial", "+sat"),
                              temperature=_num(m.get("temperature", T0), f"{w}.temperature")))
        except ScenarioError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise ScenarioError(f"{w}: {exc}") from exc
    return out


def _support(shape, R: np.ndarray, u) -> float:
    """Half-extent of ``shape`` rotated by R along the global unit vector u [m] (support function)."""
    u = np.asarray(u, dtype=float)
    ub = R.T @ u  # u in the body frame
    if isinstance(shape, Cuboid):
        return float(0.5 * np.sum(np.abs(ub) * np.asarray(shape.dims)))
    c = min(1.0, abs(ub[2]))
    return float(0.5 * shape.height * c + shape.r_out * math.sqrt(max(0.0, 1.0 - c * c)))


def layout_pose(layout: dict, magnets: dict[str, Magnet], gap: float | None = None,
                body_rotation: Rotation | None = None) -> Pose:
    """Pose of ``layout.body`` stacked on ``layout.reference`` (at its current pose) along +z (module docstring).

    ``gap`` overrides ``layout.gap``; ``body_rotation`` is the body's scenario rotation before the tilt
    (default: its current pose rotation).
    """
    w = "layout"
    body = magnets.get(str(_need(layout, "body", w)))
    ref = magnets.get(str(_need(layout, "reference", w)))
    if body is None or ref is None:
        raise ScenarioError(f"layout: body / reference must name magnets (have {sorted(magnets)})")
    if body is ref:
        raise ScenarioError("layout: body and reference must differ")
    extra = set(layout) - {"body", "reference", "gap", "offset", "tilt_deg", "tilt_axis"}
    if extra:
        raise ScenarioError(f"layout: unknown key(s) {sorted(extra)}")
    g = _num(layout.get("gap", 0.0) if gap is None else gap, "layout.gap")
    if g < 0:
        raise ScenarioError(f"layout.gap must be >= 0, got {g}")
    off = _vec(layout.get("offset", [0.0, 0.0]), 2, "layout.offset")
    tilt = _num(layout.get("tilt_deg", 0.0), "layout.tilt_deg")
    axis = _vec(layout.get("tilt_axis", [0.0, 1.0, 0.0]), 3, "layout.tilt_axis")
    if np.linalg.norm(axis) == 0:
        raise ScenarioError("layout.tilt_axis must be non-zero")
    R0 = body.pose.rotation if body_rotation is None else body_rotation
    R = Rotation.from_rotvec(np.deg2rad(tilt) * axis / np.linalg.norm(axis)) * R0
    ez = np.array([0.0, 0.0, 1.0])
    pa = ref.pose.position
    z = pa[2] + _support(ref.shape, ref.pose.matrix, ez) + g + _support(body.shape, R.as_matrix(), ez)
    return Pose(np.array([pa[0] + off[0], pa[1] + off[1], z]), R)


def scenario_poses(cfg: dict, magnets: Iterable[Magnet]) -> dict[str, Pose]:
    """Scenario poses (``layout`` applied) of every magnet, keyed by name."""
    base = {m.name: m for m in magnets}
    poses = {n: m.pose for n, m in base.items()}
    if cfg.get("layout"):
        poses[str(cfg["layout"]["body"])] = layout_pose(cfg["layout"], base)
    return poses


def _protocol(cfg: dict) -> list[tuple[str, dict]]:
    steps = cfg.get("protocol")
    if steps is None:
        steps = list(DEFAULT_PROTOCOL)
    if not isinstance(steps, list) or not steps:
        raise ScenarioError("protocol: expected a non-empty list of steps")
    out = []
    for i, s in enumerate(steps):
        if isinstance(s, str):
            kind, args = s, {}
        elif isinstance(s, dict) and len(s) == 1:
            kind, args = next(iter(s.items()))
            args = args or {}
        else:
            raise ScenarioError(f"protocol[{i}]: expected a step name or a one-key mapping, got {s!r}")
        if kind not in _STEPS:
            raise ScenarioError(f"protocol[{i}]: unknown step {kind!r} (allowed: {sorted(_STEPS)})")
        if kind == "move":
            if not isinstance(args, dict) or "name" not in args:
                raise ScenarioError(f"protocol[{i}].move: needs 'name'")
            keys = set(args) - {"name"}
            if not keys or keys - {"position", "rotation", "delta", "gap"}:
                raise ScenarioError(f"protocol[{i}].move: give position, rotation, delta or gap, got {sorted(keys)}")
            if "gap" in keys and keys != {"gap"}:
                raise ScenarioError(f"protocol[{i}].move: 'gap' cannot be combined with other keys")
            if {"position", "delta"} <= keys:
                raise ScenarioError(f"protocol[{i}].move: give position or delta, not both")
        elif args:
            raise ScenarioError(f"protocol[{i}]: step {kind!r} takes no arguments")
        out.append((kind, args))
    return out


def validate_scenario(cfg: dict) -> None:
    """Raise ScenarioError on structural problems (unknown keys, bad protocol, target, outputs, layout)."""
    if not isinstance(cfg, dict):
        raise ScenarioError("scenario must be a mapping")
    extra = set(cfg) - _TOP_KEYS
    if extra:
        raise ScenarioError(f"unknown top-level key(s) {sorted(extra)}")
    mags = build_magnets(cfg)
    names = [m.name for m in mags]
    tgt = cfg.get("target", names[-1])
    if str(tgt) not in names:
        raise ScenarioError(f"target: {tgt!r} is not a magnet (have {names})")
    _protocol(cfg)
    if cfg.get("layout"):
        layout_pose(cfg["layout"], {m.name: m for m in mags})
    _output_names(cfg, names)
    piv = cfg.get("pivot", "centroid")
    if piv not in (None, "centroid"):
        _vec(piv, 3, "pivot")


def _output_names(cfg: dict, names: list[str]) -> list[str]:
    full = list(FORCE_KEYS) + [f"{k}_{n}" for k in BODY_KEYS for n in names] + list(SOLVER_KEYS)
    req = cfg.get("outputs")
    if req is None:
        return full
    if not isinstance(req, list):
        raise ScenarioError("outputs: expected a list of names")
    out = []
    for r in req:
        if r in BODY_KEYS:
            out += [f"{r}_{n}" for n in names]
        elif r in full:
            out.append(r)
        else:
            raise ScenarioError(f"outputs: unknown output {r!r} (allowed: {full} or {list(BODY_KEYS)})")
    return out


# ---------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------
def body_outputs(system: MagnetSystem, sol: Solution) -> dict[str, float]:
    """Per-body volume-mean J.e [T], min-over-cells J.e [T], volume-mean H.e [A/m]."""
    out = {}
    Jp = np.einsum("ij,ij->i", sol.J, system.easy)
    for b, m in enumerate(system.magnets):
        sl = system.body_slice(b)
        V = system.volume[sl]
        out[f"J_mean_{m.name}"] = float((Jp[sl] * V).sum() / V.sum())
        out[f"J_min_{m.name}"] = float(Jp[sl].min())
        out[f"H_par_mean_{m.name}"] = float((sol.H_par[sl] * V).sum() / V.sum())
    return out


class ForceScenario:
    """A scenario's magnets in one ``MagnetSystem`` plus the protocol operations (module docstring)."""

    def __init__(self, cfg: dict):
        validate_scenario(cfg)
        self.cfg = cfg
        self.magnets = build_magnets(cfg)
        self.base = {m.name: m.pose for m in self.magnets}
        self.place_poses = scenario_poses(cfg, self.magnets)
        for m in self.magnets:
            m.pose = self.place_poses[m.name]
        try:
            self.system = MagnetSystem(self.magnets)
        except (ValueError, MemoryError) as exc:
            raise ScenarioError(f"magnets: {exc}") from exc
        self.names = [m.name for m in self.magnets]
        self.target = str(cfg.get("target", self.names[-1]))
        piv = cfg.get("pivot", "centroid")
        self.pivot_cfg = None if piv in (None, "centroid") else np.asarray(piv, dtype=float)
        self.solver_opts = dict(cfg.get("solver") or {})
        self.force_opts = dict(cfg.get("force") or {})
        self.far = FAR_FACTOR * max(m.shape.size for m in self.magnets)
        self.records: list[dict] = []
        self.last: Solution | None = None

    # -- primitives
    def solve(self, kind: str) -> Solution:
        sol = self.system.solve(**self.solver_opts)
        if not sol.converged:
            warnings.warn(f"{kind}: MoM solve not converged (residual {sol.residual:.2e} T)", RuntimeWarning,
                          stacklevel=2)
        self.records.append({"step": len(self.records), "kind": kind, "iterations": sol.iterations,
                             "residual": sol.residual, "converged": bool(sol.converged)})
        self.last = sol
        return sol

    def solve_commit(self, kind: str) -> Solution:
        sol = self.solve(kind)
        self.system.commit(sol)
        return sol

    # -- protocol steps
    def isolate(self):
        poses = {}
        for i, m in enumerate(self.magnets):
            poses[m.name] = Pose(np.array([i * self.far, 0.0, 0.0]), m.pose.rotation)
        self.system.set_poses(poses)
        return self.solve_commit("isolate")

    def place(self):
        self.system.set_poses(dict(self.place_poses))
        return self.solve_commit("place")

    def move(self, args: dict):
        name = str(args["name"])
        if name not in self.names:
            raise ScenarioError(f"move: unknown magnet {name!r}")
        m = self.magnets[self.names.index(name)]
        if "gap" in args:
            lay = self.cfg.get("layout")
            if not lay or str(lay["body"]) != name:
                raise ScenarioError(f"move: 'gap' needs a layout whose body is {name!r}")
            mags = {mm.name: mm for mm in self.magnets}
            pose = layout_pose(lay, mags, gap=float(args["gap"]), body_rotation=self.base[name].rotation)
        else:
            pos = m.pose.position
            rot = m.pose.rotation
            if "rotation" in args:
                rot = Pose.from_dict({"rotation": args["rotation"]}).rotation
            if "position" in args:
                pos = _vec(args["position"], 3, "move.position")
            if "delta" in args:
                pos = pos + _vec(args["delta"], 3, "move.delta")
            pose = Pose(pos, rot)
        self.system.set_pose(name, pose)
        return self.solve_commit("move")

    def separate(self):
        b = self.names.index(self.target)
        others = [i for i in range(len(self.magnets)) if i != b]
        if not others:
            d = np.array([0.0, 0.0, 1.0])
        else:
            V = np.array([self.magnets[i].shape.volume for i in others])
            c = (np.array([self.magnets[i].pose.position for i in others]) * V[:, None]).sum(0) / V.sum()
            d = self.magnets[b].pose.position - c
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 0 else np.array([0.0, 0.0, 1.0])
        self.system.set_pose(self.target, Pose(self.magnets[b].pose.position + self.far * d,
                                               self.magnets[b].pose.rotation))
        return self.solve_commit("separate")

    def measure(self) -> dict[str, float]:
        from magsim.forces import force_torque_detail

        sol = self.solve("measure")
        fr = force_torque_detail(self.system, sol, self.target, self.pivot_cfg, **self.force_opts)
        out = dict(zip(FORCE_KEYS, [*map(float, fr.F), *map(float, fr.T)]))
        out.update(body_outputs(self.system, sol))
        out.update(iterations=float(sol.iterations), residual=float(sol.residual), converged=float(sol.converged))
        self.records[-1].update({k: out[k] for k in FORCE_KEYS})
        self.system.commit(sol)
        return out

    def run(self) -> dict[str, float]:
        """Execute the protocol; return the scalar outputs (module docstring)."""
        measures = []
        for kind, args in _protocol(self.cfg):
            if kind == "isolate":
                self.isolate()
            elif kind == "place":
                self.place()
            elif kind == "move":
                self.move(args)
            elif kind == "separate":
                self.separate()
            else:
                measures.append(self.measure())
        if not measures:
            raise ScenarioError("protocol: needs at least one 'measure' step to report outputs")
        allc = float(all(r["converged"] for r in self.records))
        wanted = _output_names(self.cfg, self.names)
        res = {}
        for k, mo in enumerate(measures, start=1):
            mo["all_converged"] = allc
            if len(measures) > 1:
                res.update({f"{n}_m{k}": mo[n] for n in wanted})
        res.update({n: measures[-1][n] for n in wanted})
        return res


def run_protocol(cfg: dict) -> tuple[dict[str, float], ForceScenario]:
    """Run a scenario; return (outputs, the ForceScenario with its system, last solution and records)."""
    fs = ForceScenario(cfg)
    return fs.run(), fs


def run_scenario(cfg: dict) -> dict[str, float]:
    """Scalar outputs of one scenario configuration (top-level and picklable, for ``run_sweep``)."""
    return run_protocol(cfg)[0]


# ---------------------------------------------------------------------------------------------
# display helpers (SI in, natural display units out)
# ---------------------------------------------------------------------------------------------
_LENGTH_WORDS = ("gap", "offset", "position", "delta", "dims", "diameter", "height", "d_inner", "d_outer")


def axis_display(path: str) -> tuple[str, float, float]:
    """(label, scale, shift) for a sweep path: displayed = value * scale + shift."""
    p = path.lower()
    last = p.split(".")[-1]
    comp = ""
    if last.endswith("]"):
        idx = int(last[last.rindex("[") + 1:-1])
        last = last[: last.rindex("[")]
        comp = " " + "xyz"[idx] if idx < 3 else f"[{idx}]"
    if last == "gap":
        return "gap (mm)", 1e3, 0.0
    if last == "offset":
        return f"lateral offset{comp} (mm)", 1e3, 0.0
    if last in _LENGTH_WORDS:
        return f"{last.replace('_', ' ')}{comp} (mm)", 1e3, 0.0
    if last == "tilt_deg":
        return "tilt angle (deg)", 1.0, 0.0
    if last == "temperature":
        return "temperature (°C)", 1.0, -273.15
    if last in ("br", "jr"):
        return f"{'$B_r$' if last == 'br' else '$J_r$'} (T)", 1.0, 0.0
    if last == "br_scale":
        return "$B_r$ scale (-)", 1.0, 0.0
    if last == "grade":
        return "grade", 1.0, 0.0
    return path, 1.0, 0.0


def output_display(name: str, target: str = "B") -> tuple[str, float]:
    """(label, scale) for an output column."""
    base = name.split("_m")[0] if "_m" in name and name.rsplit("_m", 1)[-1].isdigit() else name
    if base in ("Fx", "Fy", "Fz"):
        return f"$F_{base[1]}$ on {target} (N)", 1.0
    if base in ("Tx", "Ty", "Tz"):
        return f"$T_{base[1]}$ on {target} (mN m)", 1e3
    for k, lab, sc in (("J_mean_", "mean $J\\cdot\\hat e$", 1e3), ("J_min_", "min $J\\cdot\\hat e$", 1e3),
                       ("H_par_mean_", "mean $H\\cdot\\hat e$", 1e-3)):
        if base.startswith(k):
            unit = "mT" if sc == 1e3 else "kA/m"
            return f"{lab}, {base[len(k):]} ({unit})", sc
    return name, 1.0


def _is_categorical(vals) -> bool:
    return any(isinstance(v, str) for v in vals)


# ---------------------------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------------------------
def plot_sweep(df: pd.DataFrame, axes: list[dict], out_stem: Path, ys: list[str] | None = None,
               target: str = "B") -> list[Path]:
    """Standard sweep figures: line(s) vs the numeric axis (hue = second axis when categorical or short),
    or a heatmap + contours for two numeric axes. Returns the written file paths."""
    import matplotlib.pyplot as plt

    from magsim import plotting

    plotting.use_style()
    paths = [a["path"] for a in axes]
    cat = [p for p in paths if _is_categorical(df[p].tolist())]
    num = [p for p in paths if p not in cat]
    ys = ys or ["Fz"]
    ok = df[df["error"].isna()] if "error" in df else df
    written = []
    if len(num) == 1 and len(cat) <= 1:
        xl, xs, xo = axis_display(num[0])
        for y in ys:
            yl, ysc = output_display(y, target)
            fig, ax = plt.subplots()
            d = ok.copy()
            d["_x"] = d[num[0]].astype(float) * xs + xo
            d["_y"] = d[y].astype(float) * ysc
            hue = cat[0] if cat else None
            if hue:
                order = list(dict.fromkeys(df[hue].tolist()))
                for i, lvl in enumerate(order):
                    sub = d[d[hue] == lvl].sort_values("_x")
                    ax.plot(sub["_x"], sub["_y"], marker="o", color=plotting.OKABE_ITO[i % 8], label=str(lvl))
                ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
            else:
                d = d.sort_values("_x")
                ax.plot(d["_x"], d["_y"], marker="o", color=plotting.OKABE_ITO[5])
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.axhline(0.0, color="0.7", lw=0.5, zorder=0)
            written += list(plotting.save(fig, f"{out_stem}_{y}"))
            plt.close(fig)
    elif len(num) == 2 and not cat:
        (xl, xs, xo), (yl_, ys_, yo_) = axis_display(num[0]), axis_display(num[1])
        for y in ys:
            zl, zsc = output_display(y, target)
            d = ok.copy()
            d["_x"] = d[num[0]].astype(float) * xs + xo
            d["_y"] = d[num[1]].astype(float) * ys_ + yo_
            d["_z"] = d[y].astype(float) * zsc
            fig, ax = plt.subplots()
            plotting.heatmap(d, "_x", "_y", "_z", ax=ax, xlabel=xl, ylabel=yl_, zlabel=zl)
            written += list(plotting.save(fig, f"{out_stem}_{y}_map"))
            plt.close(fig)
    else:
        warnings.warn(f"no standard plot for axes {paths} (need 1 numeric [+1 categorical] or 2 numeric)",
                      stacklevel=2)
    return written


def _cell_sections(system: MagnetSystem, b: int, p0, n, uv, wv):
    """Section polygons (k, 4..8, 2) [m] in (u, w) and the cell indices, for body b (see plot_j_map)."""
    m = system.magnets[b]
    bm = m.body_mesh
    cells = np.arange(system.n_cells)[system.body_slice(b)]
    R = m.pose.matrix
    polys, idx = [], []
    if np.all(bm.kind == CUBOID):
        cen = m.pose.apply(bm.centroid)
        dist = np.abs((cen - p0) @ n)
        near = np.nonzero(dist <= dist.min() + 1e-6 * m.shape.size)[0]
        corners = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])
        for c in near:
            v = m.pose.apply(bm.pos[c] + corners * bm.dims[c, :3])
            uv2 = np.stack([(v - p0) @ uv, (v - p0) @ wv + p0 @ wv], 1)
            polys.append(uv2[ConvexHull(uv2).vertices])
            idx.append(cells[c])
        return polys, idx
    axis = R[:, 2]
    rad = uv - (uv @ axis) * axis
    rad = rad / np.linalg.norm(rad) if np.linalg.norm(rad) > 1e-9 else R[:, 0]
    rad_b = R.T @ rad
    ang = math.atan2(rad_b[1], rad_b[0]) % (2 * np.pi)
    for c in range(bm.kind.size):
        r1, r2, h, ph1, ph2 = bm.dims[c]
        zc = bm.pos[c, 2]
        for sgn, a in ((1.0, ang), (-1.0, (ang + np.pi) % (2 * np.pi))):
            if bm.kind[c] == SEGMENT and not (ph1 <= a < ph2 or ph1 <= a + 2 * np.pi < ph2):
                continue
            if bm.kind[c] == CYLINDER and sgn < 0:  # core: one polygon spanning both sides
                continue
            lo = -r2 if bm.kind[c] == CYLINDER else r1
            rr = np.array([lo, r2, r2, lo]) * (1.0 if bm.kind[c] == CYLINDER else sgn)
            zz = zc + np.array([-.5, -.5, .5, .5]) * h
            pts = m.pose.position + rr[:, None] * rad[None, :] + (zz[:, None] * axis[None, :])
            polys.append(np.stack([(pts - p0) @ uv, (pts - p0) @ wv + p0 @ wv], 1))
            idx.append(cells[c])
    return polys, idx


def plot_j_map(system: MagnetSystem, sol: Solution, out_stem: Path, normal=(0.0, 1.0, 0.0), point=None) -> list[Path]:
    """Per-body J map: the cells cut by a section plane, coloured by J.e (mT); axes in mm.

    Section plane through ``point`` (default: mean cell centroid) with unit ``normal``. Cuboid bodies: the layer
    of cells whose centroids are nearest the plane, drawn as their projection onto it. Cylinders / rings: the
    meridional section in the plane through the body axis parallel to the in-plane horizontal direction
    (full rings appear on both sides of the axis). Vertical axis: global z for a horizontal normal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from magsim import plotting

    plotting.use_style()
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    p0 = system.centroid.mean(0) if point is None else np.asarray(point, dtype=float)
    wv = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    wv = wv - (wv @ n) * n
    wv /= np.linalg.norm(wv)
    uv = np.cross(n, wv)
    Jp = np.einsum("ij,ij->i", sol.J, system.easy)
    polys, idx = [], []
    for b in range(len(system.magnets)):
        pb, ib = _cell_sections(system, b, p0, n, uv, wv)
        polys += pb
        idx += ib
    allp = np.concatenate(polys)
    aspect = max(np.ptp(allp[:, 1]), 1e-12) / max(np.ptp(allp[:, 0]), 1e-12)
    if aspect >= 1.0:  # axes box size [in] following the data aspect
        ax_h, ax_w = 2.4, max(2.4 / aspect, 0.9)
    else:
        ax_w, ax_h = 2.25, max(2.25 * aspect, 1.0)
    fig, ax = plt.subplots(figsize=(ax_w + 1.3, ax_h + 0.6))
    pc = PolyCollection([p * 1e3 for p in polys], array=Jp[np.array(idx)] * 1e3, cmap="viridis",
                        edgecolors="face", linewidths=0.2)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=4 if ax_w > 1.5 else 2))

    def _name(v):
        i = int(np.argmax(np.abs(v)))
        sign = "-" if v[i] < 0 else ""
        return f"${sign}{'xyz'[i]}$ (mm)" if abs(v[i]) > 0.99 else "in-plane position (mm)"

    ax.set_xlabel(_name(uv))
    ax.set_ylabel(_name(wv))
    cb = fig.colorbar(pc, ax=ax)
    cb.set_label("$J\\cdot\\hat e$ (mT)")
    cb.outline.set_linewidth(0.6)
    files = list(plotting.save(fig, out_stem))
    plt.close(fig)
    return files


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def _load(scenario: Path) -> dict:
    scenario = Path(scenario)
    if not scenario.is_file():
        raise ScenarioError(f"scenario file {scenario} not found")
    try:
        cfg = load_scenario(scenario)
    except (yaml.YAMLError, ValueError) as exc:
        raise ScenarioError(f"{scenario}: {exc}") from exc
    cfg.setdefault("name", scenario.stem)
    return cfg


def _out(cfg: dict, out_dir) -> Path:
    return Path(out_dir) if out_dir is not None else RESULTS / str(cfg["name"])


def _guard(fn):
    def wrapped(scenario, out_dir=None, workers=None):
        try:
            return fn(Path(scenario), out_dir, workers)
        except ScenarioError as exc:
            print(f"magsim force: {exc}", file=sys.stderr)
            return 2
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


@_guard
def cli_run(scenario: Path, out_dir=None, workers=None) -> int:
    """`magsim force run`: one configuration -> res_sim/<name>/run.json, protocol.csv, J map, provenance."""
    cfg = _load(scenario)
    out, fs = run_protocol(cfg)
    d = _out(cfg, out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.DataFrame(fs.records).to_csv(d / "protocol.csv", index=False)
    with open(d / "scenario.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    write_provenance(d)
    plot_j_map(fs.system, fs.last, d / "j_map")
    for k, v in out.items():
        if not re.search(r"_m\d+$", k):
            unit = {"F": "N", "T": "N m"}.get(k[0], "") if k in FORCE_KEYS else ""
            print(f"{k} = {v:+.6g} {unit}".rstrip())
    print(f"wrote {d}")
    return 0 if out.get("all_converged", 1.0) == 1.0 else 1


def _sweeps(cfg: dict) -> list[dict]:
    if "sweeps" in cfg and "sweep" in cfg:
        raise ScenarioError("give 'sweep' or 'sweeps', not both")
    if "sweep" in cfg:
        sw = [dict(cfg["sweep"], name=cfg["sweep"].get("name"))]
    elif "sweeps" in cfg:
        sw = list(cfg["sweeps"])
        for i, s in enumerate(sw):
            if not isinstance(s, dict) or "name" not in s:
                raise ScenarioError(f"sweeps[{i}]: needs a 'name'")
    else:
        raise ScenarioError("no 'sweep' (or 'sweeps') block in the scenario")
    for s in sw:
        if not s.get("axes"):
            raise ScenarioError(f"sweep {s.get('name')!r}: needs 'axes'")
    return sw


@_guard
def cli_sweep(scenario: Path, out_dir=None, workers=None) -> int:
    """`magsim force sweep`: every sweep of the scenario -> CSV + provenance + standard plots + J map."""
    cfg = _load(scenario)
    sweeps = _sweeps(cfg)
    base = {k: v for k, v in cfg.items() if k not in ("sweep", "sweeps")}
    validate_scenario(base)
    root = _out(cfg, out_dir)
    nw = min(4, int(workers or 4))
    bad = 0
    for s in sweeps:
        for ax in s["axes"]:
            set_path(base, ax["path"], None)  # validates the path exists (value replaced on a copy)
        sub = root / s["name"] if s.get("name") else root
        df = run_sweep(base, s["axes"], run_scenario, workers=int(s.get("workers", nw)), out_dir=sub)
        ys = (s.get("plot") or {}).get("y") or ["Fz"]
        plot_sweep(df, s["axes"], sub / "sweep", ys=ys, target=str(cfg.get("target", base["magnets"][-1]["name"])))
        nerr = int(df["error"].notna().sum())
        bad += nerr
        cols = [a["path"] for a in s["axes"]] + [y for y in ys if y in df]
        print(f"[{s.get('name') or cfg['name']}] {len(df)} points, {nerr} errors -> {sub}")
        with pd.option_context("display.width", 160, "display.max_rows", 200):
            print(df[cols].to_string(index=False))
    _, fs = run_protocol(base)
    plot_j_map(fs.system, fs.last, root / "j_map")
    return 0 if bad == 0 else 1


@_guard
def cli_validate(scenario: Path, out_dir=None, workers=None) -> int:
    """`magsim force validate`: the scenario's validation checks -> res_sim/<name>/validation.md."""
    from magsim.validation import run_validation

    cfg = _load(scenario)
    d = _out(cfg, out_dir)
    ok = run_validation(cfg, d, workers=min(4, int(workers or 4)), scenario_path=scenario)
    print(f"wrote {d / 'validation.md'}")
    return 0 if ok else 1
