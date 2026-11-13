"""Scenario YAML helpers for the UI: parse, dump, read and patch the common "quick parameters".

All values are SI in the YAML (gap, offset in m; temperature in K once patched). The UI converts for
display (mm, degC). Patch functions return a patched deep copy and only touch fields that exist, except
where noted (`magnet.temperature` is written, replacing `temperature_C`).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from magsim.config import KELVIN_OFFSET
from magsim.ui import ROOT, SCENARIOS

SCR_MODELS = {
    "latching": "res_exp/circuits/magnetiser/magnetiser_regen.lib",
    "original": "res_exp/circuits/magnetiser/magnetiser.lib",
}
COIL_REFERENCES = ("r1", "r2", "r3")


def scenario_files(sim: str) -> list[Path]:
    """scenarios/force_*.yaml or scenarios/switch_*.yaml, sorted."""
    prefix = {"force": "force_", "switch": "switch_"}[sim]
    return sorted(SCENARIOS.glob(f"{prefix}*.yaml"))


def parse(text: str) -> tuple[dict | None, str | None]:
    """(mapping, None) or (None, error message)."""
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, str(exc)
    if not isinstance(cfg, dict):
        return None, "the top level of a scenario must be a mapping"
    return cfg, None


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=110, allow_unicode=True)


def scenario_name(cfg: dict | None, fallback: str) -> str:
    return str((cfg or {}).get("name") or fallback)


def has_sweep(cfg: dict | None) -> bool:
    return bool(cfg) and bool(cfg.get("sweep") or cfg.get("sweeps"))


def has_validation(cfg: dict | None) -> bool:
    return bool(cfg) and bool(cfg.get("validation"))


def parse_mesh(text: str) -> list[int]:
    """"8, 4, 2" or "[8, 4, 2]" -> [8, 4, 2] (positive integers)."""
    try:
        n = [int(v) for v in text.replace("[", "").replace("]", "").replace(",", " ").split()]
    except ValueError:
        raise ValueError(f"mesh n must be integers, got {text!r}") from None
    if not n or any(v < 1 for v in n):
        raise ValueError(f"mesh n must be positive integers, got {text!r}")
    return n


def _mesh_n(m: dict) -> list[int] | None:
    mesh = m.get("mesh")
    if isinstance(mesh, list):
        return list(mesh)
    if isinstance(mesh, dict) and isinstance(mesh.get("n"), list):
        return list(mesh["n"])
    return None


def _set_mesh_n(m: dict, n: list[int]) -> None:
    if isinstance(m.get("mesh"), dict):
        m["mesh"]["n"] = [int(v) for v in n]
    else:
        m["mesh"] = [int(v) for v in n]


# ----------------------------------------------------------------------------------------------- force
def force_fields(cfg: dict) -> dict[str, Any]:
    """Current quick-parameter values present in a force scenario (SI)."""
    out: dict[str, Any] = {}
    lay = cfg.get("layout")
    if isinstance(lay, dict):
        if "gap" in lay:
            out["gap"] = float(lay["gap"])
        if isinstance(lay.get("offset"), list) and lay["offset"]:
            out["offset_x"] = float(lay["offset"][0])
        if "tilt_deg" in lay:
            out["tilt_deg"] = float(lay["tilt_deg"])
    mats = cfg.get("materials")
    if isinstance(mats, dict):
        g = {k: v["grade"] for k, v in mats.items() if isinstance(v, dict) and "grade" in v}
        if g:
            out["grades"] = g
    mesh = {}
    for m in cfg.get("magnets") or []:
        if isinstance(m, dict) and (n := _mesh_n(m)) is not None:
            mesh[str(m.get("name"))] = n
    if mesh:
        out["mesh"] = mesh
    return out


def patch_force(cfg: dict, *, gap: float | None = None, offset_x: float | None = None,
                tilt_deg: float | None = None, grades: dict[str, str] | None = None,
                mesh: dict[str, list[int]] | None = None) -> dict:
    """Patched copy: layout.gap [m], layout.offset[0] [m], layout.tilt_deg [deg], materials.<k>.grade,
    magnets[name].mesh n. Fields absent from the scenario are left absent."""
    c = copy.deepcopy(cfg)
    lay = c.get("layout") if isinstance(c.get("layout"), dict) else None
    if lay is not None:
        if gap is not None and "gap" in lay:
            lay["gap"] = float(gap)
        if offset_x is not None and isinstance(lay.get("offset"), list) and lay["offset"]:
            lay["offset"][0] = float(offset_x)
        if tilt_deg is not None and "tilt_deg" in lay:
            lay["tilt_deg"] = float(tilt_deg)
    for k, g in (grades or {}).items():
        entry = (c.get("materials") or {}).get(k)
        if isinstance(entry, dict) and "grade" in entry:
            entry["grade"] = str(g)
    for m in c.get("magnets") or []:
        if isinstance(m, dict) and str(m.get("name")) in (mesh or {}) and _mesh_n(m) is not None:
            _set_mesh_n(m, mesh[str(m.get("name"))])
    return c


# ----------------------------------------------------------------------------------------------- switching
def bank_node(cfg: dict) -> str | None:
    circ = cfg.get("circuit") or {}
    node = (circ.get("bank") or {}).get("node")
    return str(node) if node is not None and node in (circ.get("ic") or {}) else None


def magnet_temperature_C(cfg: dict) -> float | None:
    mag = cfg.get("magnet") or {}
    if "temperature" in mag:
        return float(mag["temperature"]) - KELVIN_OFFSET
    if "temperature_C" in mag:
        return float(mag["temperature_C"])
    return None


def scr_model(cfg: dict) -> str | None:
    """'latching' | 'original' | 'other' when the scenario uses the board snapshot, else None."""
    circ = cfg.get("circuit") or {}
    if "board" not in circ:
        return None
    models = [str(Path(p)) for p in circ.get("models") or []]
    for key, path in SCR_MODELS.items():
        if models == [path]:
            return key
    return "other"


def switch_fields(cfg: dict) -> dict[str, Any]:
    """Current quick-parameter values present in a switching scenario (SI; temperature in degC)."""
    out: dict[str, Any] = {}
    if (node := bank_node(cfg)) is not None:
        out["voltage"] = float(cfg["circuit"]["ic"][node])
        out["bank_node"] = node
    if (scr := scr_model(cfg)) is not None:
        out["scr_model"] = scr
    mag = cfg.get("magnet") or {}
    if isinstance(mag.get("material"), dict) and "grade" in mag["material"]:
        out["grade"] = str(mag["material"]["grade"])
    if (t := magnet_temperature_C(cfg)) is not None:
        out["temperature_C"] = t
    coil = cfg.get("coil") or {}
    if "reference" in coil:
        out["coil_reference"] = str(coil["reference"])
    if (n := _mesh_n(mag)) is not None:
        out["mesh"] = n
    eddy = cfg.get("eddy")
    # magsim.switching.DEFAULT_EDDY: enabled unless the scenario says otherwise
    out["eddy"] = bool(eddy.get("enabled", True)) if isinstance(eddy, dict) else True
    return out


def patch_switch(cfg: dict, *, voltage: float | None = None, scr: str | None = None, grade: str | None = None,
                 temperature_C: float | None = None, coil_reference: str | None = None,
                 mesh: list[int] | None = None, eddy: bool | None = None) -> dict:
    """Patched copy: circuit.ic[bank node] [V]; circuit.models (board scenarios only; 'latching' or
    'original'); magnet.material.grade; magnet.temperature [K] (replaces temperature_C); coil.reference;
    magnet.mesh n; eddy.enabled."""
    c = copy.deepcopy(cfg)
    if voltage is not None and (node := bank_node(c)) is not None:
        c["circuit"]["ic"][node] = float(voltage)
    if scr is not None and "board" in (c.get("circuit") or {}):
        c["circuit"]["models"] = [SCR_MODELS[scr]]
    mag = c.get("magnet")
    if isinstance(mag, dict):
        if grade is not None and isinstance(mag.get("material"), dict) and "grade" in mag["material"]:
            mag["material"]["grade"] = str(grade)
        if temperature_C is not None:
            mag.pop("temperature_C", None)
            mag["temperature"] = round(float(temperature_C) + KELVIN_OFFSET, 6)
        if mesh is not None and _mesh_n(mag) is not None:
            _set_mesh_n(mag, mesh)
    if coil_reference is not None and isinstance(c.get("coil"), dict) and "reference" in c["coil"]:
        c["coil"]["reference"] = str(coil_reference)
    if eddy is not None:
        if isinstance(c.get("eddy"), dict):
            c["eddy"]["enabled"] = bool(eddy)
        else:
            c["eddy"] = {"enabled": bool(eddy)}
    return c


def rel(path: Path) -> str:
    """Path relative to the project root when inside it."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
