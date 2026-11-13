"""Scenario YAML loading and dotted-path access, e.g. "magnets[1].pose.position[2]"."""

import copy
import re
from pathlib import Path
from typing import Any

import yaml

KELVIN_OFFSET = 273.15  # 0 degC in K (exact, SI definition)

_TOKEN = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def load_scenario(path: str | Path) -> dict:
    """Load a scenario YAML file; convert every `*temperature_C` key to kelvin (key without `_C`)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level of a scenario must be a mapping")
    return convert_temperatures(cfg)


def convert_temperatures(obj: Any, where: str = "") -> Any:
    """Return a copy with `<x>temperature_C: v` replaced by `<x>temperature: v + 273.15` (K), recursively."""
    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            here = f"{where}.{key}" if where else str(key)
            if isinstance(key, str) and key.endswith("temperature_C"):
                k_key = key[: -len("_C")]
                if k_key in obj:
                    raise ValueError(f"{here}: give either {key!r} or {k_key!r}, not both")
                out[k_key] = _to_kelvin(val, here)
            else:
                out[key] = convert_temperatures(val, here)
        return out
    if isinstance(obj, list):
        return [convert_temperatures(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    return obj


def _to_kelvin(val: Any, where: str) -> Any:
    if isinstance(val, list):
        return [_to_kelvin(v, f"{where}[{i}]") for i, v in enumerate(val)]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{where}: temperature must be a number in degC, got {val!r}")
    return float(val) + KELVIN_OFFSET


def parse_path(path: str) -> list[str | int]:
    """Split "a.b[1].c" into ["a", "b", 1, "c"]."""
    tokens: list[str | int] = []
    pos = 0
    for m in _TOKEN.finditer(path):
        gap = path[pos : m.start()]
        if gap not in ("", "."):
            raise ValueError(f"malformed path {path!r} near {gap!r}")
        tokens.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
        pos = m.end()
    if not tokens or pos != len(path):
        raise ValueError(f"malformed path {path!r}")
    return tokens


def _step(node: Any, tok: str | int, path: str) -> Any:
    if isinstance(tok, int):
        if not isinstance(node, list):
            raise KeyError(f"{path!r}: index [{tok}] applied to a {type(node).__name__}, not a list")
        if not -len(node) <= tok < len(node):
            raise IndexError(f"{path!r}: index [{tok}] out of range for list of length {len(node)}")
        return node[tok]
    if not isinstance(node, dict):
        raise KeyError(f"{path!r}: key {tok!r} applied to a {type(node).__name__}, not a mapping")
    if tok not in node:
        raise KeyError(f"{path!r}: key {tok!r} not found (have {sorted(map(str, node))})")
    return node[tok]


def get_path(cfg: dict, path: str) -> Any:
    """Value at a dotted path with list indices, e.g. "magnets[1].pose.position[2]"."""
    node = cfg
    for tok in parse_path(path):
        node = _step(node, tok, path)
    return node


def set_path(cfg: dict, path: str, value: Any) -> dict:
    """Deep copy of cfg with the value at `path` replaced. The final key may be new; lists are not grown."""
    tokens = parse_path(path)
    out = copy.deepcopy(cfg)
    node = out
    for tok in tokens[:-1]:
        node = _step(node, tok, path)
    last = tokens[-1]
    if isinstance(last, int):
        _step(node, last, path)  # validates type and range
    elif not isinstance(node, dict):
        raise KeyError(f"{path!r}: key {last!r} applied to a {type(node).__name__}, not a mapping")
    node[last] = copy.deepcopy(value)
    return out
