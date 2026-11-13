"""Command line: `magsim {force,switch} {run,sweep,validate} SCENARIO [--out DIR] [--workers N]`.

Dispatches to `magsim.force_sim.cli_<action>` (task 5) or `magsim.switching.cli_<action>` (task 6),
called as `fn(scenario: Path, out_dir: Path | None, workers: int) -> int | None` (exit code, None = 0).
"""

import argparse
import importlib
import sys
from pathlib import Path

_MODULES = {"force": ("magsim.force_sim", 5), "switch": ("magsim.switching", 6)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="magsim", description="Permanent-magnet force and EPM switching simulators.")
    p.add_argument("simulator", choices=sorted(_MODULES), help="force or switch simulator")
    p.add_argument("action", choices=["run", "sweep", "validate"], help="what to do with the scenario")
    p.add_argument("scenario", type=Path, help="scenario YAML file (SI units)")
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument("--workers", type=int, default=4, help="sweep worker processes (default 4)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.scenario.is_file():
        print(f"magsim: scenario file not found: {args.scenario}", file=sys.stderr)
        return 2
    module_name, task = _MODULES[args.simulator]
    fn = getattr(importlib.import_module(module_name), f"cli_{args.action}", None)
    try:
        if fn is None:
            raise NotImplementedError
        code = fn(args.scenario, out_dir=args.out, workers=args.workers)
    except NotImplementedError:
        print(f"magsim {args.simulator} {args.action}: not implemented yet (task {task})", file=sys.stderr)
        return 2
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(main())
