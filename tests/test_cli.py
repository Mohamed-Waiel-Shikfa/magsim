import subprocess
import sys

import pytest

from magsim.cli import main


def test_help_exits_zero():
    res = subprocess.run([sys.executable, "-m", "magsim.cli", "--help"], capture_output=True, text=True)
    assert res.returncode == 0 and "force" in res.stdout


@pytest.mark.parametrize("sim", ["force", "switch"])
def test_missing_scenario_exits_two(sim, tmp_path):
    assert main([sim, "run", str(tmp_path / "missing.yaml")]) == 2


def test_bad_args_exit():
    with pytest.raises(SystemExit) as e:
        main(["force", "explode", "x.yaml"])
    assert e.value.code == 2
