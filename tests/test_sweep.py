import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml

from magsim.sweep import expand_axes, run_sweep, write_provenance

BASE = {"a": 1.0, "geom": {"x": [0.0, 0.0]}}


def product(cfg):
    """Top-level so it pickles for the process pool."""
    x = cfg["geom"]["x"][1]
    if x < 0:
        raise ValueError("negative x")
    return {"f": cfg["a"] * x, "g": x**2}


def test_expand_axes_grid_order():
    pts = expand_axes([
        {"path": "a", "values": [1, 2]},
        {"path": "geom.x[1]", "linspace": [0, 1, 3]},
    ])
    assert len(pts) == 6
    assert pts[0] == {"a": 1, "geom.x[1]": 0.0}
    assert pts[1] == {"a": 1, "geom.x[1]": 0.5}
    assert pts[3] == {"a": 2, "geom.x[1]": 0.0}


def test_logspace_is_geomspace():
    pts = expand_axes([{"path": "a", "logspace": [1e-3, 1e1, 5]}])
    np.testing.assert_allclose([p["a"] for p in pts], [1e-3, 1e-2, 1e-1, 1, 10], rtol=1e-12)


@pytest.mark.parametrize("axis", [{"path": "a"}, {"path": "a", "values": [1], "linspace": [0, 1, 2]},
                                  {"path": "a", "linspace": [0, 1]}, {"values": [1]}])
def test_bad_axis(axis):
    with pytest.raises(ValueError):
        expand_axes([axis])


def test_run_sweep_parallel_with_failure_and_files(tmp_path):
    axes = [{"path": "a", "values": [1.0, 2.0]}, {"path": "geom.x[1]", "values": [-1.0, 3.0]}]
    df = run_sweep(BASE, axes, product, workers=2, out_dir=tmp_path, name="demo")
    assert list(df.columns) == ["a", "geom.x[1]", "f", "g", "error"]
    assert df["a"].tolist() == [1.0, 1.0, 2.0, 2.0]
    bad = df["geom.x[1]"] < 0
    assert df.loc[bad, "f"].isna().all() and df.loc[bad, "error"].str.contains("negative x").all()
    assert df.loc[~bad, "error"].isna().all()
    np.testing.assert_allclose(df.loc[~bad, "f"], [3.0, 6.0])

    out = tmp_path / "demo"
    csv = pd.read_csv(out / "sweep.csv")
    assert len(csv) == 4 and math.isclose(csv["g"].iloc[1], 9.0)
    assert yaml.safe_load((out / "scenario.yaml").read_text()) == BASE
    prov = json.loads((out / "provenance.json").read_text())
    for key in ("git_commit", "git_dirty", "timestamp_utc", "python", "packages", "ngspice"):
        assert key in prov
    assert prov["packages"]["numpy"] == np.__version__


def test_serial_matches_parallel():
    axes = [{"path": "geom.x[1]", "linspace": [0, 2, 5]}]
    a = run_sweep(BASE, axes, product, workers=1)
    b = run_sweep(BASE, axes, product, workers=2)
    pd.testing.assert_frame_equal(a, b)


def test_write_provenance_extra(tmp_path):
    p = write_provenance(tmp_path / "x", extra={"note": "hi"})
    assert json.loads(p.read_text())["extra"] == {"note": "hi"}
