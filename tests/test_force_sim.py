"""Force simulator app: scenario parsing / errors, protocol semantics, sweep integration, CLI exit codes."""

import json

import numpy as np
import pytest

from magsim import analytic as an
from magsim.force_sim import ScenarioError, ForceScenario, run_scenario, validate_scenario, cli_run, cli_sweep


def _cfg(material=None, mesh=(2, 1, 1), gap=2e-3, protocol=None, **extra):
    cfg = {
        "name": "t",
        "materials": {"m": material or {"type": "linear", "Jr": 1.2}},
        "magnets": [
            {"name": "A", "shape": {"type": "cuboid", "dims": [0.02, 0.01, 0.005]}, "material": "m",
             "pose": {"position": [0, 0, -0.0025]}, "mesh": {"n": list(mesh)}},
            {"name": "B", "shape": {"type": "cuboid", "dims": [0.02, 0.01, 0.005]}, "material": "m",
             "mesh": {"n": list(mesh)}},
        ],
        "layout": {"body": "B", "reference": "A", "gap": gap},
        "target": "B",
    }
    if protocol is not None:
        cfg["protocol"] = protocol
    cfg.update(extra)
    return cfg


def test_rigid_pair_matches_akoun_yonnet():
    """Rigid blocks: J uniform, so the runner must reproduce the closed form [akoun1984] to ~1e-8."""
    out = run_scenario(_cfg())
    ref = an.akoun_yonnet_force(np.array([0, 0, 1.2]), np.array([0, 0, 1.2]), (0.02, 0.01, 0.005),
                                (0.02, 0.01, 0.005), np.array([0, 0, 7e-3]))
    assert out["Fz"] == pytest.approx(ref[2], rel=1e-7)
    assert out["J_mean_B"] == pytest.approx(1.2) and out["all_converged"] == 1.0


def test_layout_tilt_gap_is_clearance():
    fs = ForceScenario(_cfg(layout={"body": "B", "reference": "A", "gap": 1e-3, "tilt_deg": 30.0}))
    pB = fs.place_poses["B"]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * [0.01, 0.005, 0.0025]
    zmin = pB.apply(corners)[:, 2].min()
    assert zmin == pytest.approx(1e-3, abs=1e-12)  # A's top face is z = 0


@pytest.mark.parametrize("bad, msg", [
    ({"magnets": []}, "non-empty"),
    ({"target": "C"}, "target"),
    ({"protocol": ["isolate", "jump"]}, "unknown step"),
    ({"protocol": [{"move": {"name": "B"}}]}, "position, rotation"),
    ({"outputs": ["Fq"]}, "unknown output"),
    ({"bogus": 1}, "unknown top-level"),
    ({"materials": {"m": {"grade": "N999"}}}, "unknown grade"),
    ({"layout": {"body": "B", "reference": "A", "gap": -1e-3}}, "gap"),
])
def test_scenario_errors(bad, msg):
    cfg = _cfg()
    cfg.update(bad)
    with pytest.raises(ScenarioError, match=msg):
        validate_scenario(cfg)


def test_overlap_is_scenario_error():
    cfg = _cfg(layout={"body": "B", "reference": "A", "gap": 0.0, "offset": [0, 0]})
    cfg["layout"]["gap"] = 0.0
    ForceScenario(cfg)  # touching is fine
    cfg2 = _cfg()
    cfg2.pop("layout")
    cfg2["magnets"][1]["pose"] = {"position": [0, 0, 0.001]}
    with pytest.raises(ScenarioError, match="overlap"):
        ForceScenario(cfg2)


def test_measure_outputs_and_multiple_measures():
    out = run_scenario(_cfg(protocol=["isolate", "place", "measure", {"move": {"name": "B", "gap": 4e-3}}, "measure"],
                            outputs=["Fz", "J_mean"]))
    assert set(out) == {"Fz", "J_mean_A", "J_mean_B", "Fz_m1", "J_mean_A_m1", "J_mean_B_m1", "Fz_m2",
                        "J_mean_A_m2", "J_mean_B_m2"}
    assert out["Fz"] == out["Fz_m2"] and abs(out["Fz_m2"]) < abs(out["Fz_m1"])


def _rods(grade):
    return {
        "name": "rods", "materials": {"r": {"grade": grade}},
        "magnets": [{"name": "A", "shape": {"type": "cylinder", "diameter": 4.75e-3, "height": 12.5e-3}, "material": "r",
                     "pose": {"position": [0, 0, -6.25e-3]}, "mesh": {"n": [2, 1, 6]}},
                    {"name": "B", "shape": {"type": "cylinder", "diameter": 4.75e-3, "height": 12.5e-3}, "material": "r",
                     "mesh": {"n": [2, 1, 6]}}],
        "layout": {"body": "B", "reference": "A", "gap": 0.0}, "target": "B", "outputs": ["Fz", "J_mean"],
    }


def test_history_matters_for_alnico_not_for_ndfeb():
    """Contact force with vs without the open-circuit excursion: Alnico loses irreversibly, NdFeB barely."""
    for grade, lo, hi in (("LNG37", 0.15, 0.5), ("N42", 0.995, 1.0 + 1e-9)):  # measured 0.25 for LNG37
        direct = run_scenario(dict(_rods(grade), protocol=["place", "measure"]))
        hist = run_scenario(dict(_rods(grade), protocol=["isolate", "place", "measure"]))
        ratio = hist["Fz"] / direct["Fz"]
        assert lo < ratio < hi, (grade, ratio)


def test_pull_off_cycles_repeat_classical_preisach():
    """isolate -> (place, measure, separate) x 2: classical Preisach retraces the recoil loop exactly."""
    out = run_scenario(dict(_rods("LNG37"), protocol=["isolate", "place", "measure", "separate", "place", "measure"]))
    assert out["Fz_m1"] == pytest.approx(out["Fz_m2"], rel=1e-9)


def test_sweep_two_workers(tmp_path):
    import yaml

    cfg = _cfg(sweep={"name": "g", "axes": [{"path": "layout.gap", "values": [2e-3, 4e-3]}], "plot": {"y": ["Fz"]},
                      "workers": 2})
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert cli_sweep(p, out_dir=tmp_path / "out", workers=2) == 0
    csv = (tmp_path / "out" / "g" / "sweep.csv").read_text()
    assert "layout.gap" in csv and (tmp_path / "out" / "g" / "sweep_Fz.png").exists()
    assert (tmp_path / "out" / "j_map.png").exists()


def test_cli_run_and_errors(tmp_path, capsys):
    import yaml

    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(_cfg()))
    assert cli_run(p, out_dir=tmp_path / "o") == 0
    assert json.loads((tmp_path / "o" / "run.json").read_text())["Fz"] < 0
    assert cli_run(tmp_path / "missing.yaml") == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(dict(_cfg(), target="Z")))
    assert cli_run(bad) == 2
    assert "target" in capsys.readouterr().err
