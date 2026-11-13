"""Web UI (magsim.ui): YAML quick-parameter patching, background jobs, every page renders (no simulations)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from magsim.config import KELVIN_OFFSET, convert_temperatures
from magsim.ui import ROOT, SCENARIOS, jobs
from magsim.ui import data as D
from magsim.ui import scenario as S

PAGES = Path(__file__).resolve().parents[1] / "src" / "magsim" / "ui" / "screens"


def load(name: str) -> dict:
    cfg, err = S.parse((SCENARIOS / name).read_text(encoding="utf-8"))
    assert err is None
    return cfg


# ----------------------------------------------------------------------------------------------- YAML patching
def test_parse_errors_are_reported():
    assert S.parse("a: [1, 2")[0] is None and S.parse("a: [1, 2")[1]
    assert S.parse("- 1\n- 2")[1] == "the top level of a scenario must be a mapping"


def test_scenario_files_found():
    assert "force_e1_ndfeb_blocks.yaml" in [p.name for p in S.scenario_files("force")]
    assert "switch_r1_n52_foil.yaml" in [p.name for p in S.scenario_files("switch")]


def test_force_fields_and_patch_r2_pair():
    cfg = load("force_r2_n42_pair.yaml")
    f = S.force_fields(cfg)
    assert f["gap"] == pytest.approx(2.0e-3) and f["offset_x"] == 0.0 and f["tilt_deg"] == 0.0
    assert f["grades"] == {"block": "N42"} and f["mesh"] == {"A": [8, 4, 2], "B": [8, 4, 2]}
    p = S.patch_force(cfg, gap=3.5e-3, offset_x=1e-3, tilt_deg=10.0, grades={"block": "N52"},
                      mesh={"B": [4, 2, 1]})
    assert p["layout"]["gap"] == 3.5e-3 and p["layout"]["offset"] == [1e-3, 0.0] and p["layout"]["tilt_deg"] == 10
    assert p["materials"]["block"]["grade"] == "N52"
    assert p["magnets"][1]["mesh"]["n"] == [4, 2, 1] and p["magnets"][0]["mesh"]["n"] == [8, 4, 2]
    assert cfg["layout"]["gap"] == pytest.approx(2.0e-3)  # input untouched
    # round trip through the editor text keeps the patch
    assert S.parse(S.dump(p))[0] == p


def test_force_patch_skips_absent_fields():
    cfg = load("force_e1_ndfeb_blocks.yaml")  # layout has no offset / tilt
    f = S.force_fields(cfg)
    assert "offset_x" not in f and "tilt_deg" not in f
    p = S.patch_force(cfg, offset_x=5e-3, tilt_deg=3.0, gap=4e-3)
    assert p["layout"] == {"body": "B", "reference": "A", "gap": 4e-3}
    assert S.has_sweep(cfg) and S.has_validation(cfg)
    g = load("force_grade_gap_sweep.yaml")
    assert S.has_sweep(g) and not S.has_validation(g)


def test_switch_fields_r1():
    f = S.switch_fields(load("switch_r1_n52_foil.yaml"))
    assert f["voltage"] == 450.0 and f["bank_node"] == "cap_int"
    assert f["scr_model"] == "latching" and S.switch_fields(load("switch_r1_n52_foil_origlib.yaml"))["scr_model"] == "original"
    assert f["grade"] == "N52" and f["temperature_C"] == pytest.approx(20.0)
    assert f["coil_reference"] == "r1" and f["mesh"] == [5, 5, 3] and f["eddy"] is True


def test_switch_patch_voltage_scr_temperature():
    cfg = load("switch_r1_n52_foil.yaml")
    p = S.patch_switch(cfg, voltage=300.0, scr="original", grade="N42", temperature_C=60.0, coil_reference="r2",
                       mesh=[3, 3, 1], eddy=False)
    assert p["circuit"]["ic"]["cap_int"] == 300.0
    assert p["circuit"]["models"] == ["res_exp/circuits/magnetiser/magnetiser.lib"]
    assert p["magnet"]["material"]["grade"] == "N42"
    assert "temperature_C" not in p["magnet"] and p["magnet"]["temperature"] == pytest.approx(60.0 + KELVIN_OFFSET)
    assert p["coil"]["reference"] == "r2" and p["magnet"]["mesh"]["n"] == [3, 3, 1] and p["eddy"]["enabled"] is False
    assert p["coil"]["temperature_C"] == 20.0  # coil temperature untouched
    # the patched scenario still loads (no temperature / temperature_C clash) with the magnet in kelvin
    loaded = convert_temperatures(yaml.safe_load(S.dump(p)))
    assert loaded["magnet"]["temperature"] == pytest.approx(333.15)
    assert S.switch_fields(p)["temperature_C"] == pytest.approx(60.0)
    for path in S.SCR_MODELS.values():
        assert (ROOT / path).is_file()


def test_switch_patch_netlist_scenario_keeps_models():
    cfg = load("switch_r3_lng37_rod.yaml")  # plain netlist, no board: no SCR choice
    f = S.switch_fields(cfg)
    assert "scr_model" not in f and f["voltage"] == 12.0 and f["bank_node"] == "cap"
    p = S.patch_switch(cfg, scr="latching", voltage=20.0)
    assert "models" not in p["circuit"] and p["circuit"]["ic"]["cap"] == 20.0


def test_parse_mesh():
    assert S.parse_mesh("8, 4, 2") == [8, 4, 2] and S.parse_mesh("[3 1 9]") == [3, 1, 9]
    for bad in ("", "a, b", "0, 1"):
        with pytest.raises(ValueError):
            S.parse_mesh(bad)


# ----------------------------------------------------------------------------------------------- data helpers
def test_envelope_keeps_extremes_and_bins():
    t = np.linspace(0.0, 1.0, 100_001)
    y = np.sin(2 * np.pi * 5 * t)
    y[12_345] = 7.0
    tm, lo, hi = D.envelope(t, y, 4000)
    assert len(tm) == 4000 and hi.max() == 7.0 and lo.min() == pytest.approx(-1.0, abs=1e-6)
    assert np.all(np.diff(tm) > 0)
    tm, lo, hi = D.envelope(t[:10], y[:10], 4000)
    assert len(tm) == 10 and np.array_equal(lo, hi)


def test_read_hall_small_and_empty():
    hall = sorted((ROOT / "res_exp" / "switching" / "hall").glob("*.csv.gz"))
    if not hall:
        pytest.skip("no Hall traces")
    small = min((p for p in hall if p.stat().st_size > 1024), key=lambda p: p.stat().st_size)
    header, df = D.read_hall(small)
    s = D.hall_stats(header, df)
    assert list(df.columns) == ["t_s", "adc", "field_mT"] and s["n"] == len(df) > 0
    assert s["fs_Hz"] == pytest.approx(40000.0) and s["peak_mT"] >= s["min_mT"]
    empty = min(hall, key=lambda p: p.stat().st_size)
    header, df = D.read_hall(empty)
    assert D.hall_stats(header, df)["n"] == len(df)


def test_output_folders_exclude_ui(tmp_path):
    (tmp_path / "paper" / "a" / "figures").mkdir(parents=True)
    (tmp_path / "paper" / "a" / "figures" / "x.png").write_bytes(b"")
    (tmp_path / "ui" / "job").mkdir(parents=True)
    (tmp_path / "ui" / "job" / "run.json").write_text("{}")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "sweep.csv").write_text("a\n1\n")
    found = [p.relative_to(tmp_path).as_posix() for p in D.output_folders(tmp_path)]
    assert found == ["paper/a", "other"]


# ----------------------------------------------------------------------------------------------- jobs
def _wait(job_dir, until=("done", "failed", "stopped", "lost"), timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = jobs.poll(job_dir)
        if m["status"] in until:
            return m
        time.sleep(0.05)
    raise AssertionError(f"job did not reach {until}: {jobs.poll(job_dir)}")


def test_job_runs_and_finishes(tmp_path):
    d = jobs.new_job_dir("force", "run", "demo scenario", tmp_path)
    assert d.name.endswith("_force_run_demo-scenario")
    m = jobs.launch(d, [sys.executable, "-c", "import os; print('hello', os.environ['OMP_NUM_THREADS'])"],
                    meta={"sim": "force", "action": "run"})
    assert m["pid"] > 0 and m["returncode"] is None
    m = _wait(d)
    assert m["status"] == "done" and m["returncode"] == 0 and m["end"] >= m["start"]
    assert jobs.tail(d) == "hello 1"
    assert [j["name"] for j in jobs.list_jobs(tmp_path, "force")] == [d.name]
    assert jobs.list_jobs(tmp_path, "switch") == []


def test_job_failure_and_stop(tmp_path):
    d = jobs.new_job_dir("switch", "sweep", "x", tmp_path)
    jobs.launch(d, [sys.executable, "-c", "import sys; sys.exit(3)"], meta={"sim": "switch", "action": "sweep"})
    assert _wait(d)["status"] == "failed"
    d2 = jobs.new_job_dir("switch", "sweep", "x", tmp_path)
    assert d2 != d
    jobs.launch(d2, [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(60)"],
                meta={"sim": "switch", "action": "sweep"})
    assert jobs.poll(d2)["status"] == "running"
    assert [j["name"] for j in jobs.running("switch", "sweep", tmp_path)] == [d2.name]
    m = jobs.stop(d2)
    assert m["status"] == "stopped" and m["returncode"] is not None
    assert jobs.running("switch", jobs_dir=tmp_path) == []


def test_lost_job_after_restart(tmp_path):
    """A job.json with a dead pid and no returncode (app restarted mid-run) is 'lost'."""
    d = jobs.new_job_dir("force", "sweep", "y", tmp_path)
    jobs.launch(d, [sys.executable, "-c", "pass"])
    proc = jobs._PROCS.pop(str(d))  # forget it, as after an app restart
    proc.wait()
    assert jobs.poll(d)["status"] == "lost"


def test_start_writes_scenario_and_command(tmp_path, monkeypatch):
    launched = {}
    monkeypatch.setattr(jobs, "launch", lambda d, cmd, meta=None, **kw: launched.update(d=d, cmd=cmd, meta=meta))
    text = (SCENARIOS / "force_e1_ndfeb_blocks.yaml").read_text(encoding="utf-8")
    d = jobs.start("force", "sweep", text, "e1_ndfeb_blocks", workers=3, jobs_dir=tmp_path)
    assert (d / "scenario.yaml").read_text(encoding="utf-8") == text
    assert launched["cmd"] == [sys.executable, "-m", "magsim.cli", "force", "sweep", str(d / "scenario.yaml"),
                               "--out", str(d), "--workers", "3"]
    with pytest.raises(ValueError):
        jobs.start("force", "explode", text, "x", jobs_dir=tmp_path)


# ----------------------------------------------------------------------------------------------- pages
@pytest.mark.parametrize("page", ["overview", "force", "switching", "materials", "results", "experimental"])
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGES / f"{page}.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.title) == 1


def test_app_entry_renders_overview():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGES.parent / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "magsim"
    assert at.number_input(key="workers").value == 2


def test_force_page_quick_apply_rewrites_yaml():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PAGES / "force.py"), default_timeout=60)
    at.run()
    at.selectbox(key="force_file").set_value("force_r2_n42_pair.yaml").run()
    gap = next(w for w in at.number_input if w.label == "Gap [mm]")
    gap.set_value(4.0).run()
    next(b for b in at.button if b.label == "Apply to YAML").click().run()
    assert not at.exception
    cfg, _ = S.parse(at.text_area(key="force_yaml").value)
    assert cfg["layout"]["gap"] == pytest.approx(4.0e-3)
    # buttons: the grade sweep has a sweep but no validation block
    at.selectbox(key="force_file").set_value("force_grade_gap_sweep.yaml").run()
    labels = {b.label: b.disabled for b in at.button if b.label in ("Run", "Sweep", "Validate")}
    assert labels["Sweep"] is False and labels["Validate"] is True


def test_demote_headings_skips_code():
    from magsim.ui.views import demote_headings

    assert demote_headings("# A\ntext\n## B\n```\n# not a heading\n```") == \
        "### A\ntext\n#### B\n```\n# not a heading\n```"
