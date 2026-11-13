"""Switching simulator (task 6): scenario parsing, regenerative SCR model, gates S2-S5 on small cases."""

import copy
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from magsim import switching as sw
from magsim.circuit import build_deck, load_circuit, run_ngspice
from magsim.config import load_scenario

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "res_exp" / "circuits" / "magnetiser"
REGEN = SNAP / "magnetiser_regen.lib"
needs_ngspice = pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")


# ------------------------------------------------------------------------------------------- parsing
def test_spice_value():
    assert sw.spice_value("780u") == pytest.approx(780e-6)
    assert sw.spice_value("2.2k") == pytest.approx(2200.0)
    assert sw.spice_value("1meg") == pytest.approx(1e6)
    assert sw.spice_value("70m") == pytest.approx(0.07)
    assert sw.spice_value("1e-6") == pytest.approx(1e-6)
    with pytest.raises(sw.ScenarioError):
        sw.spice_value("abc")


def _tiny_cfg(tmp_path, material=None, eddy=False, V0=60.0):
    """Small fast case: D6 x 3 mm cylinder in a 4-layer circular solenoid, RC discharge (no switch)."""
    net = tmp_path / "rc.cir"
    net.write_text("tiny rc discharge\nC1 cap 0 200u\nRESR1 cap a 10m\n.end\n")
    return {
        "name": "tiny",
        "circuit": {"netlist": str(net), "coil_nodes": ["a", "0"], "ic": {"cap": V0}, "bank": {"element": "C1", "node": "cap"},
                    "tran": {"tstep": 50e-9, "tstop": 300e-6, "uic": True}, "options": {"reltol": 1e-6}},
        "coil": {"type": "round_wire", "params": {"former": "circular", "bore_radius": 3.5e-3, "d_bare": 0.5e-3,
                                                  "d_ins": 0.55e-3, "turns_per_layer": 8, "n_layers": 4}},
        "magnet": {"name": "m", "shape": {"type": "cylinder", "diameter": 6e-3, "height": 3e-3},
                   "material": material or {"type": "linear", "Jr": 0.0, "mu_rec": 4.0, "mu_perp": 4.0},
                   "mesh": {"n": [2, 1, 3]}, "initial": 0.0},
        "eddy": {"enabled": eddy, "n_r": 2, "n_z": 1, "resistivity": 1.5e-6},
        "coupling": {"tol": 1e-4, "max_iter": 15},
    }


@pytest.mark.parametrize("mutate,match", [
    (lambda c: c.update(bogus=1), "unknown top-level"),
    (lambda c: c.pop("magnet"), "needs 'magnet'"),
    (lambda c: c["magnet"]["shape"].update(type="sphere"), "cuboid | cylinder | ring"),
    (lambda c: c["circuit"].update(board="res_exp/circuits/magnetiser"), "exactly one of"),
    (lambda c: c["circuit"].update(element_values={"RXX": 1.0}), "not found"),
    (lambda c: c["circuit"].update(bank={"element": "C9", "node": "cap"}), "not found"),
    (lambda c: c["circuit"]["ic"].clear(), "initial voltage"),
    (lambda c: c["coupling"].update(tolerance=1), "unknown coupling"),
    (lambda c: c["coil"].update(reference="r9"), "r1 | r2 | r3"),
    (lambda c: c["eddy"].update(enabled=True, resistivity=None), "resistivity is required"),
])
def test_scenario_errors(tmp_path, mutate, match):
    cfg = _tiny_cfg(tmp_path)
    mutate(cfg)
    with pytest.raises(sw.ScenarioError, match=match):
        sw.prepare(cfg)


def test_reference_scenarios_parse():
    for name in ("switch_r1_n52_foil", "switch_r1_n52_foil_origlib", "switch_r2_n42_roundwire", "switch_r3_lng37_rod"):
        cfg = load_scenario(ROOT / "scenarios" / f"{name}.yaml")
        c = sw.build_circuit(cfg["circuit"])
        assert c["bank"]["V0"] > 0 and c["bank"]["C"] > 0
        sw.build_coil(cfg["coil"])
        sw.build_magnet(cfg["magnet"])
    c = sw.build_circuit({**load_scenario(ROOT / "scenarios/switch_r1_n52_foil.yaml")["circuit"],
                          "element_values": {"RESR1": 0.1, "RG1": 470}})
    assert "RESR1 /VBUS cap_int 0.1" in c["extra_lines"] and re.search(r"^RG1 \S+ \S+ 470$", c["base_netlist"], re.M)
    assert "Kicad/magnetiser/magnetiser.lib" not in c["base_netlist"]  # board include dropped when models given


# ------------------------------------------------------------------------------------------- regenerative SCR
def _blocks(path):
    text = path.read_text()
    return {m.group(1): m.group(0) for m in re.finditer(r"^\.subckt (\S+).*?^\.ends", text, re.S | re.M)}, \
        re.search(r"^\.model DCROWBAR.*$", text, re.M).group(0)


def test_regen_libs_only_change_the_intended_devices():
    orig, d0 = _blocks(SNAP / "magnetiser.lib")
    regen, d1 = _blocks(REGEN)
    delay, d2 = _blocks(SNAP / "magnetiser_regen_optodelay.lib")
    assert set(orig) == set(regen) == set(delay) == {"CLA100E1200HB", "VO4258H"}
    assert regen["VO4258H"] == orig["VO4258H"] and d0 == d1 == d2
    assert delay["CLA100E1200HB"] == regen["CLA100E1200HB"] != orig["CLA100E1200HB"]


def _scr_deck(lines, tstop, lib=REGEN):
    return "\n".join(["scr test", f'.include "{lib}"', *lines, f".tran 20n {tstop}", ".options method=gear reltol=1e-4",
                      ".end"]) + "\n"


@needs_ngspice
def test_regen_scr_stays_on_after_gate_current_vanishes(tmp_path):
    """Gate pulse 100 mA x 20 us into a 10 V / 10 Ohm anode circuit: latches (V_AK ~ 1 V) and stays on."""
    deck = _scr_deck(["V1 s 0 10", "R1 s an 10", "XQ an g 0 CLA100E1200HB", "IG 0 g PULSE(0 100m 10u 10n 10n 20u 1)"], "200u")
    d = run_ngspice(deck, tmp_path, ["i(v.xq.vsa)", "v(an)", "i(v.xq.vsg)"])
    t, ia = d["time"], d["i(v.xq.vsa)"]
    assert np.max(np.abs(ia[t < 10e-6])) < 1e-6  # off before the gate pulse
    late = t > 100e-6
    assert np.all(np.abs(d["i(v.xq.vsg)"][late]) < 1e-3)  # no gate current any more ...
    assert np.min(ia[late]) > 0.85 and np.max(d["v(an)"][late]) < 1.5  # ... still conducting, V_AK ~ 1 V


@needs_ngspice
def test_regen_scr_turns_off_below_holding_current(tmp_path):
    """Latched at 0.9 A, then the supply ramps down so I_A < IH = 100 mA: blocks; no refire when it comes back."""
    deck = _scr_deck(["V1 s 0 PWL(0 10 100u 10 200u 1.5 300u 1.5 400u 10)", "R1 s an 10", "XQ an g 0 CLA100E1200HB",
                      "IG 0 g PULSE(0 100m 10u 10n 10n 20u 1)"], "600u")
    d = run_ngspice(deck, tmp_path, ["i(v.xq.vsa)", "v(an)"])
    t, ia = d["time"], d["i(v.xq.vsa)"]
    assert ia[np.searchsorted(t, 90e-6)] > 0.85
    assert np.max(np.abs(ia[t > 450e-6])) < 1e-4 and d["v(an)"][-1] > 9.9


@needs_ngspice
def test_regen_scr_needs_igt_and_latching_current(tmp_path):
    """I_G = 30 mA < IGT for 500 us never fires; I_A ~ 120 mA (IH < I_A < IL) does not latch in after the pulse."""
    deck = _scr_deck(["V1 s 0 10", "R1 s an 10", "XQ an g 0 CLA100E1200HB", "IG 0 g PULSE(0 30m 10u 10n 10n 500u 1)"], "600u")
    d = run_ngspice(deck, tmp_path / "igt", ["i(v.xq.vsa)"])
    assert np.max(np.abs(d["i(v.xq.vsa)"])) < 1e-6
    deck = _scr_deck(["V1 s 0 10", "R1 s an 75", "XQ an g 0 CLA100E1200HB", "IG 0 g PULSE(0 100m 10u 10n 10n 20u 1)"], "200u")
    d = run_ngspice(deck, tmp_path / "il", ["i(v.xq.vsa)"])
    t, ia = d["time"], d["i(v.xq.vsa)"]
    assert np.max(ia[(t > 15e-6) & (t < 30e-6)]) > 0.1  # conducts while gated
    assert np.max(np.abs(ia[t > 100e-6])) < 1e-4  # but drops out: below the latching current


def _board(tmp_path, lib, rg="2.2k", V0=450.0):
    cfg = load_circuit(SNAP)
    base = sw._strip_includes(cfg["base_netlist"], ["magnetiser.lib"])
    base = re.sub(r"^(RG\d \S+ \S+) 2\.2k$", r"\1 " + rg, base, flags=re.M)
    vec = ["i(vmscs)", "v(cap_int)", "v(/vbus)-v(/coil_a)", "i(v.xq1.vsg)"]
    deck = build_deck(base, cfg["coil_nodes"], cfg["reference_coil"], extra_lines=cfg["extra_lines"], models=[str(lib)],
                      tran=cfg["tran"], ic={"cap_int": V0}, options=cfg["options"], vectors=vec)
    return run_ngspice(deck, tmp_path, ["i", "vc", "vak", "ig"])


@needs_ngspice
@pytest.mark.parametrize("rg", ["2.2k", "470"])
def test_regen_scr_board_latches(tmp_path, rg):
    """Board with the fixed 10 uH / 60 mOhm coil at 450 V: both thyristors latch (V_AK of Q1 < 10 V during the
    pulse), the bank empties and I_pk ~ 1.96 kA, for RG = 2.2k and 470 Ohm. (Original model, RG 2.2k: 1381 A,
    bank keeps 172.7 V, tests/test_circuit.py.)"""
    d = _board(tmp_path, REGEN, rg)
    t = d["time"]
    assert np.max(d["i"]) == pytest.approx(1959.0, rel=0.01)
    assert d["vc"][-1] < 1.0
    on = (t > 20e-6) & (t < 150e-6)
    assert np.max(d["vak"][on]) < 10.0


@needs_ngspice
def test_regen_scr_minimum_firing_voltage(tmp_path):
    """Anode-fired gate, RG = 2.2k: I_G = (V_AK - V_GK - V_triac)/RG - V_GK/RGK must exceed IGT = 40 mA, i.e.
    V_AK ~ 92 V per thyristor; both share the bus before firing, so 180 V does not fire within the 200 us
    gate pulse and 250 V does."""
    assert np.max(np.abs(_board(tmp_path / "a", REGEN, V0=180.0)["i"])) < 1.0
    assert np.max(_board(tmp_path / "b", REGEN, V0=250.0)["i"]) > 500.0


# ------------------------------------------------------------------------------------------- coupled runs
@needs_ngspice
@pytest.mark.slow
def test_S4_linear_magnet_matches_inductance_fold_in(tmp_path):
    """Gate S4: purely reversible linear magnet (Jr = 0, mu = 4): WR (whole response through the reciprocity
    EMFs, no splitting) = all-SPICE run with dL folded into the inductance matrix, < 0.5 % on i(t).
    Sign test: a magnet polarising along the coil field adds flux linkage (dL > 0), delaying the current peak."""
    cfg = _tiny_cfg(tmp_path)
    r = sw.linear_magnet_check(cfg)
    assert r["wr_converged"]
    assert r["dL_coil_H"] > 0 and 0.02 < r["dL_over_L"] < 1.0
    assert r["max_rel_diff"] < 5e-3
    assert r["no_magnet_rel_diff"] > 10 * r["max_rel_diff"]  # the magnet really matters at this tolerance
    assert r["t_pk_wr_s"] > r["t_pk_nomag_s"]


@needs_ngspice
@pytest.mark.slow
def test_S4_with_eddy_loops_and_split(tmp_path):
    """S4 with eddy loops (EMFs on every loop) and the default WR splitting (dL_ref = exact linear response)."""
    cfg = _tiny_cfg(tmp_path, eddy=True)
    r = sw.linear_magnet_check(cfg)
    assert r["wr_converged"] and r["max_rel_diff"] < 5e-3 and r["dL_coil_H"] > 0
    st = sw.prepare(cfg)
    S, _ = sw.run_switching(cfg, setup=st)
    assert S["wr_converged"] and S["wr_iterations"] <= 3  # split reference = the exact linear response


@needs_ngspice
@pytest.mark.slow
def test_S2_S3_nonlinear_switching_tiny(tmp_path):
    """Gates S2/S3 on a small switching case (N42 cylinder from -sat, eddy loops on): WR converges < 1e-4 within
    15 iterations, independent of the initial EMF (zero vs 0.5 x converged); energy residual < 0.5 % E0; W_magnet
    by sum int i u dt and by sum V int H_app . dJ agree to 1 % of their value."""
    cfg = _tiny_cfg(tmp_path, material={"grade": "N42"}, eddy=True, V0=150.0)
    cfg["magnet"]["initial"] = "-sat"
    cfg["eddy"]["resistivity"] = None
    S, W = sw.run_switching(cfg)
    assert S["wr_converged"] and S["wr_iterations"] <= 15 and S["wr_residual"] < 1e-4
    assert abs(S["E_residual_rel"]) < 5e-3
    assert abs(S["W_magnet_emf_J"] - S["W_magnet_HdJ_J"]) < 0.01 * abs(S["W_magnet_HdJ_J"])
    assert S["switched_fraction"] > S["switched_fraction_pre"] + 0.1  # it does switch
    ig = sw.initial_guess_check(cfg, S, W, scale=0.5)
    assert ig["converged"] and ig["max_rel_diff"] < 1e-3
    assert ig["switched_fraction"] == pytest.approx(S["switched_fraction"], abs=1e-3)


# ------------------------------------------------------------------------------------------- S5 (slow)
@needs_ngspice
@pytest.mark.slow
@pytest.mark.parametrize("name", ["switch_r1_n52_foil", "switch_r3_lng37_rod"])
def test_S5_cli_end_to_end(tmp_path, name):
    """Gate S5: `magsim switch run` on the reference scenarios writes CSV + figures, WR converged, S3 holds."""
    from magsim.cli import main

    out = tmp_path / name
    assert main(["switch", "run", str(ROOT / "scenarios" / f"{name}.yaml"), "--out", str(out)]) == 0
    import json

    S = json.loads((out / "summary.json").read_text())
    assert S["wr_converged"] and abs(S["E_residual_rel"]) < 5e-3
    for f in ("waveforms_circuit.csv", "waveforms_magnet.csv", "cells.csv", "wr_history.csv",
              "figures/circuit_waveforms.png", "figures/field_histories.png", "figures/cell_maps.png",
              "figures/J_history.png", "figures/wr_residual.png"):
        assert (out / f).exists(), f


KICAD_SCH = ROOT / "hardware" / "magnetiser" / "kicad" / "magnetiser.kicad_sch"


@pytest.mark.skipif(shutil.which("kicad-cli") is None or not KICAD_SCH.exists(), reason="kicad-cli or schematic missing")
def test_circuit_from_kicad_schematic_matches_board_snapshot():
    """circuit.schematic exports the .kicad_sch with kicad-cli on the fly; with the board's undrawn parts added
    via extra_lines it resolves to the same deck inputs as the committed board snapshot."""
    from magsim.circuit import load_circuit
    from magsim.switching import build_circuit

    board = Path(__file__).resolve().parents[1] / "res_exp" / "circuits" / "magnetiser"
    snap = load_circuit(board)
    common = {"coil_nodes": ["/COIL_A", "/COIL_B"], "models": [str(board / "magnetiser_regen.lib")],
              "extra_lines": snap["extra_lines"], "ic": {"cap_int": 450.0}, "bank": {"element": "C1", "node": "cap_int"},
              "tran": {"tstep": 50e-9, "tstop": 1e-3}}
    from_sch = build_circuit({"schematic": str(KICAD_SCH), **common})
    from_board = build_circuit({"board": str(board), **common})
    assert from_sch["base_netlist"] == from_board["base_netlist"]


def test_run_circuit_retries_timestep_too_small_with_rshunt(monkeypatch, tmp_path):
    """A 'Timestep too small' failure is retried once with rshunt/itl4 added; other errors propagate."""
    import magsim.switching as sw
    from magsim.circuit.ngspice import NgspiceError

    decks = []

    def fake_run(deck, workdir, names):
        decks.append(deck)
        if len(decks) == 1:
            raise NgspiceError("ngspice failed (exit code 1); log tail:\nTimestep too small; trouble with node")
        return {n: np.zeros(3) for n in names} | {"time": np.zeros(3)}

    monkeypatch.setattr(sw, "make_deck", lambda setup, emf, L=None, lumped=None, robust=False:
                        (f"robust={robust}", ["i_c"]))
    monkeypatch.setattr(sw, "run_ngspice", fake_run)
    setup = type("S", (), {"n_loops": 0, "robust": False})()
    with pytest.warns(UserWarning, match="rshunt"):
        d = sw.run_circuit(setup, None, tmp_path)
    assert decks == ["robust=False", "robust=True"] and d["retried"] is True
    # sticky: later runs of the same setup go straight to the robust deck
    sw.run_circuit(setup, None, tmp_path / "next")
    assert setup.robust is True and decks[-1] == "robust=True"

    def fail(deck, workdir, names):
        raise NgspiceError("some other failure")

    monkeypatch.setattr(sw, "run_ngspice", fail)
    with pytest.raises(NgspiceError):
        sw.run_circuit(type("S", (), {"n_loops": 0, "robust": False})(), None, tmp_path)
