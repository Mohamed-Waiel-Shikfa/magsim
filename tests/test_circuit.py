"""Circuit layer: deck assembly, ngspice runner, gate S1 (a-d), magnetiser snapshot regression."""

import re
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from magsim.circuit import NgspiceError, build_deck, export_spice, load_circuit, parse_wrdata, pwl, run_ngspice
from magsim.circuit.netlist import _split_base

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "res_exp" / "circuits" / "magnetiser"
KICAD = ROOT / "hardware" / "magnetiser" / "kicad"  # in-repo copy of the KiCad project
OLD_KICAD = ROOT.parent / "Kicad" / "magnetiser"  # archived run_sim results

needs_ngspice = pytest.mark.skipif(not __import__("shutil").which("ngspice"), reason="ngspice not installed")
needs_kicad = pytest.mark.skipif(not __import__("shutil").which("kicad-cli") or not KICAD.exists(),
                                 reason="kicad-cli or hardware/magnetiser/kicad not available")

# ------------------------------------------------------------------------------------------- deck text


def test_pwl_text():
    s = pwl([0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6], [0, 1, 2, 3, 4, 5, 6], per_line=3)
    assert s.startswith("PWL(0 0 1e-06 1 2e-06 2\n+ 3e-06 3") and s.endswith("6e-06 6)")
    with pytest.raises(ValueError):
        pwl([0, 1, 1], [0, 1, 2])


def _rl_base():
    return ".title test\nC1 a 0 1e-3 ic=10\n.end\n"


def test_build_deck_structure_and_couplings():
    L = np.array([[1e-5, 2e-7, 1e-7], [2e-7, 1e-7, 3e-8], [1e-7, 3e-8, 2e-7]])
    deck = build_deck(_rl_base(), ("a", "0"), {"L": 1e-5, "R0": 0.1, "alpha": 3.93e-3, "C_th": 1.0, "T0": 300.0},
                      eddy={"L": L, "R": [1e-3, 2e-3]}, emf_pwl={"coil": ([0, 1e-3], [0, 1]), "eddy1": ([0, 1], [0, 2])},
                      tran={"tstep": 1e-6, "tstop": 1e-3, "uic": True}, vectors=["i(vmscs)"])
    lines = deck.splitlines()
    assert lines[0].startswith(".title") and lines[-1] == ".end"
    assert sum(ln.strip().lower() == ".end" for ln in lines) == 1
    assert "C1 a 0 1e-3 ic=10" in lines
    k01 = 2e-7 / np.sqrt(1e-5 * 1e-7)
    assert any(ln.startswith("KMS0_1 LMSC LMSE0 ") and float(ln.split()[-1]) == pytest.approx(k01) for ln in lines)
    assert any(ln.startswith("KMS1_2 LMSE0 LMSE1") for ln in lines)
    assert "VMSE0 mse0_c mse0_a 0" in lines and any(ln.startswith("VMSE1 mse1_c mse1_a PWL(") for ln in lines)
    assert any(ln.startswith("BMSCR msc_2 msc_3 V=i(VMSCS)*0.1*(1+(0.00393)*(V(msc_t)-293.15))") for ln in lines)
    assert any(ln.startswith(".ic V(msc_t)=300") for ln in lines)
    assert ".options method=gear reltol=0.0001" in lines
    assert "wrdata magsim_wrdata.txt i(vmscs)" in lines


@pytest.mark.parametrize("bad,match", [
    ({"eddy": {"L": [[1e-5, 1.1e-6], [1.1e-6, 1e-7]], "R": [1e-3]}}, "positive definite"),
    ({"eddy": {"L": [[2e-5, 1e-7], [1e-7, 1e-7]], "R": [1e-3]}}, "differs from coil"),
    ({"eddy": {"L": [[1e-5, 1e-7], [1e-7, 1e-7]], "R": [0.0]}}, "resistances"),
    ({"coil_nodes": ("a", "nowhere")}, "not found"),
    ({"emf_pwl": {"eddy0": ([0, 1], [0, 1])}}, "emf_pwl keys"),
    ({"extra_lines": ["RMS1 a 0 1"]}, "reserved"),
])
def test_build_deck_rejects(bad, match):
    kw = {"coil_nodes": ("a", "0"), "coil": {"L": 1e-5, "R0": 0.1}}
    kw.update(bad)
    with pytest.raises(ValueError, match=match):
        build_deck(_rl_base(), **kw)


def test_coupling_check_without_pd_violation():
    """|k| >= 1 with an otherwise positive diagonal: rejected before SPICE sees it."""
    with pytest.raises(ValueError):
        build_deck(_rl_base(), ("a", "0"), {"L": 1e-5, "R0": 0.1},
                   eddy={"L": [[1e-5, 1e-6], [1e-6, 1e-7]], "R": [1e-3]})


def test_base_netlist_handling(tmp_path):
    lib = tmp_path / "parts.lib"
    lib.write_text("* empty\n")
    base = '.title KiCad schematic\n.include "/somewhere/else/parts.lib"\nR1 a b 1\nXJ1 a b CONN\n.end\n'
    deck = build_deck(base, ("a", "b"), {"L": 1e-6, "R0": 0.01}, models=[lib], remove_refs=["XJ1"])
    assert f'.include "{lib.resolve()}"' in deck and "/somewhere/else" not in deck and "XJ1" not in deck
    assert deck.splitlines()[0] == ".title magsim deck: KiCad schematic"
    with pytest.raises(ValueError, match="control"):
        _split_base("t\nR1 a 0 1\n.control\nrun\n.endc\n.end\n")
    with pytest.raises(ValueError, match="relative include"):
        build_deck('t\n.include "rel.lib"\nR1 a b 1\n', ("a", "b"), {"L": 1e-6, "R0": 0.01})


def test_parse_wrdata_layouts(tmp_path):
    t = np.array([0.0, 1.0, 1.0, 2.0])
    single = tmp_path / "s.txt"
    single.write_text(" time  i(l1)  v(a)\n" + "\n".join(f"{a} {2 * a + k} {3 * a}" for k, a in enumerate(t)))
    d = parse_wrdata(single, ["i", "v"])
    assert np.allclose(d["time"], [0, 1, 2]) and np.allclose(d["i"], [0, 4, 7])  # duplicate time: keep last
    pairs = tmp_path / "p.txt"
    pairs.write_text("\n".join(f"{a} {2 * a} {a} {3 * a}" for a in [0.0, 1.0, 2.0]))
    d = parse_wrdata(pairs, ["i", "v"])
    assert np.allclose(d["v"], [0, 3, 6])
    d = parse_wrdata(single)
    assert set(d) == {"time", "i(l1)", "v(a)"}


# ------------------------------------------------------------------------------------------- gate S1


def _rlc(L=10e-6, R=60e-3, C=780e-6, V0=450.0):
    alpha = R / (2 * L)
    wd = np.sqrt(1 / (L * C) - alpha**2)
    tpk = np.arctan(wd / alpha) / wd
    ipk = V0 / (wd * L) * np.exp(-alpha * tpk) * np.sin(wd * tpk)
    return alpha, wd, tpk, ipk, 2 * np.pi / wd


@needs_ngspice
def test_S1a_series_rlc_vs_analytic(tmp_path):
    """Gate S1a: capacitor (initial condition) discharging through the coil subcircuit (fixed R, no
    thermal node) vs i = V0/(wd L) e^(-alpha t) sin(wd t): peak and its time < 0.1 % (achieved ~1e-7)."""
    L, R, C, V0 = 10e-6, 60e-3, 780e-6, 450.0
    alpha, wd, tpk, ipk, T = _rlc(L, R, C, V0)
    deck = build_deck(f"rlc\nC1 a 0 {C} ic={V0}\n.end\n", ("a", "0"), {"L": L, "R0": R},
                      tran={"tstep": T / 4000, "tstop": 0.6 * T, "tmax": T / 20000, "uic": True},
                      options={"reltol": 1e-7, "abstol": 1e-9, "vntol": 1e-9}, vectors=["i(vmscs)"])
    d = run_ngspice(deck, tmp_path, ["i(vmscs)"])
    t, i = d["time"], d["i(vmscs)"]
    k = int(np.argmax(i))
    p = np.polyfit(t[k - 1 : k + 2], i[k - 1 : k + 2], 2)
    t_pk, i_pk = -p[1] / (2 * p[0]), np.polyval(p, -p[1] / (2 * p[0]))
    assert abs(i_pk - ipk) / ipk < 1e-3 and abs(t_pk - tpk) / tpk < 1e-3
    assert abs(i_pk - ipk) / ipk < 1e-5 and abs(t_pk - tpk) / tpk < 1e-5
    ia = V0 / (wd * L) * np.exp(-alpha * t) * np.sin(wd * t)
    assert np.max(np.abs(i - ia)) < 1e-5 * ipk


def _magnetiser_metrics(t, il, vc, fire):
    k = int(np.argmax(np.abs(il)))
    return {"I_pk": il[k], "t_pk_us": (t[k] - fire) * 1e6, "didt_max_A_per_us": np.max(np.diff(il) / np.diff(t)) / 1e6,
            "Vcap_end": vc[-1], "coil_I2t": np.trapezoid(il**2, t)}


@needs_ngspice
def test_S1b_magnetiser_pipeline_matches_direct_run(tmp_path):
    """Gate S1b: snapshot board + build_deck with the fixed 10 uH / 60 mOhm coil reproduces a direct
    ngspice run of the same deck (run_sim.py style: L1 + RCOIL1 lines). New board (RG 2.2k, RLED 470):
    I_pk = 1381.5 A at 101.0 us, di/dt max 34.2 A/us, bank left at 172.7 V (old board: 1906 A at 103 us)."""
    cfg = load_circuit(SNAP)
    vec = ["i(vmscs)", "v(cap_int)"]
    deck = build_deck(cfg["base_netlist"], cfg["coil_nodes"], cfg["reference_coil"], extra_lines=cfg["extra_lines"],
                      models=cfg["models"], tran=cfg["tran"], ic=cfg["ic"], options=cfg["options"], vectors=vec)
    d = run_ngspice(deck, tmp_path / "pipeline", vec)
    new = _magnetiser_metrics(d["time"], d["i(vmscs)"], d["v(cap_int)"], cfg["fire_time"])

    lines = cfg["base_netlist"].splitlines()
    tail = [".tran 50n 1.5m", ".ic V(cap_int)=450", ".options method=gear reltol=1e-4"]
    direct = "\n".join(lines[:-1] + cfg["reference_coil_lines"] + cfg["extra_lines"] + tail + [lines[-1]]) + "\n"
    direct = direct.replace('.include "magnetiser.lib"', f'.include "{SNAP / "magnetiser.lib"}"')  # run from a temp dir
    d2 = run_ngspice(direct, tmp_path / "direct", ["i(l1)", "v(cap_int)"])
    ref = _magnetiser_metrics(d2["time"], d2["i(l1)"], d2["v(cap_int)"], cfg["fire_time"])

    assert new["I_pk"] == pytest.approx(ref["I_pk"], rel=1e-3)
    assert abs(new["t_pk_us"] - ref["t_pk_us"]) < 0.25  # 5 output steps of 50 ns
    assert new["didt_max_A_per_us"] == pytest.approx(ref["didt_max_A_per_us"], rel=1e-2)
    assert abs(new["Vcap_end"] - ref["Vcap_end"]) < 0.5
    assert new["coil_I2t"] == pytest.approx(ref["coil_I2t"], rel=1e-3)
    # regression of the new board's nominal case (see test docstring / README)
    assert new["I_pk"] == pytest.approx(1381.5, rel=1e-2)
    assert new["Vcap_end"] == pytest.approx(172.7, abs=2.0)


@needs_ngspice
@pytest.mark.skipif(not (OLD_KICAD / "sim" / "results" / "nominal.cir").exists(), reason="old results not available")
def test_S1b_old_board_deck_reproduces_old_metrics(tmp_path):
    """The old nominal deck (RG 470, RLED 330; control block stripped because its wrdata path points into
    $KICAD) through run_ngspice reproduces sim/results/metrics.txt: the board change, not the pipeline,
    explains the new numbers."""
    old = (OLD_KICAD / "sim" / "results" / "nominal.cir").read_text()
    old = re.sub(r"\.control.*?\.endc\n", "", old, flags=re.S | re.I)
    d = run_ngspice(old, tmp_path, ["i(l1)", "v(cap_int)"])
    m = _magnetiser_metrics(d["time"], d["i(l1)"], d["v(cap_int)"], 10e-6)
    ref = load_circuit(SNAP)["reference_metrics_old"]
    assert m["I_pk"] == pytest.approx(ref["I_coil_peak_A"], rel=1e-3)
    assert abs(m["t_pk_us"] - ref["t_peak_us"]) < 0.5
    assert m["didt_max_A_per_us"] == pytest.approx(ref["coil_didt_max_A_per_us"], rel=5e-3)
    assert m["coil_I2t"] == pytest.approx(ref["coil_I2t_A2s"], rel=1e-3)


@needs_ngspice
@pytest.mark.parametrize("alpha", [0.0, 3.93e-3])
def test_S1c_thermal_node_energy(tmp_path, alpha):
    """Gate S1c: coil temperature rise = int i^2 R(T) dt / C_th (< 0.5 %); with alpha = 0 R is fixed."""
    L, R0, C, V0, Cth, T0 = 10e-6, 60e-3, 780e-6, 450.0, 2.0, 293.15
    *_, T = _rlc(L, R0, C, V0)
    coil = {"L": L, "R0": R0, "alpha": alpha, "T_ref": 293.15, "T0": T0, "C_th": Cth}
    deck = build_deck(f"rlc\nC1 a 0 {C} ic={V0}\n.end\n", ("a", "0"), coil,
                      tran={"tstep": T / 4000, "tstop": 3 * T, "tmax": T / 4000, "uic": True},
                      options={"reltol": 1e-6}, vectors=["i(vmscs)", "v(msc_t)", "v(msc_3)-v(msc_2)"])
    d = run_ngspice(deck, tmp_path, ["i(vmscs)", "v(msc_t)", "u_R"])
    t, i, Tc = d["time"], d["i(vmscs)"], d["v(msc_t)"]
    assert Tc[0] == pytest.approx(T0, abs=1e-6)
    heat = np.trapezoid(i**2 * R0 * (1 + alpha * (Tc - 293.15)), t)
    rise = Tc[-1] - T0
    assert rise > 10.0
    assert abs(rise - heat / Cth) / rise < 5e-3
    # the behavioural resistor follows R(T) = R0 (1 + alpha (T - T_ref)) along the pulse
    big = np.abs(i) > 0.1 * np.max(np.abs(i))
    R_sim = -d["u_R"][big] / i[big]  # v(msc_2) - v(msc_3) = i R(T)
    assert np.allclose(R_sim, R0 * (1 + alpha * (Tc[big] - 293.15)), rtol=1e-6)


@needs_ngspice
def test_S1d_coil_with_shorted_loop_vs_two_mesh(tmp_path):
    """Gate S1d: capacitor -> coil coupled (K) to one shorted eddy loop vs the exact linear two-mesh
    solution x(t) = expm(A t) x0 for [i1, i2, vC]."""
    L1, L2, M, R1, R2, C, V0 = 10e-6, 2e-6, 3e-6, 60e-3, 5e-3, 780e-6, 100.0
    Lm = np.array([[L1, M], [M, L2]])
    *_, T = _rlc(L1, R1, C, V0)
    deck = build_deck(f"rlc\nC1 a 0 {C} ic={V0}\n.end\n", ("a", "0"), {"L": L1, "R0": R1},
                      eddy={"L": Lm, "R": [R2]},
                      tran={"tstep": T / 2000, "tstop": T, "tmax": T / 20000, "uic": True},
                      options={"reltol": 1e-7, "abstol": 1e-9}, vectors=["i(vmscs)", "i(vmse0)"])
    d = run_ngspice(deck, tmp_path, ["i(vmscs)", "i(vmse0)"])
    Li = np.linalg.inv(Lm)
    A = np.zeros((3, 3))
    A[:2, :2] = -Li @ np.diag([R1, R2])
    A[:2, 2] = Li @ np.array([1.0, 0.0])
    A[2, 0] = -1.0 / C
    x0 = np.array([0.0, 0.0, V0])
    ts = d["time"][:: max(1, len(d["time"]) // 400)]
    X = np.array([expm(A * t) @ x0 for t in ts])
    i1 = np.interp(ts, d["time"], d["i(vmscs)"])
    i2 = np.interp(ts, d["time"], d["i(vmse0)"])
    scale = np.max(np.abs(X[:, 0]))
    assert np.max(np.abs(i1 - X[:, 0])) < 1e-4 * scale
    assert np.max(np.abs(i2 - X[:, 1])) < 1e-4 * scale
    assert np.min(X[:, 1]) < -0.1 * scale  # the loop really carries an opposing current


@needs_ngspice
def test_emf_sources_sign_convention(tmp_path):
    """PWL EMF u(t) = d(lambda_ext)/dt in the positive-current direction: L di/dt + R_tot i = -u."""
    L, R, Rext, U = 10e-6, 0.05, 0.95, 2.0
    tau = L / (R + Rext)
    base = f"emf\nRX a 0 {Rext}\n.end\n"
    deck = build_deck(base, ("a", "0"), {"L": L, "R0": R}, emf_pwl={"coil": ([0.0, 1.0], [U, U])},
                      tran={"tstep": tau / 200, "tstop": 5 * tau, "tmax": tau / 2000, "uic": True},
                      options={"reltol": 1e-7}, vectors=["i(vmscs)"])
    d = run_ngspice(deck, tmp_path / "coil", ["i(vmscs)"])
    ref = -U / (R + Rext) * (1 - np.exp(-d["time"] / tau))
    assert np.max(np.abs(d["i(vmscs)"] - ref)) < 1e-5 * U / (R + Rext)
    # eddy loop alone (no coupling): L2 di/dt + R2 i = -u_k
    L2, R2 = 1e-6, 1e-3
    tau2 = L2 / R2
    deck = build_deck(base, ("a", "0"), {"L": L, "R0": R}, eddy={"L": [[L, 0.0], [0.0, L2]], "R": [R2]},
                      emf_pwl={"eddy0": ([0.0, 1.0], [U, U])},
                      tran={"tstep": tau2 / 200, "tstop": 3 * tau2, "tmax": tau2 / 2000, "uic": True},
                      options={"reltol": 1e-7}, vectors=["i(vmse0)"])
    d = run_ngspice(deck, tmp_path / "loop", ["i(vmse0)"])
    ref = -U / R2 * (1 - np.exp(-d["time"] / tau2))
    assert np.max(np.abs(d["i(vmse0)"] - ref)) < 1e-5 * U / R2


@needs_ngspice
def test_run_ngspice_reports_failures(tmp_path):
    bad = "bad\nV1 a 0 1\nX1 a 0 NOSUCHSUBCKT\n.tran 1u 10u\n.end\n"
    with pytest.raises(NgspiceError, match="log"):
        run_ngspice(bad, tmp_path, ["v(a)"])
    assert (tmp_path / "magsim_ngspice.log").exists()


# ------------------------------------------------------------------------------------------- snapshot


@needs_kicad
def test_snapshot_matches_fresh_kicad_export(tmp_path):
    """The stored netlist equals a fresh kicad-cli export of hardware/magnetiser/kicad (RG = 2.2k, RLED = 470),
    apart from the absolute `.include` path (kicad-cli resolves Sim.Library against the schematic folder)."""
    out = export_spice(KICAD / "magnetiser.kicad_sch", tmp_path / "fresh.cir")

    def body(t):
        return [ln for ln in t.splitlines() if not ln.lstrip().lower().startswith(".include")]

    assert body(out.read_text()) == body((SNAP / "magnetiser.cir").read_text())
    text = out.read_text()
    assert re.search(r"^RG1 \S+ \S+ 2\.2k$", text, re.M) and re.search(r"^RLED1 \S+ \S+ 470$", text, re.M)


def test_snapshot_loads():
    cfg = load_circuit(SNAP)
    assert cfg["coil_nodes"] == ("/COIL_A", "/COIL_B")
    assert Path(cfg["models"][0]).exists() and cfg["reference_coil"] == {"L": 10e-6, "R0": 60e-3}
    assert "C1 cap_int 0 780u" in cfg["extra_lines"]
