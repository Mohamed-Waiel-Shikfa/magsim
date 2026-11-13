# Magnetiser board snapshot

Reference pulse driver for the switching cases R1–R3. Snapshot taken 2026-10-30 (task 4).

| file | origin |
|---|---|
| `magnetiser.cir` | `kicad-cli sch export netlist --format spice -o magnetiser.cir $KICAD/magnetiser/magnetiser.kicad_sch` with KiCad 10.0.7 (via `magsim.circuit.export_spice`). Schematic sha256 `3a18758d3b1e3373446d8833038dcbaffe7b997d4ac297ea56b7a4a8d4c69240`. Confirms RG1–RG4 = 2.2k and RLED1–RLED4 = 470. Unedited; its `.include` still points at `$KICAD`, which `build_deck` replaces with the local `magnetiser.lib`. |
| `magnetiser.lib` | byte copy of `$KICAD/magnetiser/magnetiser.lib` (sha256 `b9162f90ab2508320861e928e6eb90b626074b10c0ece70ad3abe01edd9edeca`): CLA100E1200HB SCR, VO4258H phototriac and DCROWBAR diode behavioural models. |
| `extra.yaml` | parts the schematic does not draw, transcribed from `$KICAD/magnetiser/sim/run_sim.py` (`MODEL` string, nominal case: 780 µF, ESR 70 mΩ, 450 V), plus coil nets, `.tran`, `.ic`, `.options`, saved vectors and the old fixed 10 µH / 60 mΩ coil and metrics for regression. |

`$KICAD` is the KiCad project folder, which is not part of this repository.

Use: `magsim.circuit.load_circuit("res_exp/circuits/magnetiser")` then `build_deck(...)`; the coil subcircuit
goes between `/COIL_A` and `/COIL_B` (no connector element exists in the netlist, so nothing is removed).
`tests/test_circuit.py` re-runs the old nominal case through this snapshot (gate S1b).
