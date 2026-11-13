# magsim — magnet force and EPM switching simulators

magsim answers two questions for the Magnobot electro-permanent magnets:

1. **Force:** what force and torque do two (or more) permanent magnets exert on each other, for any
   NdFeB or Alnico grade, any cuboid / cylinder / ring shape, and any position and orientation? Alnico's
   history-dependent self-demagnetisation is modelled, so "magnetise, pull apart, put back" gives a
   different answer from "magnetise once".
2. **Switching:** if a pulse driver (a KiCad schematic or a SPICE netlist) fires into a coil around a
   magnet, what current flows, what field does the coil generate, what field does the magnet actually feel
   inside, and how far does the magnet flip?

Both can sweep any input over a range and plot the results. Everything is SI (metres, tesla, A/m, volts,
seconds, kelvin). This file tells you how to run things and what you should see. The physics and the
verification numbers are in [`docs/METHODS.md`](docs/METHODS.md).

magsim is the simulation code of a study of electro-permanent magnets (EPMs) for modular robots, work in
progress. Hardware designs and measurement recordings are not part of this repository; `res_exp/` holds
only the small calibration inputs the simulators read.

| folder | what is in it |
|---|---|
| `src/magsim/` | the package: materials (Preisach), magnetostatics (MoM, forces), coil / eddy / circuit, the two simulators, validation, web UI (`ui/`) |
| `scenarios/` | ready-to-run YAML scenarios (section 3) |
| `res_exp/` | calibration inputs: fitted material curves, the magnetiser circuit snapshot, force data points; each folder's README gives its source |
| `docs/` | `METHODS.md` (methods + verification table), `references.bib`, `SPEC.md` |
| `tests/` | ~340 tests: every physics routine against an analytic or independent reference |

## Web UI (the easiest way in)

```bash
uv run magsim-ui
```

Then open http://localhost:8501. The sidebar has six pages:
- **Overview:** environment check (ngspice, kicad-cli) and recent jobs.
- **Force** and **Switching:** pick a scenario, edit its YAML or the quick parameters (gap, grade, bank
  voltage, SCR model, magnet temperature...), press Run / Sweep / Validate. Jobs run in the background:
  you can leave the page and come back. Their outputs land in `res_sim/ui/<time>_<sim>_<action>_<name>/`.
  When a job finishes, the page shows its key numbers, figures and tables, with downloads.
- **Materials:** datasheet numbers, sources and demagnetisation / recoil curves for every grade.
- **Results:** browse every results folder (figures, CSVs, validation reports).
- **Experimental data:** the data READMEs, force-measurement plots, a Hall-trace viewer and the videos.

Everything the UI does is also a command, listed below.

---

## 1. One-time setup

Run everything from the repository root.

```bash
uv sync
```

That installs Python 3.12 and the pinned packages into `.venv/`. The switching simulator also needs
`ngspice` (here `/usr/bin/ngspice`, version 47) and `kicad-cli` (KiCad 10) on the PATH. Check with:

```bash
uv run magsim --help
```

To check the install, run the quick tests (about 1 minute):

```bash
uv run pytest -q -m "not slow"
```

You should see roughly `315 passed, 1 xfailed`. The single expected failure is a known material fit
(LN10 Alnico, Hcb off by 2.6 %).

On this 15 GB machine, set `OMP_NUM_THREADS=1` (or 2) before big sweeps. Each sweep worker otherwise
grabs many BLAS threads, and four of them at once can run out of memory:

```bash
export OMP_NUM_THREADS=1
```

---

## 2. Five-minute tour

Each command reads a scenario file in `scenarios/` and writes into `res_sim/<scenario name>/`. You can
change the destination with `--out DIR`.

| what | command | time | what you should see |
|---|---|---|---|
| Force between the two measured NdFeB blocks at a 2 mm gap | `uv run magsim force run scenarios/force_e1_ndfeb_blocks.yaml` | ~8 s | `Fz = -28.64 N` (library N42; negative = attraction) plus J state per magnet |
| Same blocks, force vs gap 2–10 mm | `uv run magsim force sweep scenarios/force_e1_ndfeb_blocks.yaml` | ~1 min | `res_sim/e1_ndfeb_blocks/gap/sweep.csv` + `sweep_Fz.png` |
| Fit the blocks' real remanence to your measurements | `uv run magsim force validate scenarios/force_e1_ndfeb_blocks.yaml` | ~3 min | `validation.md`: Br = 0.860 T (95 % CI 0.854–0.867), RMS 0.24 N |
| Flip the N52 20×20×5 magnet with the magnetiser board | `uv run magsim switch run scenarios/switch_r1_n52_foil.yaml` | ~2 min | `I_pk = 1752.4 A`, switched fraction `+0.815`, relative remanence `0.858` |
| Same, but sweep bank voltage × magnet temperature | `uv run magsim switch sweep scenarios/switch_r1_v_temp_map.yaml` | ~15 min | heatmaps of switched fraction and weakest-cell felt field |

The general form is:

```bash
uv run magsim force  run|sweep|validate scenarios/<file>.yaml [--out DIR] [--workers N]
uv run magsim switch run|sweep|validate scenarios/<file>.yaml [--out DIR] [--workers N]
```

- `run` simulates the scenario once.
- `sweep` runs the scenario's `sweep:` grid, in parallel with `--workers` processes (default 4).
- `validate` runs the scenario's checks and writes `validation.md`. For force scenarios that means the
  experiment fits, mesh convergence, FEM and magpylib cross-checks; for switching, the convergence,
  energy and mesh checks.

---

## 3. The scenarios that ship

### Force (`scenarios/force_*.yaml`)

| file | models | best used with |
|---|---|---|
| `force_e1_ndfeb_blocks.yaml` | your two 20×10×5 mm NdFeB blocks, face to face, gap 2–10 mm (`res_exp/force/mag_force_data.csv`) | `run`, `sweep`, `validate` (fits Br to the measurements) |
| `force_e2_lng37_rods.yaml` | your two LNG37 Alnico rods D4.75×12.5 mm, coaxial, with the real measurement protocol (3 pull-offs per gap) | `validate` (replays the experiment, both readings of the gap column), `sweep` |
| `force_r2_n42_pair.yaml` | two N42 20×10×5 blocks: force vs gap, vs lateral offset, vs tilt angle | `sweep` (three 1-D sweeps) |
| `force_r2_n42_gap_offset_map.yaml` | same pair, 2-D map over gap × lateral offset | `sweep` (heatmaps of Fz and Fx) |
| `force_r3_lng37_pair.yaml` | LNG37 rod pair vs gap and vs lateral offset, both LNG37 fits | `sweep` |
| `force_grade_gap_sweep.yaml` | the same block pair in N35, N42, N52, LNG37 and Alnico 8, force vs gap | `sweep` (one curve per grade) |

### Switching (`scenarios/switch_*.yaml`)

| file | models | best used with |
|---|---|---|
| `switch_r1_n52_foil.yaml` | N52 20×20×5 mm, initially −z, flipped to +z by the 22-turn foil coil on the magnetiser board at 450 V (latching SCR model) | `run`, `validate` (adds a 3-mesh convergence study) |
| `switch_r1_n52_foil_origlib.yaml` | same, with your original `magnetiser.lib` SCR model | `run` (compare with the line above) |
| `switch_r1_v_temp_map.yaml` | R1 swept over bank voltage 250–450 V × magnet temperature −20…100 °C | `sweep` |
| `switch_r1_firing_rg2k2.yaml`, `switch_r1_firing_rg470.yaml` | R1 swept over bank voltage 150–300 V with the gate resistors at 2.2 kΩ (current schematic) and 470 Ω (previous): where the anode-fired SCRs stop firing | `sweep` |
| `switch_r2_n42_roundwire.yaml` | N42 20×10×5 mm in a 1 mm round-wire coil, 5 turns/layer, 8–25 layers, same board | `run` (20 layers), `sweep` (layers) |
| `switch_r3_lng37_rod.yaml` | LNG37 rod in an illustrative 126-turn solenoid, generic capacitor discharge (`scenarios/circuits/cap_discharge.cir`), bus 2–30 V | `run`, `sweep` (bus voltage) |

### What the shipped sweeps found (rerun the named scenarios to reproduce the numbers)

Force:
- **Grade comparison** (`force_grade_gap_sweep`, 20×10×5 blocks, library datasheet values):

  | gap | N52 | N42 | N35 | Alnico 8 | LNG37 |
  |---|---|---|---|---|---|
  | 0.5 mm | −58.1 N | −49.5 N | −39.3 N | −2.42 N | −0.59 N |
  | 2 mm | −33.5 N | −28.6 N | −22.8 N | −1.28 N | −0.27 N |
  | 10 mm | −5.45 N | −4.65 N | −3.71 N | −0.20 N | −0.03 N |

  A thin open-circuit Alnico block demagnetises itself almost completely, which is why EPMs put Alnico
  in a closed magnetic circuit.
- **R2 gap × offset map** (`force_r2_n42_gap_offset_map`): attraction falls with offset and turns into
  weak repulsion once the lateral offset passes about 19 mm (roughly one block length).
- **LNG37 rod pair** (`force_r3_lng37_pair`): contact 1.88 N (vendor LNG37) / 1.52 N (LNG37-femm), 1 mm
  0.62 / 0.56 N, 3 mm 0.21 / 0.20 N. The two fits differ only near contact.

Switching:
- **R2 layer sweep** (`switch_r2_n42_roundwire`, 450 V): every layer adds 5 turns but also resistance and
  inductance, so the current falls faster than the ampere-turns rise:

  | layers | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 25 |
  |---|---|---|---|---|---|---|---|---|---|
  | I_pk [A] | 1236 | 999 | 829 | 703 | 606 | 529 | 467 | 416 | 356 |
  | switched fraction | 1.000 | 1.000 | 1.000 | 0.987 | 0.945 | 0.891 | 0.821 | 0.727 | 0.560 |

  On this board, 8–12 layers (40–60 turns) flip the N42 completely, and the old 18–25-layer designs do
  not.
- **SCR firing threshold** (`switch_r1_firing_rg2k2` / `_rg470`, latching SCR model): with RG = 2.2 kΩ
  the anode-fired gate gets too little current below about 190 V, and the board does not fire at 150,
  175 or 185 V. At 200 V it fires 160 µs late, and from 250 V up it is within 15 µs of the 470 Ω timing.
  With RG = 470 Ω it fires at every voltage tested (150–300 V). Once fired, the pulse is the same:
  756 A at 200 V either way. At the nominal 450 V the change makes no difference. `magsim switch sweep` writes the plot.
- **R3 bus-voltage sweep** (`switch_r3_lng37_rod`): the Alnico rod reaches zero net polarisation near
  6 V (17 A at 12 V). Relative remanence is 0.80 at 12 V and saturates at about 0.89 from 16 V up.
- **R1 voltage × temperature map** (`switch_r1_v_temp_map`), switched fraction:

  | bank \ magnet | −20 °C | 20 °C | 60 °C | 100 °C |
  |---|---|---|---|---|
  | 250 V | −0.09 | +0.16 | +0.44 | +0.57 |
  | 350 V | +0.35 | +0.58 | +0.75 | +0.63 |
  | 450 V | +0.67 | +0.81 | +0.83 | +0.63 |

  A colder magnet (higher Hcj) is harder to flip. At 100 °C it flips easily, but once the field is gone
  its own demagnetising field undoes part of it: remanence is 0.999 relative to an ideal pulse, yet the
  switched fraction stays near 0.63.

---

## 4. What the runs write and how to read it

### Force `run`

`res_sim/<name>/` contains:
- `run.json`: every output number: `Fx, Fy, Fz` [N] and `Tx, Ty, Tz` [N·m] on the target magnet (about
  its centroid unless `pivot:` says otherwise); per magnet `J_mean_<name>` (volume mean of J along its easy
  axis, T), `J_min_<name>` (weakest cell, T) and `H_par_mean_<name>` (mean internal field along the easy
  axis, A/m, negative = demagnetising); solver `iterations`, `residual`, `converged`.
- `protocol.csv`: the state after every protocol step (`isolate`, `place`, `measure`, `separate`, ...).
- `j_map.png/.pdf`: each magnet's cells, coloured by J. A uniform colour means a nearly rigid magnet;
  for Alnico, the ends are visibly weaker than the middle.
- `scenario.yaml`: the exact input, and `provenance.json` (git commit, package and ngspice versions).

Sign convention: `Fz < 0` on the upper magnet means attraction.

### Force `sweep`

`res_sim/<name>/<sweep name>/sweep.csv` holds one row per grid point, with the swept values, every
output, and an `error` column (empty when the point worked). Next to it are `sweep_<output>.png/.pdf`:
a line plot for one axis, one line per value of a categorical axis (e.g. grade), or a heatmap with
contours for two numeric axes.

### Force `validate`

This writes `res_sim/<name>/validation.md` plus the CSVs and figures behind it. For E1 you get the fitted
Br with two confidence intervals (from the Jacobian and from a bootstrap), the RMS and χ²/dof, a mesh
study, and `e1_fit.png` (data, fit and residuals). For E2 you get the predicted forces with no fitted
parameter, for both readings of the CSV "space" column, and `e2_replay.png`.

### Switching `run`

`res_sim/<name>/` contains:
- `summary.json`: all the numbers. The ones that matter most:

  | key | meaning |
  |---|---|
  | `I_pk_A`, `didt_max_A_per_s` | peak coil current, steepest rise |
  | `H_gen_{centroid,mean,weakest}_A_per_m` | peak field generated by the coil alone, along the switching direction, at the magnet's centre cell, averaged over the magnet, and at the cell that saw the least |
  | `H_app_*` | same, coil + eddy currents in the magnet |
  | `H_felt_*` | the field the magnet material actually feels (applied field plus its own demagnetising field); this is what has to exceed `Hcj_A_per_m` to flip it |
  | `B_in_*` | flux density inside the magnet |
  | `switched_fraction` | −1 = fully in the initial direction, +1 = fully flipped (`switched_fraction_pre` = before the pulse) |
  | `relative_remanence` | magnet strength after the pulse ÷ strength after an ideal saturating pulse (1.0 = as good as a perfect magnetiser) |
  | `E0_J`, `E_R_coil_J`, `E_eddy_J`, `W_magnet_emf_J`, `E_residual_rel` | energy audit: bank energy, coil copper loss, eddy loss, work done on the magnet, and how well the books balance |
  | `dT_coil_K`, `V_bank_end_V`, `I2t_Q1_ratio` | coil heating, charge left in the bank, SCR I²t as a fraction of its rating |
  | `wr_iterations`, `wr_residual`, `wr_converged` | circuit ↔ magnet coupling iterations; converged means < 1e-4 |
- `waveforms_circuit.csv`: time, coil current, bank voltage, coil temperature, eddy currents.
- `waveforms_magnet.csv`: time, the field histories and J at the centroid / mean / weakest cell.
- `cells.csv`: per cell: position, peak felt field, final J.
- `deck_final.cir`: the exact ngspice deck of the last iteration. Open it in ngspice to inspect.
- `figures/`:
  - `circuit_waveforms`: current, bank voltage and coil temperature vs time.
  - `field_histories`: generated / applied / felt field vs time against the Hcj line, at the centre,
    the mean and the weakest cell.
  - `J_history`: magnet polarisation vs time.
  - `cell_maps`: mid-plane maps of peak felt field and final J; the corners and edges show where switching
    is incomplete.
  - `wr_residual`: coupling convergence.

What you should see for R1 (`switch_r1_n52_foil.yaml`): six coupling iterations, residual 2.2e-5. Peak
current is 1752 A at 161 µs after firing. The coil generates 1.34–1.84 MA/m, but the field the magnet
feels peaks at only 0.90 MA/m in its weakest cell. That is just above Hcj = 0.875 MA/m, so the magnet
flips only partly (switched fraction +0.815, 86 % of full remanence). The bank empties; the coil heats by
4.5 K; the energy books balance to 1e-6 of the 79 J in the bank. With the original SCR model
(`switch_r1_n52_foil_origlib.yaml`) the SCRs never fully latch: 1209 A, switched fraction +0.46, and
223 V is left on the bank.

### Switching `sweep` / `validate`

`sweep` writes `sweep.csv` and `figures/sweep_*.png` / `heatmap_*.png`, the same layout as for force.
`validate` writes `validation.md`. It checks coupling convergence and that the answer does not depend on
the initial guess, checks the energy balance, and runs a mesh-convergence table (3 meshes, with
Richardson extrapolation and GCI).

---

## 5. Changing things

Scenarios are plain YAML. Copy one and edit it. Any field can also be swept: give its dotted path in
`sweep.axes`, with list indices in brackets, e.g. `magnets[1].pose.position[2]` or `layout.offset[0]`.

### Magnets (both simulators)

```yaml
magnets:
  - name: A
    shape: {type: cuboid, dims: [0.020, 0.010, 0.005]}        # [m]
    #      {type: cylinder, diameter: 0.00475, height: 0.0125}
    #      {type: ring, d_inner: 0.004, d_outer: 0.010, height: 0.003}
    material: {grade: N42}                 # any library grade, optionally with overrides:
    # material: {grade: N42, Br: 0.86}     #   e.g. the fitted remanence of your blocks
    # material: {type: linear, Jr: 1.2, mu_rec: 1.05}   # simple linear magnet
    pose: {position: [0, 0, 0], rotation: {euler_xyz_deg: [0, 0, 0]}}
    easy_axis: [0, 0, 1]                   # magnetisation direction, body frame
    mesh: {n: [8, 4, 2]}                   # cells per axis; finer = slower but more exact
    initial: +sat                          # +sat | -sat | thermal (demagnetised) | number in [-1, 1]
    temperature_C: 20
```

Library grades: N35, N42, N48, N52, N42SH, NdFeB-bonded, Alnico 2, 3, 5, 5-7, 8, 9, LN10, LNG13, LNGT18,
LNG37, LNG40, LNG44, LNG52, LNG60, LNGT28, LNGT36J, LNGT38, LNGT44, LNGT60, LNGT72, and LNG37-femm (fitted
to the FEMM LNG37 curve). To list them with their datasheet numbers and sources:

```bash
uv run python -c "from magsim.materials.library import list_grades, grade_info; [print(g, grade_info(g)['Br'], grade_info(g)['Hcj']) for g in list_grades()]"
```

For two magnets, `layout: {body: B, reference: A, gap: 2e-3, offset: [x, y], tilt_deg: 10}` places B
above A at that clearance. That is the easiest way to sweep gap, offset or angle without overlaps.
`protocol:` controls magnetisation history (it matters for Alnico): `[isolate, place, measure]` means
"magnetised alone, then brought to the gap, then measured". Add `separate` / `move` steps to replay
pull-offs.

### Your own schematic (switching)

```yaml
circuit:
  schematic: /path/to/your_board.kicad_sch    # exported with kicad-cli on every run
  coil_nodes: ["/COIL_A", "/COIL_B"]          # net names the coil connects between
  models: [/path/to/your_models.lib]          # replaces the schematic's own .include
  extra_lines:                                # parts the schematic does not simulate
    - C1 cap_int 0 780u
    - RESR1 /VBUS cap_int 70m
    - VGP16 gp16_src /PICO_GND PULSE(0 3.3 10u 100n 100n 200u 2)
  ic: {cap_int: 450.0}                        # bank pre-charge [V]
  bank: {element: C1, node: cap_int}          # used for the energy audit
  tran: {tstep: 50.0e-9, tstop: 3.0e-3}
```

You can also give `netlist: file.cir` (any SPICE netlist) or `board: res_exp/circuits/magnetiser` (the
committed snapshot of your magnetiser board, with its undrawn parts in `extra.yaml`). To change one
component without editing the netlist, use `element_values: {RG1: 470, RESR1: 0.17}`. The coil itself
comes from the `coil:` block. Do not put the coil in the netlist: it is built from geometry (inductance,
resistance, heating, eddy coupling).

```yaml
coil:
  reference: r1                     # r1 foil coil | r2 round-wire | r3 rod solenoid
  overrides: {turns: 30}            # change any coil parameter
```

### Speed

- **Force.** A 256-cell pair takes ~10 s per configuration, including force and torque. Coarser meshes
  are seconds.
- **Switching.** One run takes 1–5 min. Most of the time goes to the 6–10 coupling iterations of ngspice
  plus the magnet.
- **Sweeps.** Points run in parallel; four workers is the most this machine handles comfortably.

---

## 6. Things to know before trusting a number

- **The SCR model decides the switching result.** Your original `magnetiser.lib` thyristor acts like a
  dimmer rather than a latch. With RG = 2.2 kΩ it never latches in simulation. The reference scenarios use
  the latching model in `res_exp/circuits/magnetiser/magnetiser_regen.lib`; see `README_regen.md` there.
  - With RG = 2.2 kΩ the anode-fired gate needs about 92 V across each SCR to fire, so the board will not
    fire below roughly 185 V bus. A bench test at low bus voltage would confirm which model is right.
- **Your NdFeB blocks are weaker than N42:** the fit gives Br = 0.86 T, against 1.32 T in the datasheet.
  Use `material: {grade: N42, Br: 0.86}` when you model them.
- **Alnico gap reading.** The force data only fits if the CSV "space" column counts 0.10 mm spacer sheets
  (χ²/dof 0.60), not millimetres (2.52).
- **Cylinder meshes with one azimuthal cell** (`mesh: {n: [nr, 1, nz]}`) cannot represent radial
  polarisation, so they ignore μ_perp. That costs about 4 % on Alnico rod forces; use nφ > 1 (slower) when
  it matters.
- **Not modelled:**
  - soft iron or steel pole pieces;
  - rotational hysteresis;
  - magnetic ageing;
  - mechanical motion during a pulse.

  See `docs/METHODS.md` §9.

---

## 7. Reproducing the numbers

Every number in section 3 comes from a shipped scenario. The commands:

| command |
|---|
| `magsim force validate scenarios/force_e1_ndfeb_blocks.yaml` |
| `magsim force validate scenarios/force_e2_lng37_rods.yaml` |
| `magsim force sweep scenarios/force_r2_n42_pair.yaml` |
| `magsim force sweep scenarios/force_r2_n42_gap_offset_map.yaml` |
| `magsim force sweep scenarios/force_r3_lng37_pair.yaml` |
| `magsim force sweep scenarios/force_grade_gap_sweep.yaml` |
| `magsim switch run scenarios/switch_r1_n52_foil.yaml` |
| `magsim switch run scenarios/switch_r1_n52_foil_origlib.yaml` |
| `magsim switch validate scenarios/switch_r1_n52_foil.yaml` (`validation.md`) |
| `magsim switch sweep scenarios/switch_r1_firing_rg2k2.yaml` (and `_rg470`) |
| `magsim switch sweep scenarios/switch_r1_v_temp_map.yaml` |
| `magsim switch run` + `sweep scenarios/switch_r2_n42_roundwire.yaml` |
| `magsim switch run` + `sweep scenarios/switch_r3_lng37_rod.yaml` |

If ngspice reports "Timestep too small" (it can at the instant the bridge thyristors turn off), magsim
retries that circuit run once with 1 GΩ node shunts (`rshunt`) and keeps that setting for the rest of
the run. You will see a `UserWarning` saying so.

## 8. Using it from Python

```python
from magsim.geometry import Cuboid, Magnet, Pose
from magsim.materials.library import get_material
from magsim.mom import MagnetSystem
from magsim.forces import force_torque

mat = get_material("N42", Br=0.86)
A = Magnet("A", Cuboid((0.02, 0.01, 0.005)), Pose((0, 0, 0)), mat, mesh=(8, 4, 2))
B = Magnet("B", Cuboid((0.02, 0.01, 0.005)), Pose((0, 0, 0.007)), mat, mesh=(8, 4, 2))
system = MagnetSystem([A, B])
solution = system.solve()
F, T = force_torque(system, solution, "B")    # N, N·m on B
```

For switching, `magsim.switching.run_switching(cfg)` takes a scenario dict (`magsim.config.load_scenario`)
and returns `(summary, waveforms)`.

## 9. More documentation

- `docs/METHODS.md`: models, equations, numerics, verification table, limitations (paper methods section)
- `docs/references.bib`: bibliography (keys are cited in the code)
- `docs/SPEC.md`: design specification and acceptance gates
