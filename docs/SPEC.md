# magsim — specification

Publication-grade simulators for the Magnobot electro-permanent-magnet (EPM) work:

1. **Force simulator**: force and torque between permanent magnets (NdFeB and Alnico grades), any
   cuboid / cylinder / ring geometry, any position and orientation, with nonlinear, history-dependent
   self-demagnetisation.
2. **Switching simulator**: a pulse-driver schematic (KiCad or SPICE), coil geometry and magnet go in;
   a coupled transient comes out with peak generated field, peak field felt inside the magnet, final
   magnetisation state, energy audit, device stress and graphs.

Both support parameter sweeps (any scenario field, 1-D and 2-D) and plot quantities of interest
against the swept range. Methods are cited (keys refer to `docs/references.bib`, written in task 7;
see "References" below).

Prior work lives read-only in `$PARENT/simulations_old/` and `$PARENT/analysis/` (parent repo) and
`$KICAD/magnetiser/`. Reuse what is validated; never edit it.

## Goals

- Every physics routine is checked against an analytic or independent reference (tests, tolerances below).
- Discretisation error is reported (mesh refinement + Richardson extrapolation, GCI [celik2008]).
- Every material number has a source; every fitted number is reported with its uncertainty.
- Reproducible: `uv` with a lock file, scenario files (YAML), results folders carry the scenario copy,
  git hash and package versions.

## Non-goals (v1)

- Soft iron / steel keepers (volume-integral MoM with high-permeability iron is a known accuracy trap;
  FEM is the right tool — follow-up).
- Full 3-D eddy-current diffusion FEM (we use a validated loop-network model instead).
- Vector (rotational) hysteresis, magnetic viscosity / ageing, thermally-activated loss, mechanics.
- GUI.

## Stack

Python 3.12 managed by `uv` (lock file committed). numpy, scipy, matplotlib, magpylib (exact field
kernels [ortner2020, ortner2023, slanovc2022]), pyyaml, pandas, scikit-fem (axisymmetric FEM cross-check),
pytest. External tools: `ngspice` 47 (`/usr/bin/ngspice`, batch mode) and `kicad-cli` 10
(schematic → SPICE netlist). Machine: 16 cores, 15 GB RAM with ~3 GB free → keep dense matrices
under ~1 GB, default ≤ 4 sweep workers.

## Conventions

- **SI units everywhere** (m, A, T, A/m, s, V, Ω, K), including YAML. Plots may display mm, kA/m, mT.
- `MU0 = 1.25663706212e-6` H/m (CODATA 2018) in `magsim/constants.py`; never `4e-7*pi` inline.
- Global frame: right-handed Cartesian, metres. A body's pose = centroid position + `scipy.spatial.transform.Rotation`
  (YAML: `position: [x,y,z]`, `rotation: {euler_xyz_deg: [..]}` or `{rotvec_deg: [..]}`).
  Body frame: cylinder/ring axis = body z; easy axis given in the body frame, default `[0,0,1]`.
- Polarisation **J = μ0 M** [T]. Material laws take the field **H** [A/m] in the material.
  `H_par = H·ê`, `H_perp` = the two components orthogonal to ê.
- Force/torque reported on the named target body, from all other bodies. Torque about a pivot (default
  target centroid).
- Temperatures in K in code; YAML may give `temperature_C` for convenience (converted once on load).

## Architecture

```
src/magsim/
  constants.py        MU0, copper data
  config.py           YAML load, dotted-path get/set (lists by index), scenario validation
  sweep.py            grid expansion over dotted paths, process-pool map, tidy DataFrame, CSV + provenance
  plotting.py         publication style + line / multi-line / heatmap+contour helpers
  cli.py              `magsim force|switch run|sweep|validate <scenario.yaml>`
  materials/
    base.py           Material protocol + LinearMaterial (task 1)
    preisach.py       per-cell scalar Preisach + reversible part (task 2)
    library.py        sourced grade table, temperature scaling, identification (task 2)
  geometry.py         Cuboid, Cylinder, Ring, Pose, Magnet, meshing into cells (task 3)
  mom.py              nonlinear magnetostatic volume-integral solver (task 3)
  forces.py           force/torque by magnetic-charge integration (task 3)
  analytic.py         closed-form references (task 3)
  coil.py             coil geometry, L, M, R(T), field per amp (task 4)
  eddy.py             eddy-current loop network in conducting magnets (task 4)
  circuit/
    netlist.py        schematic → SPICE, deck assembly, coil/eddy/thermal subcircuit (task 4)
    ngspice.py        batch runner + result parser (task 4)
  force_sim.py        force scenarios, protocols, outputs (task 5)
  validation/         experiment.py, convergence.py, axisym_fem.py (task 5)
  switching.py        coupled transient (waveform relaxation), metrics, energy audit (task 6)
scenarios/            reference YAML files
data/                 measured force data, material curves, circuit snapshot
res_sim/              generated (git-ignored)
tests/
```

### Interface contracts (tasks 2–6 must honour these)

```python
# materials/base.py
class Material(Protocol):
    name: str
    def init_state(self, n: int, initial: str | float) -> "State": ...
        # initial: "+sat" | "-sat" | "thermal" (Preisach demagnetised) | float J_irr/Js_irr in [-1, 1]
    def response(self, H_par, H_perp, state, T) -> tuple[J_par, dJ_dH_par, J_perp, chi_perp]: ...
        # trial evaluation, does NOT modify state. H_par (n,), H_perp (n,2) [A/m]; T scalar or (n,) [K]
        # J_par, J_perp [T]; dJ_dH_par [T·m/A]; chi_perp dimensionless (J_perp = MU0*chi_perp*H_perp)
    def commit(self, H_par, state, T) -> "State": ...
        # advance history along the monotone segment from the last committed H_par to H_par
```

`State` is an opaque object holding per-cell numpy arrays. `LinearMaterial(Jr, mu_rec, mu_perp)` (task 1)
is the reference implementation: `J_par = Jr·s + MU0(mu_rec-1)H_par`, history-free; `mu_rec = mu_perp = 1`
gives a rigid magnet.

```python
# geometry.py / mom.py / forces.py (task 3)
Magnet(name, shape, pose, material, easy_axis=(0,0,1), mesh=..., initial="+sat", temperature=293.15)
system = MagnetSystem(magnets)              # self-blocks cached; cross-blocks rebuilt when poses change
pts, w = system.field_points()              # where/how callers must sample an applied field
sol = system.solve(H_app=None | (n_pts,3))  # trial solve at current poses + applied field
system.commit(sol)                          # accept the step into material history
F, T = force_torque(system, sol, target="B", pivot=None)
```

```python
# coil.py / eddy.py / circuit (task 4)
coil.inductance(); coil.resistance(T); coil.field_per_amp(points) -> (n,3) A/m per A
loops = eddy_loops(magnet, n_r, n_z, resistivity); loops.R, loops.M, loops.field_per_amp(points)
deck = build_deck(base_netlist, coil_nodes, lumped, extra_lines, models, tran, ic, emf_pwl)
data = run_ngspice(deck, vectors) -> dict[str, np.ndarray]
```

## Force simulator

- **Field kernels**: magpylib exact fields of uniformly polarised cuboids and cylinder segments
  [ortner2020, slanovc2022]. Rings and cylinders are meshed into cylinder-segment cells (exact geometry);
  cuboids into cuboid cells; graded meshing allowed.
- **Self-consistency**: volume-integral method of moments (Radia/MacMMems family [chubar1998, chadebec2006,
  carpentier2013]): uniform J per cell; `H_i = H_app,i + Σ_k N_ik J_k`. Interaction matrix by collocation
  averaged over `q³` Gauss points per target cell (q=1 centroid, q=2 default). Nonlinear solve per load
  step: damped Newton on the cell residual with continuation fallback (pattern in `$PARENT/analysis/fem3d.py:397-545`).
- **History**: scenarios carry a `protocol` (e.g. saturate → open-circuit → place at gap → pull off).
  Each step is solved then committed, so Alnico recoil and irreversible loss follow the real load path.
- **Force/torque**: Kelvin force on the target body as magnetic-charge integration,
  `F = Σ_faces ∮ (ΔJ·n) H_other dA`, torque `T = Σ ∮ (r−p)×(ΔJ·n) H_other dA` (H_other = field of all
  other bodies). The moment of the charge forces already contains the couple `∫J×H_other dV` (integration
  by parts); adding it again double counts (task 3 verified against virtual work). Exact rectangle–rectangle closed form for parallel faces
  (Akoun–Yonnet / Allag–Yonnet family [akoun1984, allag2009]), adaptive Gauss / tanh-sinh quadrature otherwise.
- **References in `analytic.py`**: full-vector Akoun–Yonnet cuboid force with correct zero-gap limits
  (the old `magnet_force.py:214` drops the ±π/2·uv branch — do not copy that bug); coaxial-cylinder force from
  the Lipschitz–Hankel kernels (`$PARENT/simulations_old/Force_compute/python/magnet_force.py:44-103`, [vokoun2009]);
  demagnetising factors [aharoni1998, chen1991]; dipole–dipole force.

## Switching simulator

- **Circuit**: `kicad-cli sch export netlist --format spice` (or a given `.cir`); the coil connector is
  replaced by the coil subcircuit: coil L (from geometry), behavioural R(T) with an adiabatic thermal node
  (`dT/dt = i²R/(m c_p)`, ρ_Cu(T) [matula1979]), eddy loops as resistive loops coupled with `K` elements,
  and one PWL EMF source per circuit for the magnet back-EMF. Undrawn parts (bank, ESR, stimulus, `.ic`)
  come from the scenario's `extra` lines (reference: `$KICAD/magnetiser/sim/run_sim.py`). Models
  `.include`d with absolute paths resolved at run time. ngspice batch, `wrdata` output.
- **Coil**: round-wire (circular or rectangular former, layers × turns) and foil (rectangular former).
  Filament model: Maxwell elliptic mutual inductance for coaxial circles, Neumann/Grover partial inductances
  for straight segments [grover1946, ruehli1972], geometric-mean-distance self terms. Field per amp via
  magpylib `Circle`/`Polyline`. DC resistance with ρ_Cu(T); report skin depth / Dowell factor at the pulse's
  dominant frequency [dowell1966] and warn when > 1.05.
- **Eddy currents** in the magnet (ρ_e from the grade table): nested loops per axial layer, linear, inside
  SPICE. Checked against the exact long-cylinder Bessel solution.
- **Magnet**: the force simulator's MoM + Preisach, driven quasi-statically by `H_app(x,t) = h_coil(x)i_c(t)
  + Σ h_k(x)i_k(t)`, re-sampled where the field path changes by more than a tolerance (Preisach is
  rate-independent). Back-EMF on every circuit by reciprocity: `λ_c(t) = Σ_cells V J·h_c`, `e_c = −dλ_c/dt`.
- **Coupling**: Gauss–Jacobi waveform relaxation [lelarasmee1982] — run ngspice, march the magnet, update
  EMF PWLs, repeat until `max|Δi|/max|i| < 1e-4` (default, ≤ 15 iterations). Report the residual history.
- **Field definitions** (ŝ = target switching direction):
  - generated field `H_gen(x,t) = h_coil(x)i_c(t)` and applied field `H_app = H_gen + H_eddy`;
  - felt field `H_felt = H_app + H_demag` (the total H inside the material, i.e. what drives the material law);
    `B_in = MU0 H_felt + J`;
  - report max over time of `ŝ·H` at the centroid, volume mean and minimum over cells (the weakest cell
    decides full switching), for H_gen, H_app and H_felt, in A/m and as MU0·H in T.
- **Outputs**: waveforms (i, V_bank, coil T, EMFs), field-history plots, per-cell maps of peak felt field
  and final J, switched fraction `⟨J_irr·ŝ⟩/Js_irr ∈ [−1,1]`, relative open-circuit remanence vs an ideal
  saturating pulse, energy audit (`½CV0² = losses + ΔE_C + W_magnet + residual`, with W_magnet checked two
  ways: `∫i·e dt` and `∫∫H_app·dJ dV`), device stress (peak I, di/dt, I²t vs datasheet), WR history.

## Reference cases

| id | use | magnet | coil / driver |
|---|---|---|---|
| R1 | switching | N52 20×20×5 mm, magnetised through 5 mm | 22-turn Cu foil 0.2×5 mm, 0.05 mm kapton, 20×20 bore; magnetiser board (`Kicad/magnetiser`, RG = 2.2k, RLED = 470), 780 µF / 450 V |
| R2 | switching + force | N42 20×10×5 mm (size of the measured blocks) | round wire 1.0 mm Cu (1.07 mm OD), 5 turns/layer, 20 layers (sweep 18–25), 20×10 bore; same board |
| R3 | force + switching | Alnico LNG37 rod D4.75×12.5 mm, axial | force: measured pair; switching (illustrative): round-wire solenoid on the rod, sweep bus voltage |
| E1 | validation | NdFeB block pair 20×10×5 (data in `Mag Force Data.csv`) | – |
| E2 | validation | LNG37 rod pair (same CSV) | – |

Old-code parameter sources: R1 `$PARENT/simulations_old/Switch_sim/magnet-simulation/magnet_pulse_sim.py:157-191`,
R2 `$PARENT/simulations_old/Switch_sim/magnet-flip-sim/src/params.py`, board `$KICAD/magnetiser/sim/run_sim.py`.

## Acceptance gates (tests must assert these)

Force / magnetostatics
- F1 rigid aligned cuboid pair vs Akoun–Yonnet: rel. error < 1e-5 for gaps {0, 1e-4, 1e-2, 0.1, 1, 10}×size
  and lateral offsets; force continuous at gap → 0⁺.
- F2 rigid coaxial cylinders vs Lipschitz–Hankel reference: < 1e-4.
- F3 open-circuit magnet: volume-mean demag factor vs Aharoni (prism) / Chen–Brug–Goldfarb (cylinder) after
  Richardson extrapolation: < 1e-4 for χ = 0 (rigid), < 0.5 % for μ_rec = 1.05. These tables are χ = 0 values;
  for μ_rec = 4 (χ = 3) −μ0⟨H⟩/⟨J⟩ is the χ-dependent factor, so the reference is the FEM of task 5 (gate F6).
- F4 dipole limit at 20× size < 0.1 %. F5 `F_AB = −F_BA` < 1e-6 rel.; isolated body net force/torque ≈ 0;
  torque vs finite-difference virtual work < 1e-3 rel.
- F6 nonlinear Alnico rod (open circuit, and pair at 0.1 mm): MoM vs old 1-D slab solver and vs axisymmetric
  FEM; agreement < 2 % or the discrepancy is explained in the report. GCI reported.
- F7 experiment E1/E2 replay: fitted parameters with 95 % CI, RMS, χ²/dof, residual plot. Honest about what is fitted.

Materials
- P1 every library grade reproduces its datasheet Br, Hcj (±0.5 %), Hcb, (BH)max, Hk (±2 %).
- P2 recoil slope at a reversal point = MU0·μ_rec (±1 %); wiping-out and congruency hold; major loop is odd-symmetric.
- P3 fit to the FEMM LNG37 table: RMS < 0.02 T. P4 10⁴ cells × 1 step < 50 ms.

Coil / circuit / switching
- C1 single-layer solenoid L vs exact Lorenz/Nagaoka < 1e-4; thick circular coil vs a published Lyle/Grover/
  Babic–Akyel value < 1e-3. C2 on-axis fields vs closed form < 1e-8.
- C3 R1 foil coil reproduces the old validated 16.84 µH and 38.6 mΩ (20 °C) within 1 %, or the difference is explained.
- C4 eddy-loop model vs long-cylinder Bessel solution: |H_in/H0| within 3 % at f = 1/τ_diffusion.
- S1 pipeline with fixed L,R and no magnet reproduces a direct ngspice run of the same deck; series RLC vs analytic < 0.1 %.
- S2 WR converges (< 1e-4) in ≤ 15 iterations for R1–R3; independent of initial guess.
- S3 energy residual < 0.5 % of ½CV0². S4 linear magnet: WR result = all-SPICE run with the magnet's reversible
  response folded into the inductance matrix, < 0.5 %.
- S5 R1–R3 run end-to-end from `magsim switch run` and produce CSV + figures.

## Open questions (do not block; report both readings)

- E2 gap: the old validation treated the CSV "space" column as spacer sheets of 0.10 mm (model gap = 2× the
  CSV value). The user has not confirmed which reading is right.
- Opto-triac turn-on: the SPICE model switches in 1 µs, the firmware comment says 30–70 µs. Firmware pulse 200 µs vs
  the schematic note "hold ≥ 1 ms".

## References (bib keys; DOIs verified by the literature survey unless marked)

akoun1984 10.1109/TMAG.1984.1063554 · allag2009 10.1109/TMAG.2009.2025047 · vokoun2009 10.1016/j.jmmm.2009.07.030 ·
ravaud2009 10.1109/TMAG.2009.2016088 · caciagli2018 10.1016/j.jmmm.2018.02.003 · derby2010 10.1119/1.3256157 ·
slanovc2022 10.1016/j.jmmm.2022.169482 · ortner2020 10.1016/j.softx.2020.100466 · ortner2023 10.3390/magnetism3010002 ·
chubar1998 10.1107/S0909049597013502 · chadebec2006 10.1109/TMAG.2006.870929 · carpentier2013 10.1109/TMAG.2013.2241750 ·
newell1993 10.1029/93JB00694 · aharoni1998 10.1063/1.367113 · chen1991 10.1109/20.102932 ·
mayergoyz1991 (book, Mathematical Models of Hysteresis) · yu2015 10.1109/TMAG.2014.2362615 ·
campbell1982 10.1109/TMAG.1982.1061943 · knaian2010 (MIT PhD thesis, hdl 1721.1/60151) ·
jiles1986 10.1016/0304-8853(86)90066-1 · grover1946 (book) · ruehli1972 10.1147/rd.165.0470 ·
lyle1914 10.1098/rsta.1914.0009 · babic2003 10.1109/TMAG.2003.810550 · conway2007 10.1109/TMAG.2006.888565 ·
dowell1966 (Proc. IEE 113(8) 1387, verify) · matula1979 10.1063/1.555614 · ruoho2010 10.1109/TMAG.2009.2027815 ·
zhilichev2002 10.1109/TMAG.2002.803195 · przybylski2016 10.3390/s16040569 · dorrell2007 10.1109/TMAG.2007.893715 ·
nakata1986 10.1109/TMAG.1986.1064377 · jewell1990 10.1109/20.104473 · lelarasmee1982 (IEEE TCAD 1(3) 131, verify) ·
celik2008 (J. Fluids Eng. 130 078001, verify) · katter2005 10.1109/TMAG.2005.854872 (verify) ·
magnequench2023 (SAE Automotive Engineering, Oct 2023, MQ1 table) · mmpa0100 (MMPA Standard 0100-00) · arnold_alnico (Arnold Cast Alnico brochure Rev. C) · arnold_neo (Arnold Neo Catalog) · ngspice (manual v47) · herman2017 10.21105/joss.00097
