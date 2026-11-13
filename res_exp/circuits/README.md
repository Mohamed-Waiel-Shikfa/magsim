# Circuit snapshots

`magnetiser/` is the SPICE view of the magnetiser board, the driver the switching simulator runs.

- `magnetiser.cir`: netlist exported with `kicad-cli sch export netlist --format spice` from
  the magnetiser KiCad schematic (not part of this repository) (RG1–RG4 = 2.2 kΩ, RLED1–RLED4 = 470 Ω). A test
  re-exports the schematic and checks that the netlist still matches (skipped when the KiCad project is absent).
- `magnetiser.lib`: the original behavioural device models (CLA100E1200HB thyristor, VO4258H opto-triac,
  VS-60EPS12 diode), fitted to datasheet numbers.
- `magnetiser_regen.lib`, `magnetiser_regen_optodelay.lib`: the same models with a latching
  (regenerative) thyristor, and optionally a realistic opto turn-on delay. Why they exist and what they
  change is in `README_regen.md`.
- `extra.yaml`: the parts the schematic does not simulate: the 780 µF bank with its ESR, the Pico gate
  pulse, the isolated ground. It also holds the coil nodes and the old run_sim reference numbers.
- `README.md`: provenance of each file.

To simulate a different board, either export its schematic the same way and add a folder like this one,
or point a scenario straight at the schematic with `circuit: {schematic: path.kicad_sch, ...}` (see the
top-level README).
