# res_exp

Everything magsim reads that does not come from the code itself: measurements, material curves and
circuit snapshots. Each subfolder has its own README describing how its data were obtained.

| folder | contents | used by |
|---|---|---|
| [`force/`](force/README.md) | pull-force measurements of an NdFeB block pair and an Alnico (LNG37) rod pair, with the compiled videos of the readings | `force_e1_*`, `force_e2_*` scenarios, `magsim.validation.experiment` |
| [`switching/`](switching/README.md) | Hall-probe recordings of the field near a magnet during magnetiser pulses, and a video of the first successful EPM switch | experimental-data page of the UI; planned switching validation |
| [`materials/`](materials/README.md) | demagnetisation curves (FEMM library tables), the cached Preisach fits, vendor grade sheets (local only) | `magsim.materials.library` |
| [`circuits/`](circuits/README.md) | SPICE snapshot of the magnetiser board exported from KiCad, device models, the parts the schematic does not draw | `switch_*` scenarios, `magsim.circuit.load_circuit` |

Conventions:
- **Units:** SI, except where a file says otherwise in its header. The Hall CSVs are in mT, and the force
  CSV gives gaps in mm (see its README for how the Alnico gap column is read).
- **Raw files are not edited.** The Hall CSVs are gzipped byte-for-byte copies of the logger output
  (`gzip -dk file.csv.gz` restores them). The force CSV is a byte copy of the original spreadsheet export.
- **Copyrighted third-party documents** (vendor sheets, datasheets) are kept locally and listed in
  `.gitignore`. They are never published.
- **Not stored here** (too large or superseded):
  - the 1.7 GB of raw force-experiment clips;
  - the 2-D FEMM `.ans` solutions;
  - the old simulation outputs.

  They stay in the archived root project, see the root README's archive plan.
