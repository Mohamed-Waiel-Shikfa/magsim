# Material data

| file | what | source |
|---|---|---|
| `femm_lng37.csv` | second-quadrant B(H) of cast Alnico 5 (LNG37), 11 points | FEMM 4.2 material library, block "Cast Alnico 5 (LNG37)" (from `accurate_alnico.FEM`) |
| `femm_lng56.csv` | same for LNG56 | FEMM library (`new_alnico.FEM`) |
| `femm_n52.csv` | same for sintered NdFeB N52 | FEMM library (`new_alnico.FEM`) |
| `femm_455_stainless.csv` | B(H) of 455 stainless steel (soft, not used by magsim v1) | FEMM library |
| `preisach_fits.json` | identified Preisach parameters for every grade, keyed by a hash of the inputs | written by `uv run python -m magsim.materials.library` |
| `vendor_sheets/` (local only, git-ignored) | supplier grade table for the Alnico rods, demagnetisation-curve sheets for N42 and cast Alnico | supplier / manufacturer documents (copyrighted) |

## How the curves were extracted

FEMM stores a nonlinear permanent-magnet curve shifted right by its coercivity, so it starts at (0, 0)
and ends at (H_c, B_r). The CSVs undo that shift (H = H_FEMM − H_c), giving the physical second-quadrant
curve with B = B_r at H = 0. Each file's header comment says so. Columns: `H_A_per_m,B_T`.

## How the grade library uses them

`magsim.materials.library` holds 27 grades, each with datasheet values (Br, Hcb, Hcj, (BH)max, μ_rec,
temperature coefficients, resistivity, density, heat capacity) and a source string for every field group.
Main sources:
- MMPA Standard 0100-00;
- Arnold Magnetic Technologies catalogues;
- the Eclipse Magnetics N52 minima;
- the supplier sheet for the vendor LNG/LNGT rows.

The library records every assumption. Each grade is identified as a scalar Preisach model, from its
datasheet numbers or, for `LNG37-femm`, by least squares on `femm_lng37.csv`, which fits to 0.96 mT RMS.
The fits are cached in `preisach_fits.json`. A stale entry, one whose inputs changed, is refitted with a
warning.

To regenerate the fits and the loop figures:

```bash
uv run python -m magsim.materials.library   # refit every grade (~35 s), rewrite preisach_fits.json
uv run python -m magsim.materials.plots     # loops_<grade>.png / demag_LNG37.png -> res_sim/materials/
```
