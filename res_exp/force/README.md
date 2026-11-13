# Force measurements

`mag_force_data.csv` holds the pull force between two identical magnets at a set separation, for two
magnet pairs. It is a byte copy of the original `Mag Force Data.csv`, recorded on 2 July 2026 (date of
the video clips).

## Specimens

| group | magnets | magnetisation | grade |
|---|---|---|---|
| `NIB` | two NdFeB blocks 20 × 10 × 5 mm | through the 5 mm thickness (poles on the 20 × 10 faces) | unknown, supplier says "probably N42" |
| `Alnico` | two cast Alnico rods Ø 4.75 × 12.5 mm | axial | Alnico 5 / LNG37 (supplier grade table in `../materials/vendor_sheets/`, local only) |

The two magnets of a pair face each other coaxially, attracting. 3-D-printed jigs held them; the STL
files are not part of this repository.

## Procedure (as recorded)

- **Gaps.** NdFeB: 2 to 10 mm in 1 mm steps. Alnico: "space" column 0 to 0.40 in 0.05 steps.
- **Readings.** Three per gap. Their average is stored as given (the 7 mm NdFeB average has a typo, 3.63
  instead of 3.58; `load_force_data` flags it, and the fits use the three individual readings).
- **Order.** Each gap was measured in increasing order, separating the magnets after every reading.
- **Uncertainty.** ±0.25 N per reading, as written in the file. The readings are quantised at 0.25 N
  (NdFeB) and about 0.05 N (Alnico), which suggests a spring or dial gauge.
- **Videos.** `videos/ndfeb_blocks_compiled.mp4` and `videos/alnico_lng37_rods_compiled.mp4` are the
  compiled clips of the gauge readings. The 56 raw clips (1.7 GB) stay in the archive.

**To confirm** (questions for the author):
1. The force gauge model, its range and resolution.
2. How the gap was set. The validation reads the Alnico "space" column two ways: as millimetres, and as a
   count of 0.10 mm spacer sheets (gap = 2 × value). Only the second fits the physics (χ²/dof 0.60 vs
   2.52).
3. Whether the force was read at the moment of pull-off (peak) or at a fixed position.
4. How the Alnico rods were magnetised before the test, and whether they were re-magnetised between gaps.

## How magsim uses it

- **NdFeB pair:** `scenarios/force_e1_ndfeb_blocks.yaml` with `magsim force validate`. It fits one
  parameter, the remanence: Br = 0.860 T (95 % CI 0.854–0.867), RMS 0.24 N.
- **Alnico pair:** `scenarios/force_e2_lng37_rods.yaml` with `magsim force validate`. It replays the
  protocol (magnetised → open circuit → three pull-offs per gap) with no fitted parameter.

Reproduce with `magsim force validate scenarios/force_e1_ndfeb_blocks.yaml` and `scenarios/force_e2_lng37_rods.yaml`.
