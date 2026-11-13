# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [Semantic Versioning](https://semver.org/).
Below 1.0.0, scenario fields, the CLI and the Python API may change in any minor release.

## [0.1.0] - 2026-11-14

First public release of magsim, the simulation code of a study of electro-permanent magnets (EPMs) for modular robots. Research code in progress; hardware designs and measurement recordings are not included.

### Added
- Force simulator: force and torque between permanent magnets (NdFeB and Alnico grades; cuboid, cylinder and ring shapes; any position and orientation) with history-dependent self-demagnetisation (Preisach model).
- Switching simulator: a pulse driver (KiCad schematic or SPICE netlist, run in ngspice) coupled to a coil and a magnet; reports coil current, applied and felt field, and how far the magnet flips.
- Sweeps over any input, validation checks, and a web UI (`magsim-ui`, Streamlit) for running scenarios, browsing results and materials.
- 13 ready-to-run YAML scenarios, calibration inputs in `res_exp/` (material fits, magnetiser circuit snapshot, force data points), methods and verification notes in `docs/METHODS.md`.
- 352 tests, each physics routine checked against an analytic or independent reference.
