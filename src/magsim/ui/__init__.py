"""Local Streamlit web UI for magsim (`uv run magsim-ui`).

Modules: `launch` (console script), `app` (Streamlit entry, navigation), `jobs` (background CLI jobs in
res_sim/ui/), `scenario` (YAML quick-parameter patching), `data` (Hall trace decimation, folder scans),
`views` (shared result rendering), `screens/` (one Streamlit script per page).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # project root (src/magsim/ui/__init__.py -> root)
RESULTS = ROOT / "res_sim"
JOBS_DIR = RESULTS / "ui"
SCENARIOS = ROOT / "scenarios"
EXPERIMENTAL = ROOT / "res_exp"
