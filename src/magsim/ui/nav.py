"""Page registry shared by the entry point (navigation) and the Overview page (links)."""

from pathlib import Path

import streamlit as st

# Not named "pages/": Streamlit treats a pages/ folder next to the entry script as a legacy multipage app and,
# on a fresh server, runs a deep-linked page (e.g. /force) without app.py.
PAGES_DIR = Path(__file__).resolve().parent / "screens"

# (file, title, icon, url path, one-line description)
SPECS = [
    ("overview.py", "Overview", ":material/home:", "overview", "environment check and recent jobs"),
    ("force.py", "Force", ":material/compare_arrows:", "force",
     "force / torque scenarios: run, sweep, validate against the measurements"),
    ("switching.py", "Switching", ":material/bolt:", "switching",
     "pulse circuit + magnet switching scenarios: run, sweep, validate"),
    ("materials.py", "Materials", ":material/science:", "materials",
     "grade datasheet numbers, demagnetisation curves, recoil loops"),
    ("results.py", "Results", ":material/folder_open:", "results",
     "browse paper results and other output folders"),
    ("experimental.py", "Experimental data", ":material/lab_profile:", "experimental",
     "force measurements, Hall traces, videos"),
]


def pages() -> list:
    return [st.Page(PAGES_DIR / f, title=t, icon=i, url_path=u, default=(u == "overview"))
            for f, t, i, u, _ in SPECS]
