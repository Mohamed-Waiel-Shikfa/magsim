"""Overview page: what magsim does, links to the pages, environment check, recent jobs."""

import streamlit as st

from magsim.ui import data as D
from magsim.ui import nav
from magsim.ui import views as V


@st.cache_data(ttl=300, show_spinner=False)
def environment() -> dict:
    return D.environment()


V.header("magsim", "Local interface to the magsim force and switching simulators.")
st.markdown(
    "magsim computes the force and torque between permanent magnets of any NdFeB or Alnico grade, shape and "
    "pose, including Alnico's history-dependent self-demagnetisation. It also simulates a pulse circuit firing "
    "into a coil around a magnet and reports the current, the field the magnet feels and how far it switches."
)
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Pages")
    for page, (_, title, icon, _, desc) in zip(nav.pages(), nav.SPECS):
        if title == "Overview":
            continue
        try:
            st.page_link(page, label=title, icon=icon, help=desc)
        except Exception:  # page not registered (page run on its own, e.g. in tests)
            st.markdown(f"**{title}**")
        st.caption(desc)
    st.subheader("Environment")
    env = environment()
    rows = [
        ("ngspice", env["ngspice"]["version"] or "not found", env["ngspice"]["path"] or "-"),
        ("kicad-cli", env["kicad-cli"]["version"] or "not found", env["kicad-cli"]["path"] or "-"),
        ("magsim", env["magsim"] or "not installed", "-"),
        ("git commit", env["git"] or "unknown", "-"),
    ]
    st.dataframe([{"component": a, "version": b, "path": c} for a, b, c in rows], hide_index=True)
    missing = [t for t in ("ngspice", "kicad-cli") if not env[t]["path"]]
    if missing:
        st.warning(f"{', '.join(missing)} not on PATH: switching runs need both.", icon=":material/warning:")
with right:
    st.subheader("Running jobs")
    V.jobs_panel(None)
    st.subheader("Recent jobs")
    V.jobs_table(None)
