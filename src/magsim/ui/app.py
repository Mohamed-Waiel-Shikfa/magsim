"""Streamlit entry point of the magsim UI: page config, sidebar settings, navigation.

Run with `uv run magsim-ui` (or `streamlit run src/magsim/ui/app.py` from the project root).
"""

import streamlit as st

from magsim.ui import nav

st.set_page_config(layout="wide", page_title="magsim")
page = st.navigation(nav.pages())
with st.sidebar:
    st.number_input("Sweep workers", min_value=1, max_value=8, value=2, step=1, key="workers",
                    persist_state="session",
                    help="--workers passed to every job (sweep points run in parallel; each worker runs with "
                         "OMP_NUM_THREADS=1).")
page.run()
