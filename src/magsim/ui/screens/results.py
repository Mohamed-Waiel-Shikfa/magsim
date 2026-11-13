"""Results page: browse committed paper results and any other output folder under res_sim/."""

import streamlit as st

from magsim.ui import RESULTS
from magsim.ui import data as D
from magsim.ui import views as V

V.header("Results", "Browse the output folders under res_sim/ (paper results first; UI jobs are on the Force "
         "and Switching pages).")
folders = D.output_folders(RESULTS, exclude=("ui",))
if not folders:
    st.caption("No output folders under res_sim/.")
    st.stop()
labels = [str(p.relative_to(RESULTS)) for p in folders]
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Folder")
    lab = st.selectbox("Output folder", labels, key="results_folder", persist_state="session")
    folder = folders[labels.index(lab)]
    files = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
    st.caption(f"`res_sim/{lab}` · {len(files)} files")
    with st.expander("Files"):
        st.code("\n".join(map(str, files)) or "(empty)", language="text")
    for name in ("scenario.yaml", "provenance.json"):
        if (folder / name).is_file():
            with st.expander(name):
                st.code((folder / name).read_text(encoding="utf-8"),
                        language="yaml" if name.endswith(".yaml") else "json")
with right:
    st.subheader("Outputs")
    V.show_folder(folder)
