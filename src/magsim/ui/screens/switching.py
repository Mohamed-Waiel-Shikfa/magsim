"""Switching page: edit an EPM switching scenario, run / sweep / validate it, inspect the outputs."""

import streamlit as st

from magsim.materials.library import list_grades
from magsim.ui import scenario as S
from magsim.ui import views as V

SCR_LABELS = {"latching": "latching (magnetiser_regen.lib)", "original": "original (magnetiser.lib)",
              "other": "keep the scenario's models"}


def quick(cfg: dict, text_key: str) -> None:
    f = S.switch_fields(cfg)
    h = V.yaml_key(text_key)
    k = lambda name: f"switch_q_{name}_{h}"  # noqa: E731
    grades = list_grades()
    c1, c2 = st.columns(2)
    if "voltage" in f:
        c1.number_input(f"Bank voltage [V] (node `{f['bank_node']}`)", value=f["voltage"], step=10.0,
                        format="%.1f", key=k("v"))
    if "temperature_C" in f:
        c2.number_input("Magnet temperature [°C]", value=f["temperature_C"], step=5.0, format="%.1f",
                        key=k("temp"))
    if "scr_model" in f:
        opts = ["latching", "original"] + (["other"] if f["scr_model"] == "other" else [])
        st.radio("SCR model", opts, index=opts.index(f["scr_model"]), format_func=SCR_LABELS.get,
                 horizontal=True, key=k("scr"))
    c3, c4 = st.columns(2)
    if "grade" in f:
        opts = grades if f["grade"] in grades else [f["grade"], *grades]
        c3.selectbox("Magnet grade", opts, index=opts.index(f["grade"]), key=k("grade"))
    if "coil_reference" in f:
        opts = list(S.COIL_REFERENCES)
        if f["coil_reference"] not in opts:
            opts.insert(0, f["coil_reference"])
        c4.selectbox("Coil reference", opts, index=opts.index(f["coil_reference"]), key=k("coil"))
    if "mesh" in f:
        st.text_input("Magnet mesh n (cells per axis)", value=", ".join(map(str, f["mesh"])), key=k("mesh"))
    st.checkbox("Eddy currents in the magnet", value=f["eddy"], key=k("eddy"))

    def patch(c: dict) -> dict:
        ss = st.session_state
        kw = {"eddy": bool(ss[k("eddy")])}
        if "voltage" in f:
            kw["voltage"] = float(ss[k("v")])
        if "temperature_C" in f and float(ss[k("temp")]) != f["temperature_C"]:
            kw["temperature_C"] = float(ss[k("temp")])
        if "scr_model" in f and ss[k("scr")] != "other":
            kw["scr"] = ss[k("scr")]
        if "grade" in f:
            kw["grade"] = ss[k("grade")]
        if "coil_reference" in f:
            kw["coil_reference"] = ss[k("coil")]
        if "mesh" in f:
            kw["mesh"] = S.parse_mesh(ss[k("mesh")])
        return S.patch_switch(c, **kw)

    err_key = f"{text_key}_patch_error"
    if st.session_state.get(err_key):
        st.error(st.session_state.pop(err_key))
    st.button("Apply to YAML", key=k("apply"), on_click=V.apply_patch, args=(text_key, patch),
              help="Rewrites the YAML with these values (comments are dropped). The magnet temperature is "
                   "written as magnet.temperature in K.")


V.header("Switching", "Pulse circuit coupled to the magnet: what current flows, what field the magnet feels, "
         "and how far it flips.")
V.simulator_page("switch", quick)
