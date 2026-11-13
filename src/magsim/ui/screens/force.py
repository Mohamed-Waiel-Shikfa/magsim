"""Force page: edit a force scenario, run / sweep / validate it in the background, inspect the outputs."""

import streamlit as st

from magsim.materials.library import list_grades
from magsim.ui import scenario as S
from magsim.ui import views as V


def quick(cfg: dict, text_key: str) -> None:
    f = S.force_fields(cfg)
    if not f:
        st.caption("None of the quick parameters (layout gap / offset / tilt, material grades, mesh) are in "
                   "this scenario.")
        return
    h = V.yaml_key(text_key)
    k = lambda name: f"force_q_{name}_{h}"  # noqa: E731
    grades = list_grades()
    c1, c2, c3 = st.columns(3)
    if "gap" in f:
        c1.number_input("Gap [mm]", min_value=0.0, value=f["gap"] * 1e3, step=0.1, format="%.3f", key=k("gap"))
    if "offset_x" in f:
        c2.number_input("Offset x [mm]", value=f["offset_x"] * 1e3, step=0.5, format="%.3f", key=k("off"))
    if "tilt_deg" in f:
        c3.number_input("Tilt [°]", value=f["tilt_deg"], step=1.0, format="%.2f", key=k("tilt"))
    for mat, g in f.get("grades", {}).items():
        opts = grades if g in grades else [g, *grades]
        st.selectbox(f"Grade of material `{mat}`", opts, index=opts.index(g), key=k(f"grade_{mat}"))
    for mag, n in f.get("mesh", {}).items():
        st.text_input(f"Mesh n of magnet `{mag}` (cells per axis)", value=", ".join(map(str, n)),
                      key=k(f"mesh_{mag}"))

    def patch(c: dict) -> dict:
        ss = st.session_state
        kw = {}
        if "gap" in f:
            kw["gap"] = float(f"{ss[k('gap')] * 1e-3:.12g}")
        if "offset_x" in f:
            kw["offset_x"] = float(f"{ss[k('off')] * 1e-3:.12g}")
        if "tilt_deg" in f:
            kw["tilt_deg"] = float(ss[k("tilt")])
        kw["grades"] = {m: ss[k(f"grade_{m}")] for m in f.get("grades", {})}
        kw["mesh"] = {m: S.parse_mesh(ss[k(f"mesh_{m}")]) for m in f.get("mesh", {})}
        return S.patch_force(c, **kw)

    err_key = f"{text_key}_patch_error"
    if st.session_state.get(err_key):
        st.error(st.session_state.pop(err_key))
    st.button("Apply to YAML", key=k("apply"), on_click=V.apply_patch, args=(text_key, patch),
              help="Rewrites the YAML with these values (comments are dropped).")


V.header("Force", "Force and torque between permanent magnets: edit a scenario, then run it once, sweep it, "
         "or validate it against the measurements.")
V.simulator_page("force", quick)
