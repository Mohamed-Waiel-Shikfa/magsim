"""Materials page: datasheet numbers of the library grades and their modelled hysteresis curves."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from magsim.constants import MU0  # noqa: E402
from magsim.materials import library as L  # noqa: E402
from magsim.materials.plots import recoil_loop  # noqa: E402
from magsim.plotting import OKABE_ITO, use_style  # noqa: E402
from magsim.ui import views as V  # noqa: E402

RECOIL_FRACS = (0.6, 0.9, 1.0, 1.1)  # recoil from -f Hcj, as magsim.materials.plots.loops_figure


@st.cache_data(show_spinner=False)
def info(name: str) -> dict:
    return L.grade_info(name)


@st.cache_data(show_spinner=False)
def second_quadrant(name: str, n: int = 500) -> tuple[np.ndarray, np.ndarray, dict]:
    """H in [-1.25 Hcj, 0] [A/m] and J on the descending major branch [T]; model characteristics."""
    m = L.get_material(name)
    ch = m.characteristics()
    H = np.linspace(-1.25 * ch["Hcj"], 0.0, n)
    return H, m.major_loop(H, "descending")[0], ch


@st.cache_data(show_spinner="Computing loops")
def full_loop(name: str) -> dict:
    """Major loop over +-3 Hcj and recoil loops (from -f Hcj up to 0.6 Hcj and back), as in
    magsim.materials.plots.loops_figure."""
    m = L.get_material(name)
    ch = m.characteristics()
    Hcj = ch["Hcj"]
    H = np.linspace(-3 * Hcj, 3 * Hcj, 1201)
    rec = [recoil_loop(m, -f * Hcj, 0.6 * Hcj) for f in RECOIL_FRACS]
    return {"H": H, "Jd": m.major_loop(H, "descending")[0], "Ja": m.major_loop(H, "ascending")[0], "rec": rec,
            "ch": ch}


def numbers_table(grades: list[str]) -> pd.DataFrame:
    rows = []
    for g in grades:
        i = info(g)
        nz = lambda v, s=1.0: None if v is None else v * s  # noqa: E731
        rows.append({
            "grade": g, "family": i["family"], "Br [T]": nz(i["Br"]), "Hcb [kA/m]": nz(i["Hcb"], 1e-3),
            "Hcj [kA/m]": nz(i["Hcj"], 1e-3), "BHmax [kJ/m³]": nz(i["BHmax"], 1e-3),
            "mu_rec": i["mu_rec_model"], "mu_perp": i["mu_perp_model"],
            "alpha_Br [%/K]": i["alpha_Br"] * 100, "beta_Hcj [%/K]": i["beta_Hcj"] * 100,
            "rho_e [µΩ·m]": i["rho_e"] * 1e6, "rho_e par [µΩ·m]": i["rho_e_par"] * 1e6,
        })
    return pd.DataFrame(rows)


V.header("Materials", "Library grades with their sourced datasheet numbers and the Preisach model "
         "identified on them.")
grades = L.list_grades()
left, right = st.columns([2, 3], gap="large")
with left:
    st.subheader("Grades")
    sel = st.multiselect("Grades", grades, default=[g for g in ("N42", "N52", "LNG37") if g in grades],
                         key="mat_grades", persist_state="session")
    g = st.selectbox("Grade for the full loop", grades, index=grades.index(sel[0]) if sel else grades.index("N42"),
                     key="mat_loop")
    if sel:
        with st.expander("Sources and assumptions"):
            for name in sel:
                i = info(name)
                st.markdown(f"**{name}** ({i['family']})")
                for k, v in (i.get("sources") or {}).items():
                    st.markdown(f"- {k}: {v}")
                for a in i.get("assumptions") or ():
                    st.markdown(f"- assumption: {a}")

with right:
    use_style()
    if not sel:
        st.caption("Select at least one grade.")
    else:
        st.subheader("Datasheet numbers")
        df = numbers_table(sel)
        st.dataframe(df, hide_index=True, column_config=V.numeric_config(df))
        st.caption("mu_rec and mu_perp are the values the model uses (datasheet value, or identified / assumed "
                   "where the source gives none; see the assumptions). Temperature coefficients are relative "
                   "to T_ref = 20 °C.")
        st.subheader("Second quadrant")
        fig, (axJ, axB) = plt.subplots(1, 2, figsize=V.FIG_WIDE)
        for c, name in zip(OKABE_ITO * 3, sel):
            H, J, _ = second_quadrant(name)
            axJ.plot(H / 1e3, J, color=c, label=name)
            axB.plot(H / 1e3, MU0 * H + J, color=c, label=name)
        for ax, lab in ((axJ, "J (T)"), (axB, "B (T)")):
            ax.set_xlabel("H (kA/m)")
            ax.set_ylabel(lab)
            ax.set_ylim(bottom=0.0)
            ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader(f"Full loop with recoil lines: {g}")
    d = full_loop(g)
    fig, ax = plt.subplots(figsize=V.FIG_ONE)
    ax.plot(d["H"] / 1e3, d["Jd"], color=OKABE_ITO[0], label="major loop")
    ax.plot(d["H"] / 1e3, d["Ja"], color=OKABE_ITO[0])
    for c, f, (Hl, Jl) in zip(OKABE_ITO[1:], RECOIL_FRACS, d["rec"]):
        ax.plot(Hl / 1e3, Jl, color=c, lw=0.8, label=f"recoil from $-{f:g}\\,H_{{cJ}}$")
    ax.axhline(0, color="0.7", lw=0.4)
    ax.axvline(0, color="0.7", lw=0.4)
    ax.set_xlabel("H (kA/m)")
    ax.set_ylabel("J (T)")
    ax.legend(loc="lower right")
    st.pyplot(fig)
    plt.close(fig)
    ch = d["ch"]
    cols = st.columns(4)
    cols[0].metric("model Br [T]", V.fmt(ch["Br"]))
    cols[1].metric("model Hcj [kA/m]", V.fmt(ch["Hcj"] * 1e-3))
    cols[2].metric("model Hcb [kA/m]", V.fmt(ch["Hcb"] * 1e-3))
    cols[3].metric("model BHmax [kJ/m³]", V.fmt(ch["BHmax"] * 1e-3))
