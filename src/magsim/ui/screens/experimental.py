"""Experimental data page: READMEs, measured force vs gap, Hall-probe traces, experiment videos."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from magsim.plotting import OKABE_ITO, use_style  # noqa: E402
from magsim.ui import EXPERIMENTAL, ROOT  # noqa: E402
from magsim.ui import data as D  # noqa: E402
from magsim.ui import views as V  # noqa: E402

HALL_BINS = 4000


@st.cache_data(show_spinner="Reading force data")
def force_data():
    from magsim.validation.experiment import DATA, load_force_data

    return load_force_data(DATA)


@st.cache_data(show_spinner="Reading Hall trace", max_entries=8)
def hall_trace(path: str, mtime: float) -> dict:
    """Header, statistics and the min/max envelope (<= HALL_BINS bins) of one Hall trace."""
    header, df = D.read_hall(path)
    stats = D.hall_stats(header, df)
    t, lo, hi = D.envelope(df["t_s"].to_numpy(), df["field_mT"].to_numpy(), HALL_BINS)
    return {"header": header, "stats": stats, "t": t, "lo": lo, "hi": hi}


V.header("Experimental data", "Measurements used to validate the simulators: force between magnet pairs, "
         "Hall-probe field traces of switching pulses, and videos of the experiments.")
use_style()

# ----------------------------------------------------------------------------------------------- READMEs
st.subheader("Documentation")
readmes = ([EXPERIMENTAL / "README.md"] if (EXPERIMENTAL / "README.md").is_file() else []) + \
    sorted(p for p in EXPERIMENTAL.glob("*/**/README*.md"))
if not readmes:
    st.caption("No README files in res_exp/ yet.")
for p in readmes:
    with st.expander(str(p.relative_to(ROOT)), expanded=p.parent == EXPERIMENTAL):
        st.markdown(p.read_text(encoding="utf-8"))

# ----------------------------------------------------------------------------------------------- force
st.subheader("Force vs gap")
try:
    fd = force_data()
except (OSError, ValueError, StopIteration) as exc:
    st.warning(f"Cannot read res_exp/force/mag_force_data.csv: {exc}")
else:
    left, right = st.columns([2, 3], gap="large")
    groups = (("NdFeB blocks (E1)", fd.ndfeb, "gap (mm)"),
              ("LNG37 rods (E2)", fd.alnico, "CSV 'space' column (mm, or 0.05 = one 0.10 mm sheet)"))
    with right:
        fig, axes = plt.subplots(1, 2, figsize=V.FIG_WIDE)
        for ax, (label, df, xlabel), c in zip(axes, groups, (OKABE_ITO[5], OKABE_ITO[6])):
            mean = df.groupby("gap_mm")["F_N"].mean()
            ax.plot(df["gap_mm"], df["F_N"], "o", color="0.6", ms=2, label="readings")
            ax.errorbar(mean.index, mean.to_numpy(), yerr=fd.sigma_N, fmt="o-", color=c, capsize=2,
                        label=f"mean ± {fd.sigma_N:g} N")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("|F| (N)")
            ax.text(0.97, 0.95, label, transform=ax.transAxes, ha="right", va="top")
            ax.legend(loc="center right")
        st.pyplot(fig)
        plt.close(fig)
    with left:
        st.caption(f"Three readings per gap; stated reading uncertainty ±{fd.sigma_N:g} N. "
                   "Source: res_exp/force/mag_force_data.csv.")
        for label, df, _ in groups:
            t = df.pivot_table(index="gap_mm", columns="rep", values="F_N").reset_index()
            t.columns = ["gap (CSV value)"] + [f"F{r} [N]" for r in t.columns[1:]]
            st.markdown(f"**{label}**")
            st.dataframe(t, hide_index=True, column_config=V.numeric_config(t))
        for n in fd.notes:
            st.caption(f"Note: {n}")

# ----------------------------------------------------------------------------------------------- Hall traces
st.subheader("Hall-probe traces")
traces = sorted((EXPERIMENTAL / "switching" / "hall").glob("*.csv.gz"))
if not traces:
    st.caption("No Hall traces in res_exp/switching/hall/.")
else:
    sizes = {p.name: p.stat().st_size for p in traces}
    names = [p.name for p in traces]
    default = next((i for i, p in enumerate(traces) if sizes[p.name] > 1024), 0)
    left, right = st.columns([2, 3], gap="large")
    with left:
        name = st.selectbox("Trace", names, index=default, key="hall_trace", persist_state="session",
                            format_func=lambda n: f"{n} ({sizes[n] / 1e6:.2f} MB)")
        path = EXPERIMENTAL / "switching" / "hall" / name
        tr = hall_trace(str(path), path.stat().st_mtime)
        s = tr["stats"]
        c1, c2 = st.columns(2)
        c1.metric("duration [s]", V.fmt(s["duration_s"]))
        c2.metric("sample rate [kHz]", V.fmt(s["fs_Hz"] / 1e3 if s["fs_Hz"] else None))
        c3, c4 = st.columns(2)
        c3.metric("peak field [mT]", V.fmt(s["peak_mT"]))
        c4.metric("min field [mT]", V.fmt(s["min_mT"]))
        st.caption(f"{s['n']:,} samples. " + " · ".join(tr["header"]))
    with right:
        if s["n"] == 0:
            st.caption("This file has no samples.")
        else:
            fig, ax = plt.subplots(figsize=V.FIG_WIDE)
            ax.fill_between(tr["t"], tr["lo"], tr["hi"], color=OKABE_ITO[5], lw=0)
            ax.plot(tr["t"], tr["hi"], color=OKABE_ITO[5], lw=0.5)
            ax.plot(tr["t"], tr["lo"], color=OKABE_ITO[5], lw=0.5)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("B (mT)")
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Min/max envelope over {len(tr['t'])} bins of the {s['n']:,} samples.")

# ----------------------------------------------------------------------------------------------- videos
st.subheader("Videos")
videos = sorted(EXPERIMENTAL.glob("**/videos/*.mp4"))
if not videos:
    st.caption("No videos in res_exp/**/videos/.")
else:
    labels = [str(p.relative_to(EXPERIMENTAL)) for p in videos]
    left, right = st.columns([2, 3], gap="large")
    with left:
        lab = st.selectbox("Video", labels, key="video", persist_state="session")
        p = videos[labels.index(lab)]
        st.caption(f"{p.stat().st_size / 1e6:.1f} MB")
    with right:
        st.video(str(p), width=420)  # fixed width: the phone videos are portrait
