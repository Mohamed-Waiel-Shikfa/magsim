"""Shared Streamlit rendering: number formatting, result folders, jobs panel, simulator page skeleton."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from magsim.ui import JOBS_DIR, SCENARIOS
from magsim.ui import data as D
from magsim.ui import jobs
from magsim.ui import scenario as S

MAX_ROWS = 5000  # rows shown per CSV table (the download has all of them)
# Figure sizes [in] for st.pyplot: the image is stretched to the column width, so a narrower figure keeps the
# 8 pt publication fonts (magsim.plotting.use_style) readable at about 12 px.
FIG_WIDE = (5.2, 2.4)
FIG_ONE = (4.4, 2.9)


# ----------------------------------------------------------------------------------------------- formatting
def fmt(v, digits: int = 4) -> str:
    """Number to `digits` significant figures; booleans as yes/no; None/NaN as '-'."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return "-"
        return f"{v:.{digits}g}"
    return str(v)


def rel(p: Path) -> str:
    return S.rel(p)


def style() -> None:
    """Monospace scenario editor (functional, not decorative)."""
    st.html("<style>textarea{font-family:'Source Code Pro',ui-monospace,monospace !important;"
            "font-size:0.85rem !important;white-space:pre !important;overflow-x:auto !important;}</style>")


def header(title: str, caption: str) -> None:
    style()
    st.title(title)
    st.caption(caption)


@st.cache_data(show_spinner=False, max_entries=64)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def read_csv(path: Path) -> pd.DataFrame:
    return _read_csv(str(path), Path(path).stat().st_mtime)


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _key(*parts) -> str:
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def numeric_config(df: pd.DataFrame) -> dict:
    """column_config with 4 significant figures for every float column."""
    return {c: st.column_config.NumberColumn(c, format="%.4g") for c in df.columns
            if pd.api.types.is_float_dtype(df[c])}


def table(path: Path, label: str | None = None) -> None:
    """A CSV as a dataframe (first MAX_ROWS rows) with a download button for the whole file."""
    try:
        df = read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.warning(f"{rel(path)}: cannot read ({exc})")
        return
    st.markdown(f"`{label or rel(path)}` ({len(df)} rows)")
    st.dataframe(df.head(MAX_ROWS), hide_index=True, column_config=numeric_config(df))
    st.download_button("Download CSV", functools.partial(Path(path).read_bytes), file_name=Path(path).name,
                       mime="text/csv", key=f"dl_{_key(path)}", icon=":material/download:")


def figure_grid(paths: list[Path], base: Path) -> None:
    """PNG figures in a 2-column grid, captioned with their path relative to `base`."""
    for i in range(0, len(paths), 2):
        cols = st.columns(2)
        for col, p in zip(cols, paths[i:i + 2]):
            with col:
                st.image(str(p), caption=str(p.relative_to(base)), width="stretch")


# ----------------------------------------------------------------------------------------------- force / switch numbers
def force_quantity(key: str) -> tuple[str, float]:
    """(display unit, scale from SI) of a force run.json key."""
    if key in ("Fx", "Fy", "Fz") or key.split("_m")[0] in ("Fx", "Fy", "Fz"):
        return "N", 1.0
    if key in ("Tx", "Ty", "Tz") or key.split("_m")[0] in ("Tx", "Ty", "Tz"):
        return "N·m", 1.0
    if key.startswith(("J_mean", "J_min")) or key == "residual":
        return "T", 1.0
    if key.startswith("H_par_mean"):
        return "kA/m", 1e-3
    return "", 1.0


def show_force_run(run: dict) -> None:
    """Metric rows for Fx..Tz, J_mean_*, converged, then the full output table."""
    fkeys = [k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz") if k in run]
    if fkeys:
        cols = st.columns(len(fkeys))
        for c, k in zip(cols, fkeys):
            unit, sc = force_quantity(k)
            c.metric(f"{k} [{unit}]", fmt(run[k] * sc))
    jk = [k for k in run if k.startswith("J_mean_") and "_m" not in k[len("J_mean_"):]]
    extra = [(f"{k} [T]", fmt(run[k])) for k in jk]
    if "converged" in run:
        extra.append(("converged", "yes" if run["converged"] == 1 else "no"))
    if extra:
        cols = st.columns(max(len(extra), 3))
        for c, (lab, val) in zip(cols, extra):
            c.metric(lab, val)
    rows = []
    for k, v in run.items():
        unit, sc = force_quantity(k)
        rows.append({"output": k, "value": v * sc if isinstance(v, (int, float)) else v, "unit": unit})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, column_config={"value": st.column_config.NumberColumn("value", format="%.4g")})


SWITCH_METRICS = [
    ("I_pk_A", "I_pk [A]", 1.0),
    ("switched_fraction", "switched fraction", 1.0),
    ("relative_remanence", "relative remanence", 1.0),
    ("E_residual_rel", "energy residual / E0", 1.0),
    ("H_gen_centroid_A_per_m", "H_gen centroid [MA/m]", 1e-6),
    ("H_gen_mean_A_per_m", "H_gen mean [MA/m]", 1e-6),
    ("H_gen_weakest_A_per_m", "H_gen weakest [MA/m]", 1e-6),
    ("Hcj_A_per_m", "Hcj [MA/m]", 1e-6),
    ("H_felt_centroid_A_per_m", "H_felt centroid [MA/m]", 1e-6),
    ("H_felt_mean_A_per_m", "H_felt mean [MA/m]", 1e-6),
    ("H_felt_weakest_A_per_m", "H_felt weakest [MA/m]", 1e-6),
    ("dT_coil_K", "coil dT [K]", 1.0),
    ("V_bank_end_V", "V_bank end [V]", 1.0),
    ("wr_iterations", "WR iterations", 1.0),
    ("wr_converged", "WR converged", 1.0),
]


def summary_unit(key: str) -> tuple[str, float]:
    """(display unit, scale from SI) of a switching summary.json key, from its suffix."""
    for suf, unit, sc in (("_A_per_m", "MA/m", 1e-6), ("_A_per_s", "A/s", 1.0), ("_A2s", "A²·s", 1.0),
                          ("_ohm_m", "Ω·m", 1.0), ("_ohm", "Ω", 1.0), ("_J_per_K", "J/K", 1.0),
                          ("_A", "A", 1.0), ("_V", "V", 1.0), ("_K", "K", 1.0), ("_J", "J", 1.0),
                          ("_T", "T", 1.0), ("_H", "H", 1.0), ("_s", "s", 1.0)):
        if key.endswith(suf):
            return unit, sc
    return "", 1.0


def show_switch_summary(summ: dict) -> None:
    present = [(k, lab, sc) for k, lab, sc in SWITCH_METRICS if k in summ]
    for i in range(0, len(present), 4):
        cols = st.columns(4)
        for c, (k, lab, sc) in zip(cols, present[i:i + 4]):
            v = summ[k]
            c.metric(lab, fmt(v) if isinstance(v, bool) else fmt(v * sc if isinstance(v, (int, float)) else v))
    rows = []
    for k, v in summ.items():
        if isinstance(v, (list, dict)):
            continue
        unit, sc = summary_unit(k)
        num = isinstance(v, (int, float)) and not isinstance(v, bool)
        rows.append({"quantity": k, "value": fmt(v * sc) if num else fmt(v), "unit": unit})
    with st.expander("All summary values"):
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def demote_headings(md: str, by: int = 2) -> str:
    """Shift Markdown ATX headings down `by` levels (max h6) so a report sits under the page's subheaders."""
    out, fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and re.match(r"#{1,6} ", line):
            n = len(line) - len(line.lstrip("#"))
            line = "#" * min(6, n + by) + line[n:]
        out.append(line)
    return "\n".join(out)


def show_folder(folder: Path) -> None:
    """Everything a run / sweep / validate wrote: numbers, report, sweep tables, figures, other CSVs."""
    folder = Path(folder)
    shown = False
    if (folder / "run.json").is_file():
        st.markdown("**Force run outputs** (`run.json`)")
        show_force_run(read_json(folder / "run.json"))
        shown = True
    if (folder / "summary.json").is_file():
        st.markdown("**Switching summary** (`summary.json`)")
        show_switch_summary(read_json(folder / "summary.json"))
        shown = True
    if (folder / "validation.md").is_file():
        with st.expander("validation.md", expanded=True):
            st.markdown(demote_headings((folder / "validation.md").read_text(encoding="utf-8")))
        shown = True
    csvs = sorted(folder.rglob("*.csv"))
    sweeps = [p for p in csvs if p.name == "sweep.csv"]
    for p in sweeps:
        table(p, str(p.relative_to(folder)))
        shown = True
    figs = D.figures(folder)
    if figs:
        st.markdown(f"**Figures** ({len(figs)})")
        figure_grid(figs, folder)
        shown = True
    others = [p for p in csvs if p.name != "sweep.csv"]
    if others:
        with st.expander(f"Other tables ({len(others)})"):
            for p in others:
                table(p, str(p.relative_to(folder)))
        shown = True
    if not shown:
        st.caption("No outputs in this folder yet.")


# ----------------------------------------------------------------------------------------------- jobs
def _job_line(j: dict) -> str:
    el = jobs.elapsed(j)
    t = f", {el:.0f} s" if el is not None else ""
    rc = f", exit {j['returncode']}" if j.get("returncode") is not None else ""
    return f"{j.get('sim', '?')} {j.get('action', '?')} — {j['status']}{t}{rc}"


@st.fragment(run_every="2s")
def jobs_panel(sim: str | None = None) -> None:
    """Running jobs (of one simulator, or all) with the last 30 log lines and a Stop button.
    When a job that was running finishes, the whole app reruns so the results panel refreshes."""
    run = jobs.running(sim)
    seen = f"_running_{sim}"
    now = {j["name"] for j in run}
    prev = st.session_state.get(seen, set())
    st.session_state[seen] = now
    if prev - now:
        st.rerun()
    if not run:
        st.caption("No running jobs.")
        return
    for j in run:
        with st.container(border=True):
            st.markdown(f"`{j['name']}` · {_job_line(j)}")
            st.code(jobs.tail(j["dir"], 30) or "(no output yet)", language="text")
            if st.button("Stop", key=f"stop_{j['name']}", icon=":material/stop:"):
                jobs.stop(j["dir"])
                st.rerun()


def jobs_table(sim: str | None = None, limit: int = 20) -> None:
    js = jobs.list_jobs(JOBS_DIR, sim)[:limit]
    if not js:
        st.caption("No jobs yet. Jobs started from the Force and Switching pages are written to res_sim/ui/.")
        return
    rows = [{"job": j["name"], "simulator": j.get("sim"), "action": j.get("action"), "status": j["status"],
             "duration [s]": jobs.elapsed(j), "exit code": j.get("returncode")} for j in js]
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={"duration [s]": st.column_config.NumberColumn(format="%.0f")})


def job_results(sim: str) -> None:
    """Job picker (newest first; the job just started is preselected) and its outputs."""
    js = jobs.list_jobs(JOBS_DIR, sim)
    if not js:
        st.caption("No jobs yet. Run, sweep or validate a scenario to see its outputs here.")
        return
    by_name = {j["name"]: j for j in js}
    key = f"{sim}_job_select"
    if st.session_state.get(key) not in by_name:
        st.session_state[key] = js[0]["name"]
    name = st.selectbox("Job", list(by_name), key=key, format_func=lambda n: f"{n} ({by_name[n]['status']})")
    j = by_name[name]
    st.caption(_job_line(j) + f" · `{rel(j['dir'])}`")
    with st.expander("Command and log", expanded=j["status"] in ("failed", "lost")):
        st.code(" ".join(j.get("cmd", [])), language="bash")
        st.code(jobs.tail(j["dir"], 200) or "(empty log)", language="text")
    if j["status"] == "running":
        st.info("Running. Outputs appear here as the job writes them.", icon=":material/hourglass_top:")
    show_folder(j["dir"])


# ----------------------------------------------------------------------------------------------- simulator page
def workers() -> int:
    return int(st.session_state.get("workers", 2))


def simulator_page(sim: str, quick: Callable[[dict, str], None]) -> None:
    """Scenario picker + YAML editor + quick parameters + Run/Sweep/Validate (left); jobs and results (right)."""
    files = S.scenario_files(sim)
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.subheader("Scenario")
        if not files:
            st.warning(f"No scenarios/{sim}_*.yaml files found.")
            return
        names = [f.name for f in files]
        choice = st.selectbox("Scenario file", names, key=f"{sim}_file", persist_state="session")
        text_key, loaded_key = f"{sim}_yaml", f"_{sim}_loaded"
        if st.session_state.get(loaded_key) != choice or text_key not in st.session_state:
            st.session_state[text_key] = (SCENARIOS / choice).read_text(encoding="utf-8")
            st.session_state[loaded_key] = choice
        st.text_area("Scenario YAML (SI units)", key=text_key, height=450, persist_state="session")
        text = st.session_state[text_key]
        cfg, err = S.parse(text)
        if err:
            st.error(f"YAML parse error: {err}", icon=":material/error:")
        else:
            with st.expander("Quick parameters"):
                quick(cfg, text_key)
        name = S.scenario_name(cfg, Path(choice).stem)
        busy = {a: bool(jobs.running(sim, a)) for a in ("run", "sweep", "validate")}
        enabled = {"run": cfg is not None, "sweep": S.has_sweep(cfg), "validate": S.has_validation(cfg)}
        cols = st.columns(3)
        for c, (action, label, icon) in zip(cols, (("run", "Run", ":material/play_arrow:"),
                                                   ("sweep", "Sweep", ":material/stacked_line_chart:"),
                                                   ("validate", "Validate", ":material/fact_check:"))):
            help_ = ("a job of this kind is running" if busy[action] else
                     None if enabled[action] else f"the scenario has no {'sweep' if action == 'sweep' else 'validation'} block")
            if c.button(label, key=f"{sim}_{action}", icon=icon, width="stretch",
                        type="primary" if action == "run" else "secondary",
                        disabled=busy[action] or not enabled[action], help=help_):
                d = jobs.start(sim, action, text, name, workers=workers())
                st.session_state[f"{sim}_job_select"] = d.name
                st.rerun()
        st.caption(f"Jobs run `python -m magsim.cli {sim} <action> <job>/scenario.yaml --out <job>` with "
                   f"{workers()} sweep workers and OMP_NUM_THREADS=1; outputs go to res_sim/ui/.")
    with right:
        st.subheader("Running jobs")
        jobs_panel(sim)
        st.subheader("Results")
        job_results(sim)


def yaml_key(text_key: str) -> str:
    """Short hash of the current YAML so quick-parameter widgets reset whenever the YAML changes."""
    return _key(st.session_state.get(text_key, ""))


def apply_patch(text_key: str, patch: Callable[[dict], dict]) -> None:
    """Callback: parse the editor text, patch it, write it back (comments are not kept)."""
    cfg, err = S.parse(st.session_state[text_key])
    if cfg is None:
        return
    try:
        st.session_state[text_key] = S.dump(patch(cfg))
    except ValueError as exc:
        st.session_state[f"{text_key}_patch_error"] = str(exc)
