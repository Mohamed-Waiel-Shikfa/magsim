"""Material figures (task 2): major loops with recoil / minor loops for N42, N52 and LNG37.

Writes res_sim/materials/loops_<grade>.{pdf,png} (full intrinsic loop J(H) with recoil loops, and the
second quadrant B(H), J(H) with recoil lines) and res_sim/materials/demag_LNG37.{pdf,png} (second quadrant
of the vendor-datasheet and FEMM-identified LNG37 models with the FEMM table points).

Usage: uv run python -m magsim.materials.plots [--out res_sim/materials]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magsim.constants import MU0
from magsim.materials import library as L
from magsim.plotting import IEEE_COL, OKABE_ITO, save, use_style

T0 = 293.15


def path_response(m, path):
    """J along a field path [A/m] from positive saturation, committing every point (quasi-static)."""
    st = m.init_state(1, "+sat")
    J = np.empty(len(path))
    for i, H in enumerate(path):
        J[i] = m.response(np.array([H]), np.zeros((1, 2)), st, T0)[0][0]
        st = m.commit(np.array([H]), st, T0)
    return J


def recoil_loop(m, H_d, H_top, n=200):
    """+sat -> H_d (major branch) -> H_top -> H_d: returns (H, J) of the minor loop part."""
    down = np.linspace(0.0, H_d, n)
    up = np.linspace(H_d, H_top, n)
    back = np.linspace(H_top, H_d, n)
    J = path_response(m, np.concatenate([down, up, back]))
    return np.concatenate([up, back]), J[n:]


def loops_figure(name, out):
    m = L.get_material(name)
    ch = m.characteristics()
    Hcj = ch["Hcj"]
    H = np.linspace(-3 * Hcj, 3 * Hcj, 1201)
    Jd = m.major_loop(H, "descending")[0]
    Ja = m.major_loop(H, "ascending")[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(IEEE_COL, IEEE_COL * 1.35))
    ax1.plot(H / 1e3, Jd, color=OKABE_ITO[0], label="major loop")
    ax1.plot(H / 1e3, Ja, color=OKABE_ITO[0])
    fracs = (0.6, 0.9, 1.0, 1.1)
    for c, f in zip(OKABE_ITO[1:], fracs):
        Hl, Jl = recoil_loop(m, -f * Hcj, 0.6 * Hcj)
        ax1.plot(Hl / 1e3, Jl, color=c, lw=0.8, label=f"recoil from $-{f:g}\\,H_{{cJ}}$")
    ax1.set_xlabel("H (kA/m)")
    ax1.set_ylabel("J (T)")
    ax1.axhline(0, color="0.7", lw=0.4)
    ax1.axvline(0, color="0.7", lw=0.4)
    ax1.legend(loc="lower right", fontsize=6, frameon=False, handlelength=1.2)
    # second quadrant: B and J with recoil lines back to H = 0
    H2 = np.linspace(-1.25 * Hcj, 0.0, 600)
    J2 = m.major_loop(H2, "descending")[0]
    ax2.plot(H2 / 1e3, J2, color=OKABE_ITO[0], label="J, major branch")
    ax2.plot(H2 / 1e3, MU0 * H2 + J2, color=OKABE_ITO[5], label="B, major branch")
    for c, f in zip(OKABE_ITO[1:], fracs):
        Hl, Jl = recoil_loop(m, -f * Hcj, 0.0)
        ax2.plot(Hl / 1e3, MU0 * Hl + Jl, color=c, lw=0.8)
    ax2.plot([-ch["Hcb"] / 1e3, -ch["Hcj"] / 1e3, 0.0, ch["H_d"] / 1e3], [0.0, 0.0, ch["Br"], ch["B_d"]], "o",
             color=OKABE_ITO[6], ms=3, label=r"model $H_{cB}$, $H_{cJ}$, $B_r$, $(BH)_{max}$ point")
    g = L.GRADES[name]
    pts = [(-v / 1e3, 0.0) for v in (g.Hcb, g.Hcj) if v is not None] + [(0.0, g.Br)]
    ax2.plot(*zip(*pts), "x", color=OKABE_ITO[0], ms=4, label="datasheet")
    ax2.set_xlim(-1.25 * Hcj / 1e3, 0.0)
    ax2.set_ylim(0.0, 1.08 * ch["Br"])
    ax2.set_xlabel("H (kA/m)")
    ax2.set_ylabel("B, J (T)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, frameon=False)
    stem = out / f"loops_{name}"
    save(fig, stem)
    plt.close(fig)
    return stem


def demag_lng37(out):
    H_f, B_f = L.read_curve("femm_lng37.csv")
    fig, ax = plt.subplots()
    H = np.linspace(-50e3, 0.0, 600)
    for name, c, label in (("LNG37-femm", OKABE_ITO[5], "model identified on FEMM table"),
                           ("LNG37", OKABE_ITO[1], "model identified on vendor datasheet")):
        m = L.get_material(name)
        ax.plot(H / 1e3, MU0 * H + m.major_loop(H)[0], color=c, label=label)
    ax.plot(H_f / 1e3, B_f, "o", color=OKABE_ITO[0], ms=3, label="FEMM 'Cast Alnico 5 (LNG37)'")
    g = L.GRADES["LNG37"]
    ax.plot([-g.Hcb / 1e3, 0.0], [0.0, g.Br], "x", color=OKABE_ITO[6], ms=4, label="vendor $H_{cB}$, $B_r$")
    rms = L.grade_info("LNG37-femm")["fit"]["rms"]
    ax.text(0.03, 0.95, f"FEMM fit RMS = {rms * 1e3:.2f} mT", transform=ax.transAxes, va="top")
    ax.set_xlim(-50, 0)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel("H (kA/m)")
    ax.set_ylabel("B (T)")
    ax.legend(loc="best", fontsize=6.5, framealpha=0.9)
    stem = out / "demag_LNG37"
    save(fig, stem)
    plt.close(fig)
    return stem


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "res_sim" / "materials"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    use_style()
    stems = [loops_figure(n, out) for n in ("N42", "N52", "LNG37")] + [demag_lng37(out)]
    for s in stems:
        print(f"wrote {s}.pdf, {s}.png")


if __name__ == "__main__":
    main()
