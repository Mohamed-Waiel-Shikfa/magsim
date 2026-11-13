"""Publication figure style and helpers: line plots and heatmap + contours over sweep DataFrames.

No titles inside figures; callers pass axis labels with units.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

IEEE_COL = 3.5  # single-column width [in]
IEEE_PAGE = 7.16  # double-column (full page) width [in]

# Okabe & Ito (2008) colour-blind-safe palette.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def use_style() -> None:
    """Apply the publication rcParams (sans-serif 8 pt, Okabe-Ito cycle, thin open spines)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "figure.figsize": (IEEE_COL, IEEE_COL * 0.75),
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed TrueType so text stays editable in the PDF
    })


def save(fig, path_stem: str | Path) -> tuple[Path, Path]:
    """Write <stem>.pdf (vector) and <stem>.png (300 dpi); return both paths."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    return pdf, png


def _axes(ax):
    if ax is None:
        _, ax = plt.subplots()
    return ax


def line(df: pd.DataFrame, x: str, ys: str | list[str], hue: str | None = None,
         xlabel: str | None = None, ylabel: str | None = None, ax=None):
    """One line per y column (and per hue level if given), sorted by x. Returns the Axes."""
    ax = _axes(ax)
    ys = [ys] if isinstance(ys, str) else list(ys)
    groups = [(None, df)] if hue is None else list(df.groupby(hue, sort=True))
    for level, sub in groups:
        sub = sub.sort_values(x)
        for y in ys:
            lvl = f"{level:.4g}" if isinstance(level, (int, float)) else str(level)
            parts = ([y] if len(ys) > 1 else []) + ([f"{hue} = {lvl}"] if hue else [])
            ax.plot(sub[x], sub[y], marker="o", label=", ".join(parts) or None)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or (ys[0] if len(ys) == 1 else ""))
    if len(ys) > 1 or hue is not None:
        ax.legend()
    return ax


def heatmap(df: pd.DataFrame, x: str, y: str, z: str, levels=None, ax=None,
            xlabel: str | None = None, ylabel: str | None = None, zlabel: str | None = None):
    """pcolormesh of z on the (x, y) grid with labelled contour lines and a colourbar. Returns the Axes."""
    ax = _axes(ax)
    grid = df.pivot_table(index=y, columns=x, values=z, aggfunc="mean").sort_index().sort_index(axis=1)
    X, Y, Z = grid.columns.to_numpy(float), grid.index.to_numpy(float), grid.to_numpy(float)
    mesh = ax.pcolormesh(X, Y, Z, shading="nearest", cmap="viridis")
    if len(X) > 1 and len(Y) > 1:
        cs = ax.contour(X, Y, Z, levels=levels if levels is not None else 6, colors="white", linewidths=0.6)
        ax.clabel(cs, fontsize=6, fmt="%.3g")
    cbar = ax.figure.colorbar(mesh, ax=ax)
    cbar.set_label(zlabel or z)
    cbar.outline.set_linewidth(0.6)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    return ax
