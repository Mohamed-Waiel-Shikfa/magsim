import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magsim import plotting


def test_style_and_save(tmp_path):
    plotting.use_style()
    assert plt.rcParams["font.size"] == 8
    assert plt.rcParams["axes.prop_cycle"].by_key()["color"][1] == "#E69F00"
    assert not plt.rcParams["axes.spines.top"]
    x = np.linspace(0, 1, 5)
    df = pd.DataFrame({"gap": np.tile(x, 2), "F": np.r_[x, 2 * x], "T": np.r_[x**2, x], "V": np.repeat([100, 200], 5)})

    fig, axs = plt.subplots(1, 2, figsize=(plotting.IEEE_PAGE, 2.5))
    plotting.line(df, "gap", "F", hue="V", xlabel="gap (mm)", ylabel="force (N)", ax=axs[0])
    plotting.line(df[df.V == 100], "gap", ["F", "T"], xlabel="gap (mm)", ax=axs[1])
    pdf, png = plotting.save(fig, tmp_path / "sub" / "lines")
    plt.close(fig)
    assert pdf.stat().st_size > 0 and png.stat().st_size > 0
    assert axs[0].get_title() == ""


def test_heatmap(tmp_path):
    plotting.use_style()
    rows = [{"x": x, "y": y, "z": x * y} for x, y in itertools.product(np.linspace(0, 1, 6), np.linspace(1, 2, 4))]
    fig, ax = plt.subplots()
    plotting.heatmap(pd.DataFrame(rows), "x", "y", "z", levels=[0.5, 1.0], ax=ax)
    pdf, png = plotting.save(fig, tmp_path / "heat")
    plt.close(fig)
    assert pdf.exists() and png.stat().st_size > 0
