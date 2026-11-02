#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plot_cluster_common import get_system, setup_pgf

setup_pgf(font_size=25, height_factor=1.45)

FOLDER = Path("eight_rescal")
TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
TIME_LABELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
COLORS = ["purple", "grey", "blue", "black"]
MARKERS = ["o", "s", "^", "v", "d"]

def main() -> None:
    params = get_system(str(FOLDER))
    cell_lo, cell_hi = params["cell"]
    #_ = params["N_Na"] + params["N_EMIM"] * params["n_emim"] + params["N_TFSI"] * params["n_tfsi"]
    #_ = cell_hi - cell_lo

    for idx, t_index in enumerate(TIMES):
        bonds = np.load(FOLDER / f"bonds_clm_{t_index}_33.npy")
        cbd = bonds[:, 0].astype(int)
        lpm = (bonds[:, 1] + bonds[:, 2]).astype(int)

        #matrix for collating CBD data to allow for more efficient plotting
        arr = np.zeros((1000, 1000), dtype=int)
        np.add.at(arr, (cbd, lpm), 1)
        rows, cols = np.where(arr != 0)

        if len(rows) == 0:
            continue

        plt.figure()
        for row, col in zip(rows, cols):
            if col > 1:
                plt.scatter(
                    col,
                    row / col,
                    s=arr[row, col] * 20,
                    marker=MARKERS[0],
                    facecolor="none",
                    color=COLORS[0],
                )

        lpm_axis = np.linspace(1, 80, 80)
        cbd_axis = (lpm_axis - 1) / lpm_axis
        title = rf"$x = 0.5$, $t =$ {TIME_LABELS[idx]} ps" if idx == 0 else rf"$t =$ {TIME_LABELS[idx]} ps"
        plt.text(1, 1.35, title, fontsize=20)
        plt.plot(lpm_axis, cbd_axis, linewidth=2, color="black", label=r"$(l+m-1)/(l+m)$")
        plt.xlabel(r"$l+m$", fontsize=30)
        plt.ylabel(r"CBD", fontsize=30)
        plt.locator_params("x", tight=True, nbins=6)
        plt.locator_params("y", tight=True, nbins=4)
        plt.legend(loc="lower right", prop={"size": 25}, frameon=False, title_fontsize=20)
        plt.tick_params(axis="both", which="major", labelsize=30, width=2, length=5, direction="in", right=True, top=True)
        plt.rcParams["axes.linewidth"] = 2
        plt.ylim(0, 1.5)
        plt.xlim(-1, 30)
        plt.tight_layout()
        plt.savefig(FOLDER / f"CBD_{t_index}_33.pdf")
        plt.close()


if __name__ == "__main__":
    main()
