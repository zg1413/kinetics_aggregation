#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from plot_cluster_common import get_system, setup_pgf

setup_pgf(font_size=25, height_factor=1.0)

FOLDER = Path("eight_rescal")
TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
TIME_LABELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def main() -> None:
    params = get_system(str(FOLDER))
    #_ = params["cell"]
    #_ = params["N_Na"] + params["N_EMIM"] * params["n_emim"] + params["N_TFSI"] * params["n_tfsi"]
    om = params["N_Na"] * 1 + params["N_EMIM"] * 7 + params["N_TFSI"] * 7

    clm = np.load(FOLDER / "clm_33.npy")

    for idx, t_index in enumerate(TIMES):
        plt.figure()
        label = rf"MD, $x = 0.5$, $t =$ {TIME_LABELS[idx]} ps" if idx == 0 else rf"$t =$ {TIME_LABELS[idx]} ps"
        plt.text(0.5, 13, label, fontsize=20)

        image = plt.imshow(clm[:16, :16, idx].T / om, cmap="Blues", interpolation="nearest", origin="lower", vmin=-9, vmax=-1)
        cbar = plt.colorbar(image, fraction=0.045)
        cbar.ax.set_ylabel(r"$\ln(c_{lm})$", fontsize=25)
        plt.xlabel(r"$l$ Na$^+$ cations", fontsize=25)
        plt.ylabel(r"$m$ TFSI$^-$ anions", fontsize=25)
        plt.locator_params("x", tight=True, nbins=5)
        plt.locator_params("y", tight=True, nbins=5)
        plt.tick_params(axis="both", which="major", labelsize=25, width=2, length=5, direction="in", right=True, top=True)
        plt.rcParams["axes.linewidth"] = 2
        plt.ylim(-0.5, 15 - 0.5)
        plt.xlim(-0.5, 15 - 0.5)
        plt.tight_layout()
        plt.savefig(FOLDER / f"clm_Na_ALL_{t_index}_33.pdf")
        plt.close()


if __name__ == "__main__":
    main()
