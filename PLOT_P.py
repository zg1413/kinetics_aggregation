#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from plot_p_common import configure_matplotlib, get_system, time_dependence

configure_matplotlib(fontsize=20)

T_STEPS = 700
T_MAX_STEPS = 699
FOLDER = "eigth_rescal/"
FP = 4
FM = 3

P0_FIXED = 0.09
PI_FIXED = 0.87

system = get_system(FOLDER)
_ = system

coord_na_tfsi = np.load(Path(FOLDER) / "coordination_NaO_33.npy")
coord_na_tfsi_avg = coord_na_tfsi.sum(axis=0) / coord_na_tfsi.shape[0]
p_pm = coord_na_tfsi_avg / FP
print(p_pm[-1], p_pm[0])

coord_tfsi_na = np.load(Path(FOLDER) / "coordination_ONa_33.npy")
coord_tfsi_na_avg = coord_tfsi_na.sum(axis=0) / coord_tfsi_na.shape[0]
p_mp = coord_tfsi_na_avg / FM

t = np.linspace(0, int(700 * 100 / 1000), 700)

param_mp, _ = curve_fit(
    lambda tt, G: time_dependence(tt, 0.0, 0.0, G, P0_FIXED, PI_FIXED),
    t[:T_MAX_STEPS],
    p_mp[:T_MAX_STEPS],
    p0=[1.0],
)
print(param_mp, coord_tfsi_na_avg[0] / FM, (coord_tfsi_na_avg[T_MAX_STEPS] / FM) )

eq_tfsi = time_dependence(t, 0.0, 0.0, param_mp[0], P0_FIXED, PI_FIXED) * FM

plt.figure()
plt.plot(t, coord_tfsi_na_avg, color="blue", linestyle="-.", label=r"$-+$, MD")
plt.plot(t, eq_tfsi, color="blue", linestyle="--", label=r"$-+$, Theory")
plt.xlim(t[0] - 1.5, t[T_MAX_STEPS] + 1.5)
plt.ylim(0, 5)
plt.legend(loc="lower right", prop={"size": 18}, frameon=False, title_fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=20, width=2, length=5, direction="in", right=True, top=True)
plt.rcParams["axes.linewidth"] = 2
plt.locator_params("x", tight=True, nbins=10)
plt.locator_params("y", tight=True, nbins=10)
plt.ylabel(r"$CN_{ij}$", fontsize=20)
plt.xlabel(r"$t$ / ps", fontsize=20)
plt.tight_layout()
plt.savefig(Path(FOLDER) / "test_TFSI_long.pdf")
plt.show()
