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
FOLDER = "eightht_rescale/" 
FP = 4 #cation functionality
FM = 3 #anion functionality

#this will only work for a symmetric case
P0_FIXED = 0.09 #initial probablility - has to be extracted from the MD simulations
PI_FIXED = 0.87 #final probablility - has to be extracted from the MD simulations

system = get_system(FOLDER)

coord_na_tfsi = np.load(Path(FOLDER) / "coordination_NaO_33.npy") #importing data
coord_na_tfsi_avg = coord_na_tfsi.sum(axis=0) / coord_na_tfsi.shape[0] #taking the average
p_pm = coord_na_tfsi_avg / FP #converting to probability

coord_tfsi_na = np.load(Path(FOLDER) / "coordination_ONa_33.npy") #importing data
coord_tfsi_na_avg = coord_tfsi_na.sum(axis=0) / coord_tfsi_na.shape[0] #taking the average
p_mp = coord_tfsi_na_avg / FM #converting to probability

t = np.linspace(0, int(700 * 100 / 1000), 700) #list of times to evaluate theory

############################
#Fitting the theory equation - cations
############################
param_pm, _ = curve_fit(
    lambda tt, G: time_dependence(tt, 0.0, 0.0, G, P0_FIXED, PI_FIXED),
    t[:T_MAX_STEPS],
    p_pm[:T_MAX_STEPS],
    p0=[1.0],
)

#printing values
print(param_pm, coord_na_tfsi_avg[0] / FP, (coord_na_tfsi_avg[T_MAX_STEPS] / FP))

############################
#Fitting the theory equation - anionss
############################
param_mp, _ = curve_fit(
    lambda tt, G: time_dependence(tt, 0.0, 0.0, G, P0_FIXED, PI_FIXED),
    t[:T_MAX_STEPS],
    p_mp[:T_MAX_STEPS],
    p0=[1.0],
)

#printing values
print(param_mp, coord_tfsi_na_avg[0] / FM, (coord_tfsi_na_avg[T_MAX_STEPS] / FM))

#evaluating the theory at different times
eq = time_dependence(t, 0.0, 0.0, param_pm[0], P0_FIXED, PI_FIXED) * FP
eq_tfsi = time_dependence(t, 0.0, 0.0, param_mp[0], P0_FIXED, PI_FIXED) * FM

plt.figure()
plt.plot(t, p_pm * p_mp, color="black", linestyle="-", label=r"MD")
plt.plot(t, eq * eq_tfsi, color="black", linestyle=":", label=r"Theory")
plt.xlim(t[0] - 0.25, t[T_MAX_STEPS] + 0.25)
plt.ylim(0, 1)
plt.hlines(1 / 3 / 2, -1, 20, color="black", linestyle="-.", label=r"Percolation")
plt.legend(loc="upper left", prop={"size": 18}, frameon=False, title_fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=20, width=2, length=5, direction="in", right=True, top=True)
plt.rcParams["axes.linewidth"] = 2
plt.locator_params("x", tight=True, nbins=10)
plt.locator_params("y", tight=True, nbins=6)
plt.ylabel(r"$p_{+-}p_{-+}$", fontsize=20)
plt.xlabel(r"$t$ / ps", fontsize=20)
plt.tight_layout()
plt.savefig(Path(FOLDER) / "percolation_4_eq_700_both_4_fit_175_33.pdf")
plt.show()
