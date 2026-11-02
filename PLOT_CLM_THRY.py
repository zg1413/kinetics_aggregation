#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from math import lgamma
from scipy.special import factorial
from plot_cluster_common import get_system, setup_pgf

setup_pgf(font_size=25, height_factor=1.0)

TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
TIME_LABELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
XI_P = 1
XI_CP = 7
XI_M = 7
FP = 4
FM = 3

@dataclass(frozen=True)
class TheoryCase:
    folder: str
    p_pm: np.ndarray
    p_mp: np.ndarray
    psi_p: float
    psi_m: float
    lam: np.ndarray
    N_Na: int
    N_EMIM: int
    N_TFSI: int

def build_case(folder: str) -> TheoryCase:
    params = get_system(folder)
    N_Na = params["N_Na"]
    N_EMIM = params["N_EMIM"]
    N_TFSI = params["N_TFSI"]
    t_vals = np.array(TIME_LABELS, dtype=float)

    om = N_Na * XI_P + N_EMIM * XI_CP + N_TFSI * XI_M
    psi_p = FP * N_Na / om
    psi_m = FM * N_TFSI / om

    if folder == "eigth_rescal":
        g = 0.40147563#units of inverse ps
        p0 = 0.09#initial condition
        pi = 0.81015#final condition
        lamda = (1 - p0 / pi) / (1 - p0 * pi)
        p_pm = pi * (1 - lamda * np.exp(-g * t_vals)) / (1 - pi * pi * lamda * np.exp(-g * t_vals))
        p_mp = p_pm.copy()#as this is for the symmetric case
        lam = p_pm/(1-p_pm)/(1-p_mp)/psi_m#the association constant changes in time, lamda does not ... 
    else:
        raise ValueError(f"Unsupported theory case: {folder!r}")

    return TheoryCase(
        folder=folder,
        p_pm=p_pm,
        p_mp=p_mp,
        psi_p=psi_p,
        psi_m=psi_m,
        lam=lam,
        N_Na=N_Na,
        N_EMIM=N_EMIM,
        N_TFSI=N_TFSI,
    )

def compute_clm(case: TheoryCase) -> np.ndarray:
    clm = np.zeros((17, 17, len(TIME_LABELS)))
    for i in range(16):
        for j in range(16):
            for q in range(len(TIME_LABELS)):
                if i + j == 0:
                    continue
                if FP * i - i - j + 1 < 0 or FM * j - i - j + 1 < 0:
                    continue
                if i + j < 5:
                    wlm = (
                        factorial(FP * i - i)
                        * factorial(FM * j - j)
                        / factorial(i)
                        / factorial(j)
                        / factorial(FP * i - i - j + 1)
                        / factorial(FM * j - i - j + 1)
                        * (1 - case.p_pm[q]) ** (FP * i)
                        * (1 - case.p_mp[q]) ** (FM * j)
                    )
                else:
                    log_wlm = (
                        lgamma(FP * i - i)
                        + lgamma(FM * j - j)
                        - lgamma(i)
                        - lgamma(j)
                        - lgamma(FP * i - i - j + 1)
                        - lgamma(FM * j - i - j + 1)
                        + FP * i * np.log(1 - case.p_pm[q])
                        + FM * j * np.log(1 - case.p_mp[q])
                    )
                    wlm = np.exp(log_wlm)

                clm[i, j, q] = wlm * (case.psi_p * case.lam[q]) ** i * (case.psi_m * case.lam[q]) ** j / case.lam[q]

    clm[0, 0, :] = 0
    return clm

def main() -> None:
    folder = Path("kinetics_rescal_05")
    case = build_case(folder.name)
    clm = compute_clm(case)

    om = case.N_Na * XI_P + case.N_EMIM * XI_CP + case.N_TFSI * XI_M
    print((case.N_Na * XI_P + case.N_TFSI * XI_M) / om)

    for q in range(len(TIME_LABELS)):
        philm = np.zeros((17, 17))
        for i in range(16):
            for j in range(16):
                philm[i, j] = (i * XI_P + j * XI_M) * clm[i, j, q]
        print(np.sum(philm))

    print(np.sum(clm[:, :, 0]))

    for idx, t_index in enumerate(TIMES):
        plt.figure()
        title = rf"Theory, $x = 0.5$, $t =$ {TIME_LABELS[idx]} ps" if idx == 0 else rf"$t =$ {TIME_LABELS[idx]} ps"
        plt.text(0.5, 13, title, fontsize=20)
        image = plt.imshow(np.log(clm[:, :, idx].T), cmap="Blues", interpolation="nearest", origin="lower", vmin=-9, vmax=-1)
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
        plt.savefig(folder / f"clm_Na_ALL_{t_index}_33_THRY.pdf")
        plt.close()


if __name__ == "__main__":
    main()
