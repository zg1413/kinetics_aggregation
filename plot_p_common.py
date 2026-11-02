from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import matplotlib as mpl
import numpy as np

def figsize(scale: float) -> list[float]:
    fig_width_pt = 469.755
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean * 1.25
    return [fig_width, fig_height]

def configure_matplotlib(fontsize: int = 20) -> None:
    mpl.use("pgf")
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "figure.figsize": figsize(1),
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
        ],
    }
    mpl.rcParams.update(pgf_with_latex)

SYSTEMS: Dict[str, Dict[str, float | int | Tuple[float, float]]] = {
    "eigth_rescal": {
        "cell": (8.1109426679660146e-02, 4.9918890573305006e+01),
        "N_Na": 300,
        "N_EMIM": 100,
        "N_TFSI": 400,
        "n_emim": 19,
        "n_tfsi": 15,
    },
}

def get_system(folder: str) -> dict:
    key = Path(folder).name
    if key not in SYSTEMS:
        raise ValueError(f"Unknown folder preset: {folder!r}")
    return SYSTEMS[key]

def time_dependence(t: np.ndarray, p0: float, pi: float, G: float, p0_fixed: float, pi_fixed: float) -> np.ndarray:
    """Time-dependence model used by the original scripts."""
    # The original scripts hard-coded p0 and pi inside the function.
    # This version keeps that behavior configurable
    p0 = p0_fixed
    pi = pi_fixed
    lam = (1 - p0 / pi) / (1 - p0 * pi)
    return pi * (1 - lam * np.exp(-G * t)) / (1 - pi * pi * lam * np.exp(-G * t))
