from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import numpy as np

SYSTEMS = {
    "eigth_rescal": {
        "cell": (8.1109426679660146e-02, 4.9918890573305006e+01),
        "N_Na": 300,
        "N_EMIM": 100,
        "N_TFSI": 400,
        "n_emim": 19,
        "n_tfsi": 15,
    },
}

def normalize_folder(folder: str) -> str:
    return Path(folder).name.rstrip("/")

def get_system(folder: str) -> dict:
    key = normalize_folder(folder)
    if key not in SYSTEMS:
        raise ValueError(f"Unknown folder preset: {folder!r}")
    return SYSTEMS[key].copy()

def setup_pgf(*, font_size: int = 25, width_pt: float = 469.755, height_factor: float = 1.0) -> None:
    """Apply the LaTeX/PGF plotting style used in the original scripts."""
    mpl.use("pgf")

    def figsize(scale: float):
        inches_per_pt = 1.0 / 72.27
        golden_mean = (np.sqrt(5.0) - 1) / 2.0
        fig_width = width_pt * inches_per_pt * scale
        fig_height = fig_width * golden_mean * height_factor
        return [fig_width, fig_height]

    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "figure.figsize": figsize(1),
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
        ],
    }
    mpl.rcParams.update(pgf_with_latex)
