from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Plotting setup (LaTeX/PGF style, as in the original script)
# If you do not have a LaTeX toolchain installed, comment out mpl.use('pgf').
mpl.use('pgf')

def figsize(scale: float) -> list[float]:
    """Return a square figure size scaled to LaTeX text width."""
    fig_width_pt = 469.755
    inches_per_pt = 1.0 / 72.27
    fig_width = fig_width_pt * inches_per_pt * scale
    return [fig_width, fig_width]


PGF_WITH_LATEX = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 25,
    "legend.fontsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "figure.figsize": figsize(1),
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ],
}

mpl.rcParams.update(PGF_WITH_LATEX)

FOLDER = Path("eigth_rescal")
CUT_OFF = 3.0
TIME_WINDOWS = [5, 50, 100, 200]
TIME_LABELS_PS = [0.5, 5, 10, 20]
N_FRAMES = 10
SEARCH_MARGIN = 4.0
N_BINS = 1000
OUTPUT_PDF = FOLDER / "pair_10s.pdf"

def load_system_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load coordinates and return cell parameters and coordinate arrays."""
    cell_param = np.array([8.1109426679660146e-02, 4.9918890573305006e+01])
    xyz_na = np.load(FOLDER / "XYZ_Na_PBC.npy")
    xyz_tfsi_o = np.load(FOLDER / "XYZ_TFSI_O_PBC.npy")
    return cell_param, xyz_na, xyz_tfsi_o

def build_distance_sample(
    xyz_na: np.ndarray,
    xyz_tfsi_o: np.ndarray,
    frame_idx: int,
    min_xyz: float,
    max_xyz: float,
) -> list[float]:
    """Collect Na-O distances for Na atoms in the inner part of the box."""
    distances: list[float] = []
    n_na = xyz_na.shape[0]
    n_tfsi = xyz_tfsi_o.shape[0]

    for j in range(n_na):
        x, y, z = xyz_na[j, :, frame_idx]
        if not (min_xyz < x < max_xyz and min_xyz < y < max_xyz and min_xyz < z < max_xyz):
            continue

        na_pos = xyz_na[j, :, frame_idx]
        for p in range(n_tfsi):
            for q in range(4):
                tfsi_pos = xyz_tfsi_o[p, q, :, frame_idx]
                distances.append(np.linalg.norm(na_pos - tfsi_pos))

    return distances

def compute_gr(distances: list[float], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute g(r) from a list of distances."""
    if not distances:
        raise ValueError("No distances were collected; check the selection window and cutoff.")

    bins = np.linspace(0, max(distances), N_BINS)
    hist, edges = np.histogram(distances, bins)
    delta_r = edges[1] - edges[0]
    r_mid = 0.5 * (edges[:-1] + edges[1:])

    # Keep the original normalization.
    g_r = hist / (8 * np.pi * delta_r * r_mid**2) / n_frames
    return r_mid, g_r

def main() -> None:
    cell_param, xyz_na, xyz_tfsi_o = load_system_params()
    min_box, max_box = cell_param
    min_xyz = min_box + SEARCH_MARGIN
    max_xyz = max_box - SEARCH_MARGIN

    r_series = []
    g_series = []

    for start in TIME_WINDOWS:
        all_distances: list[float] = []
        for offset in range(N_FRAMES):
            frame_idx = start - 5 + offset
            all_distances.extend(
                build_distance_sample(
                    xyz_na=xyz_na,
                    xyz_tfsi_o=xyz_tfsi_o,
                    frame_idx=frame_idx,
                    min_xyz=min_xyz,
                    max_xyz=max_xyz,
                )
            )

        r_mid, g_r = compute_gr(all_distances, N_FRAMES)
        r_series.append(r_mid)
        g_series.append(g_r)

    colors = ["black", "red", "grey", "purple"]
    styles = [":", "-.", "-", "--"]

    plt.figure()
    for idx, (r_mid, g_r) in enumerate(zip(r_series, g_series)):
        label = rf"$t = {TIME_LABELS_PS[idx]}$ ps"
        plt.plot(r_mid, g_r, color=colors[idx], linestyle=styles[idx], label=label)

    plt.xlim(1, 6)
    plt.ylim(-0.3, 8.7)
    plt.vlines(3.3, -1, 10, color="black")
    plt.xlabel(r"$r /$ \AA", fontsize=25)
    plt.ylabel(r"$g(r)$", fontsize=25)
    plt.locator_params("x", tight=True, nbins=5)
    plt.locator_params("y", tight=True, nbins=5)
    plt.legend(loc="upper right", prop={"size": 18}, ncol=1, frameon=False, title_fontsize=25)
    plt.tick_params(axis="both", which="major", labelsize=25, width=2, length=5, direction="in", right=True, top=True)
    plt.rcParams["axes.linewidth"] = 2
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    plt.show()


if __name__ == "__main__":
    main()
