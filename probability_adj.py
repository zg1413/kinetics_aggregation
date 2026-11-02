#from __future__ import annotations
from pathlib import Path
import numpy as np
from probability_common import get_system, load_xyz, periodic_shifts, unique_hits

CUT_OFF = 3.3#real space cutoff in Angstrom
FOLDER = "eight_rescal/"
TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]#time steps to consider - don't do all of them, as it will become expensive

def main() -> None:
    """function for calculating adjacency matrix between Na and TFSI"""
    params = get_system(FOLDER)
    cell_lo, cell_hi = params["cell"]
    n_na = params["N_Na"]
    n_tfsi = params["N_TFSI"]

    #importing positions
    xyz_na = load_xyz(FOLDER, "XYZ_Na_PBC.npy")
    xyz_tfsi_o = load_xyz(FOLDER, "XYZ_TFSI_O_PBC.npy")

    #create adjacency matrix to populate
    adj_matrix = np.zeros((n_na, n_tfsi, len(TIMES)), dtype=np.int8)

    #looping over selected times
    for ti, t in enumerate(TIMES):
        for j in range(n_na):
            center = xyz_na[j, :, t]
            neighbors = []

            for shift in periodic_shifts(center, cell_lo, cell_hi, CUT_OFF):
                shifted = xyz_tfsi_o[:, :, :, t] + shift 
                distances = np.linalg.norm(center - shifted, axis=2)  
                hits = np.where(distances < CUT_OFF)
                if hits[0].size:
                    neighbors.extend(hits[0].tolist())

            for idx in unique_hits(neighbors):
                adj_matrix[j, idx, ti] = 1

    #saving adjacency matrix
    np.save(Path(FOLDER) / "adj_matrix_33", adj_matrix)

if __name__ == "__main__":
    main()
