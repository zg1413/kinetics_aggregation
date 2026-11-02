#from __future__ import annotations
from pathlib import Path
import numpy as np
from probability_common import get_system, load_xyz, periodic_shifts, unique_hits

CUT_OFF = 3.3#real space cutoff in Angstom
FOLDER = "eigth_rescal/"
T = 700#number of time steps considered

def main() -> None:
    params = get_system(FOLDER)
    cell_lo, cell_hi = params["cell"]
    n_na = params["N_Na"]
    n_tfsi = params["N_TFSI"]

    #importting positions
    xyz_na = load_xyz(FOLDER, "XYZ_Na_PBC.npy")
    xyz_tfsi_o = load_xyz(FOLDER, "XYZ_TFSI_O_PBC.npy")

    #empty arrays for saving coordination information
    coord = np.zeros((n_na, T), dtype=np.int32)
    coord_tot = np.zeros((n_na, T), dtype=np.int32)
    coord_tfsi = np.zeros((n_tfsi, T), dtype=np.int32)
    coord_tot_tfsi = np.zeros((n_tfsi, T), dtype=np.int32)

    #looping over all times
    for i in range(T):
        print(i)

        #finding what is coordinated to Na
        for j in range(n_na):
            center = xyz_na[j, :, i]
            neighbors = []

            for shift in periodic_shifts(center, cell_lo, cell_hi, CUT_OFF):
                shifted = xyz_tfsi_o[:, :, :, i] + shift
                distances = np.linalg.norm(center - shifted, axis=2)
                hits = np.where(distances < CUT_OFF)
                if hits[0].size:
                    neighbors.extend(hits[0].tolist())

            unique_neighbors = unique_hits(neighbors)
            coord[j, i] = len(unique_neighbors)
            coord_tot[j, i] = len(neighbors)

        #finding what is coordinated to O's in tfsi
        for j in range(n_tfsi):
            for qq in range(4):
                center = xyz_tfsi_o[j, qq, :, i]
                neighbors = []

                for shift in periodic_shifts(center, cell_lo, cell_hi, CUT_OFF):
                    shifted = xyz_na[:, :, i] + shift
                    distances = np.linalg.norm(center - shifted, axis=1)
                    hits = np.where(distances < CUT_OFF)[0]
                    if hits.size:
                        neighbors.extend(hits.tolist())

                unique_neighbors = unique_hits(neighbors)
                coord_tfsi[j, i] += len(unique_neighbors)
                coord_tot_tfsi[j, i] += len(neighbors)

    #saving to binary files
    np.save(Path(FOLDER) / "coordination_NaO_33", coord)
    np.save(Path(FOLDER) / "coordination_NaO_tot_33", coord_tot)
    np.save(Path(FOLDER) / "coordination_ONa_33", coord_tfsi)
    np.save(Path(FOLDER) / "coordination_ONa_tot_33", coord_tot_tfsi)


if __name__ == "__main__":
    main()
