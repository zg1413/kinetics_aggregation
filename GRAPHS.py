#!/usr/bin/env python3
"""Build cluster statistics from an adjacency matrix."""

from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

FOLDER = Path("kinetics_rescal_05")
TIMES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]#don't do all times, as it will be expensive

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

def system_params(folder: Path):
    key = folder.name
    if key not in SYSTEMS:
        raise ValueError(f"Unknown folder preset: {folder!s}")
    return SYSTEMS[key]

def main() -> None:
    params = system_params(FOLDER)
    cell = np.array(params["cell"], dtype=float)
    N_Na = params["N_Na"]
    N_TFSI = params["N_TFSI"]

    adjacency_lt = np.load(FOLDER / "adj_matrix_33.npy")
    n_times = len(TIMES)
    n_nodes = N_Na + N_TFSI

    #construct full adjacency matrix
    adjacency = np.zeros((n_nodes, n_nodes, n_times), dtype=adjacency_lt.dtype)
    adjacency[:N_Na, N_Na:N_Na + N_TFSI, :] = adjacency_lt
    adjacency += adjacency.transpose(1, 0, 2)

    #initiate empty cluster distribution
    clm = np.zeros((N_Na + 1, N_TFSI + 1, n_times), dtype=int)

    #loop over select times
    for ti in range(n_times):
        graph = csr_matrix(adjacency[:, :, ti])
        n_components, labels = connected_components(graph)
        bonds = np.zeros((n_components, 3), dtype=float)

        for comp_id in range(n_components):
            indices = np.where(labels == comp_id)[0]
            n_li = int(np.sum(indices < N_Na))
            n_mol = int(np.sum(indices >= N_Na))
            bonded_edges = 0

            for idx in indices:
                bonded_edges += int(np.sum(adjacency[idx, :, ti]))

            #print(n_li, n_mol)
            clm[n_li, n_mol, ti] += 1
            bonds[comp_id, 0] = bonded_edges / 2.0
            bonds[comp_id, 1] = n_li
            bonds[comp_id, 2] = n_mol

        np.save(FOLDER / f"bonds_clm_{ti}_33.npy", bonds)

    np.save(FOLDER / "clm_33.npy", clm)


if __name__ == "__main__":
    main()
