from pathlib import Path
import numpy as np

T = 700 #number of time steps
FOLDER = Path("eigth_rescal") #folder where the trajectory is saved 
DUMP_FILE = FOLDER / "dump.lammpstrj" 

#defining box sizes, number of each species, number of atoms in each of those
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

def wrap_pbc(value: float, lo: float, hi: float) -> float:
    """function to wrap a coordinate into the unit cell"""
    box = hi - lo
    return ((value - lo) % box) + lo#does this actually do the right thing tohugh? 

def get_system_params(folder_name: str):
    """function to get system parameters"""
    key = folder_name.rstrip("/").split("/")[-1]
    if key not in SYSTEMS:
        raise ValueError(f"Unknown folder preset: {folder_name!r}")
    return SYSTEMS[key]

#def parse_dump_lines(lines, n_steps, n_atoms_per_frame, header_lines=9):
#    """Extracting one frame at a time from a LAMMPS trajectory dump."""
#    for step in range(n_steps):
#        start = header_lines + step * (n_atoms_per_frame + header_lines)
#        end = start + n_atoms_per_frame
#        if end > len(lines):
#            raise ValueError(
#                f"Dump file ended early at step {step}. "
#                f"Expected at least {end} lines, found {len(lines)}."
#            )
#        yield lines[start:end]

#getting parameters
params = get_system_params(str(FOLDER))
cell_lo, cell_hi = params["cell"]
N_Na = params["N_Na"]
N_EMIM = params["N_EMIM"]
N_TFSI = params["N_TFSI"]
n_emim = params["n_emim"]
n_tfsi = params["n_tfsi"]

#calculating number of atoms in system
NT = N_Na + N_EMIM * n_emim + N_TFSI * n_tfsi

#arrays to store coodinates
XYZ_Na = np.zeros((N_Na, 3, T)) #Na
XYZ_TFSI_O = np.zeros((N_TFSI, 4, 3, T)) #O of TFSI - 4 of them
XYZ_EMIM_N = np.zeros((N_EMIM, 2, 3, T)) #N of EMIM - 2 of them (not explicitly used here)

#arrays for storing index iof each ion, to ensure the correct assignments are made
Na_index = np.zeros(N_Na, dtype=int)
TFSI_index = np.zeros(N_TFSI, dtype=int)
EMIM_index = np.zeros(N_EMIM, dtype=int)

#arrays for keeping track of the index for each atom in a molecule
TFSI_COUNT = np.zeros((N_TFSI, T), dtype=int)
EMIM_COUNT = np.zeros((N_EMIM, T), dtype=int)

#reading lines of dump file
with DUMP_FILE.open("r") as f:
    lines = f.readlines()

#counters 
tot_na = 0
tot_tfsi = 0
tot_emim = 0
# Read atom IDs from the first frame
for line in lines[9:9 + NT]:
    parts = line.split()
    atom_id = int(parts[1])#might need to change, depending on your file
    atom_type = int(parts[2])#might need to change, depending on your file

    if atom_type == 6:#this will extract Na index - change based on the LAMMPS input
        Na_index[tot_na] = atom_id
        tot_na += 1
    elif atom_type == 4:#this will extract O of TFSO index - change based on the LAMMPS input
        TFSI_index[tot_tfsi] = atom_id
        tot_tfsi += 1
    elif atom_type == 15:#this will extract N of EMIM index - change based on the LAMMPS input
        EMIM_index[tot_emim] = atom_id
        tot_emim += 1

na_map = {atom_id: i for i, atom_id in enumerate(Na_index)}
tfsi_map = {atom_id: i for i, atom_id in enumerate(TFSI_index)}
emim_map = {atom_id: i for i, atom_id in enumerate(EMIM_index)}

for step in range(T):
    print(step)

    start = 9 + step * (NT + 9)
    frame_lines = lines[start:start + NT]

    for line in frame_lines:
        parts = line.split()
        atom_id = int(parts[1])#might need to change, depending on your file
        atom_type = int(parts[2])#might need to change, depending on your file

        x = wrap_pbc(float(parts[5]), cell_lo, cell_hi)
        y = wrap_pbc(float(parts[6]), cell_lo, cell_hi)
        z = wrap_pbc(float(parts[7]), cell_lo, cell_hi)

        if atom_type == 6:#might need to change, depending on your file
            idx = na_map[atom_id]
            XYZ_Na[idx, :, step] = (x, y, z)

        elif atom_type == 4:#might need to change, depending on your file
            idx = tfsi_map[atom_id]
            pos = TFSI_COUNT[idx, step]
            if pos >= XYZ_TFSI_O.shape[1]:
                raise IndexError(
                    f"TFSI atom count overflow for molecule {idx} at step {step}"
                )
            XYZ_TFSI_O[idx, pos, :, step] = (x, y, z)
            TFSI_COUNT[idx, step] += 1

        elif atom_type == 15:#might need to change, depending on your file
            idx = emim_map[atom_id]
            pos = EMIM_COUNT[idx, step]
            if pos >= XYZ_EMIM_N.shape[1]:
                raise IndexError(
                    f"EMIM atom count overflow for molecule {idx} at step {step}"
                )
            XYZ_EMIM_N[idx, pos, :, step] = (x, y, z)
            EMIM_COUNT[idx, step] += 1

#saving matrices to files
np.save(FOLDER / "XYZ_Na_PBC.npy", XYZ_Na)
np.save(FOLDER / "XYZ_TFSI_O_PBC.npy", XYZ_TFSI_O)
np.save(FOLDER / "XYZ_EMIM_N_PBC.npy", XYZ_EMIM_N)