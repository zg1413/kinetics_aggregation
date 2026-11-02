#from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List
import numpy as np


SYSTEMS: Dict[str, dict] = {
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
    """function for importing parameters"""
    key = folder.rstrip("/").split("/")[-1]
    if key not in SYSTEMS:
        raise ValueError(f"Unknown folder preset: {folder!r}")
    return SYSTEMS[key]

#def wrap_value(value: float, lo: float, hi: float) -> float:
#    """function for considering perioidic images"""
#    box = hi - lo
#    return ((value - lo) % box) + lo#won't this only consider things in one direction though?


#def wrap_xyz(xyz: np.ndarray, lo: float, hi: float) -> np.ndarray:
#    """ """
#    return np.array([wrap_value(xyz[0], lo, hi), wrap_value(xyz[1], lo, hi), wrap_value(xyz[2], lo, hi)])

def unique_hits(indices: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out

def periodic_shifts(xyz: np.ndarray, lo: float, hi: float, cutoff: float) -> List[np.ndarray]:
    """Return vectors for periodic boundary conditions"""
    shifts = [np.zeros(3)]
    box = hi - lo
    low_edge = lo + cutoff + 1e-5
    high_edge = hi - cutoff - 1e-5

    x, y, z = xyz
    if x < low_edge:
        shifts.append(np.array([-box, 0.0, 0.0]))
    if x > high_edge:
        shifts.append(np.array([box, 0.0, 0.0]))
    if y < low_edge:
        shifts.append(np.array([0.0, -box, 0.0]))
    if y > high_edge:
        shifts.append(np.array([0.0, box, 0.0]))
    if z < low_edge:
        shifts.append(np.array([0.0, 0.0, -box]))
    if z > high_edge:
        shifts.append(np.array([0.0, 0.0, box]))

    # Preserve the original combination rules.
    if len(shifts) == 3:
        shifts.append(shifts[1] + shifts[2])
    if len(shifts) == 4:
        s1, s2, s3 = shifts[1], shifts[2], shifts[3]
        shifts.extend([s1 + s2, s2 + s3, s1 + s3, s1 + s2 + s3])

    return shifts

def load_xyz(folder: str, filename: str) -> np.ndarray:
    """function for loading xyz np binary file """
    return np.load(Path(folder) / filename)
