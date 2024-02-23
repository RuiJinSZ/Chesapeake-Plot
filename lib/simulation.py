"""
Parse simulation data
"""

from typing import List

import numpy as np
import netCDF4


def read_ncs(ncs: List[str], species: str) -> List[np.ndarray]:
    tensors = []
    for nc in ncs:
        file = netCDF4.Dataset(nc, 'a', format="NETCDF4")
        nc_variable = file.variables[species]
        np_tensor = np.array(nc_variable)
        tensors.append(np_tensor)
    return tensors


def parse_simulation(
    ncs: List[str],
    species: str,
    missing_threshold: float = 1e36,
    replacement: float = 0.0,
) -> np.ndarray:
    tensors = read_ncs(ncs, species)
    # Concatenate all tensors along time dimension (0)
    tensor = np.concatenate(tensors)
    # Replace missing values
    tensor[np.where(np.abs(tensor) > missing_threshold)] = replacement
    return tensor
