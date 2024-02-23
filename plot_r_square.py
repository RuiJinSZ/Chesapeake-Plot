"""
Plot for r^2 analysis

Concretely, scatter plot simulation-observation, and plot straight line y = x

In principle, simulation-observation points should all fall on to line y = x,
in practise, there may be deviations due to observation fluctutation or simulation inperfection,
so we scatter plot and compute r^2 of y = x fitting to measure the quality of simulation
"""

import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import lib.observation
import lib.simulation
import lib.utils


STATION_TO_LATITUDE_LONGITUDE_INDICES = {"CB3.3C": (100, 38)}


def parse_args() -> argparse.Namespace:
    DEFAULT_YEAR = 2017
    DEFAULT_STATION = "CB3.3C"
    DEFAULT_OBSERVATION_CSV = "WaterQualityWaterQualityStation.csv"
    DEFAULT_SIMULATION_NCS = [
        f"ocean_avg_{n}.nc" for n in [
            "0001", "0002", "0003", "0004", "0005",
            "0006", "0007", "0008", "0009", "0010",
            "0011", "0012", "0013"]
    ]

    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        "--observed_species",
        type=str,
        required=True,
        help="Species to compare observation vs simulation",
    )
    parser.add_argument(
        "--simulated_species",
        type=str,
        required=True,
        help="Species to compare observation vs simulation",
    )
    parser.add_argument(
        "--molecular_weight",
        type=float,
        required=True,
        help="Molecular weight of the species",
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Year to compare observation vs simulation, default = {DEFAULT_YEAR}"
    )
    parser.add_argument(
        "--station",
        type=str,
        default = DEFAULT_STATION,
        help=f"Station to compare observation vs simulation, default = {DEFAULT_STATION}",
    )
    parser.add_argument(
        "--observation_csv",
        type=str,
        default=DEFAULT_OBSERVATION_CSV,
        help=f"CSV file to read observation data from, default = {DEFAULT_OBSERVATION_CSV}",
    )
    parser.add_argument(
        "--simulation_ncs",
        type=str,
        nargs='+',
        default=DEFAULT_SIMULATION_NCS,
        help=f"NC files to read simulation data from, default = {DEFAULT_SIMULATION_NCS}"
    )

    args = parser.parse_args()
    return args


def align_value_on_observed_and_simulated_grids(
    simulated_value: np.ndarray,
    station: str,
    observed_time: np.ndarray,
    observed_depth: np.ndarray,
) -> np.ndarray:
    """
    Observation and simulation use different grids

    Since simulation grids are much denser,
    we interpolate from simulation grids to observation grids
    """

    # For simulation grids
    #     time is in 1-day grid, 1 year long
    #     depth is in 1-meter grid, 20 meters total
    simulated_time = np.arange(1, 365)
    simulated_depth = np.arange(1, 21)

    latitude_index, longitude_index = STATION_TO_LATITUDE_LONGITUDE_INDICES[station]
    simulated_value = simulated_value[0:364, :, latitude_index, longitude_index].transpose()
    simulated_value = np.flip(simulated_value, axis=0)

    # Linearly interpolate from simulation grids to observed grids
    aligned_value = []
    for i in range(observed_depth.shape[0]):
        shallower = simulated_depth.shape[0] - 1
        deeper    = 0
        for j in range(simulated_depth.shape[0]):
            distance = simulated_depth[j] - observed_depth[i]
            # is shallower
            if distance < 0:
                if abs(distance) < abs(simulated_depth[shallower] - observed_depth[i]): shallower = j
            # is deeper
            else:
                if abs(distance) < abs(simulated_depth[deeper   ] - observed_depth[i]): deeper    = j
        for j in range(simulated_time.shape[0]):
            if simulated_time[j] == observed_time[i]:
                slope = (simulated_value[deeper, j] - simulated_value[shallower, j]) / (simulated_depth[deeper] - simulated_depth[shallower])
                data = slope * (observed_depth[i] - simulated_depth[shallower]) + simulated_value[shallower, j]
                aligned_value.append(data)

    return np.array(aligned_value)


if __name__ == "__main__":
    """ initialize """
    args = parse_args()

    observed_species: str = args.observed_species
    simulated_species: str = args.simulated_species
    molecular_weight: float = args.molecular_weight

    year: int = args.year
    station: str = args.station
    observation_csv: str = args.observation_csv
    simulation_ncs: List[str] = args.simulation_ncs

    """ parse observation """
    observed_time, observed_depth, observed_value = lib.observation.parse_observation(
        observation_csv,
        year,
        station,
        observed_species,
        molecular_weight,
    )

    """ parse simulation """
    simulated_value = lib.simulation.parse_simulation(simulation_ncs, simulated_species)

    """ align observation and simulation grids """
    aligned_value = align_value_on_observed_and_simulated_grids(
        simulated_value,
        station,
        observed_time,
        observed_depth,
    )
    assert(len(observed_value) == len(aligned_value))

    """ compute and output RMSD """
    r_square = lib.utils.compute_r_square(observed_value, aligned_value)
    print(f"Between observed {observed_species} and simulated {simulated_species}, r^2 = {r_square}")

    """ plot """
    plt.scatter(observed_value, aligned_value)

    lower = min(min(observed_value), min(aligned_value))
    upper = max(max(observed_value), max(aligned_value))
    dashed_x = np.linspace(lower, upper, 100)
    dashed_y = dashed_x.copy()
    plt.plot(dashed_x, dashed_y)

    plt.show()
