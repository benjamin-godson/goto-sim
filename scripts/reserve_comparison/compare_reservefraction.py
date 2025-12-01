import os

import numpy as np
import pandas as pd

from goto_sim.sim import Simulator, AltAzCache, GOTONode
from goto_sim.scheduling import HEATSurvey
from gototile.grid import SkyGrid
from astropy.time import Time
import astropy.units as u
import logging
from time import perf_counter


logging.basicConfig(level=logging.INFO)
# Filter out astropy warnings for cleaner output
logging.getLogger("astropy").setLevel(logging.ERROR)


def compare_reserved_fraction(
    simulator: Simulator, reserved_fractions: list[float], results_dir=None
) -> pd.DataFrame:
    """
    Compare different reserve fractions for a list of surveys.

    :param simulator: Simulator object to use for the comparison.
    :param reserve_fractions: List of reserve fractions to test (between 0 and 1).
    :param results_dir: Directory to save the results CSV files. If None, files are not saved.
    :return: DataFrame with cadence results for each survey and reserve fraction.
    """
    results = []
    for reserve_fraction in reserved_fractions:
        simulator.reserved_fraction = reserve_fraction
        simulator.run()
        if results_dir is not None:
            results_file = os.path.join(
                results_dir, f"simulation_results_reserve_{reserve_fraction:.2f}.csv"
            )
            simulator.save_results(results_file)
        obs_df = pd.DataFrame(simulator.results)
        results.append(obs_df)
    combined_results = pd.concat(
        results, keys=reserved_fractions, names=["reserve_fraction", "index"]
    )
    return combined_results


if __name__ == "__main__":
    reserve_times = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cadences_hot = []
    cadences_cold = []
    cadences_all = []
    start_time = Time("2026-01-01T00:00:00")
    stop_time = Time("2027-01-01T00:00:00")
    time_step = 5 * u.min  # minutes
    altlim = 30
    cache_file = "../2026_altaz_cache.npz"
    nodes = [
        GOTONode(site="GOTO-N", altlim=altlim),
        GOTONode(site="GOTO-S", altlim=altlim),
    ]
    cache = AltAzCache(
        start_time=start_time,
        stop_time=stop_time,
        nodes=nodes,
        time_step=time_step,
    )
    # Create simulator
    start = perf_counter()

    # Create HEAT survey
    base_heat_survey = HEATSurvey()
    print(f"Base HEAT survey has {len(base_heat_survey.tiles)} tiles.")
    base_cold_survey = base_heat_survey.generate_colds()

    # Create extra HEAT surveys with different declination ranges
    grid: SkyGrid = SkyGrid.from_name("GOTO")
    simulator = Simulator(
        start_time=start_time,
        stop_time=stop_time,
        surveys=[base_heat_survey, base_cold_survey],
    )
    cache.load_data(cache_file)
    simulator.cache = cache
    results_dir = "sims"
    combined_results = compare_reserved_fraction(
        simulator, reserve_times, results_dir=results_dir
    )
    for reserve_fraction, group in combined_results.groupby(level="reserve_fraction"):
        obs_df = group.droplevel("reserve_fraction")
        heat_tiles = set(base_heat_survey.tiles)
        cold_tiles = set(base_cold_survey.tiles)

        cadences_hot = []
        cadences_cold = []
        cadences_all = []

        for tile, tile_obs in obs_df.groupby("tile"):
            times = tile_obs["time"].sort_values()
            if len(times) < 2:
                continue
            time_diffs = times.diff().to_numpy()[1:]  # Skip the first NaT
            avg_cadence = (
                np.mean(time_diffs).astype("timedelta64[s]").item().total_seconds()
                / 3600.0
            )  # in hours
            cadences_all.append(avg_cadence)
            if tile in heat_tiles:
                cadences_hot.append(avg_cadence)
            elif tile in cold_tiles:
                cadences_cold.append(avg_cadence)

        mean_cadence_hot = np.mean(cadences_hot) if cadences_hot else float("nan")
        mean_cadence_cold = np.mean(cadences_cold) if cadences_cold else float("nan")
        mean_cadence_all = np.mean(cadences_all) if cadences_all else float("nan")

        print(
            f"Reserve Fraction: {reserve_fraction:.2f} | Mean Cadence HEAT: {mean_cadence_hot:.2f} hrs | Mean Cadence COLD: {mean_cadence_cold:.2f} hrs | Mean Cadence ALL: {mean_cadence_all:.2f} hrs"
        )
