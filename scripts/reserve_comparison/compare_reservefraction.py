import logging
import os
from time import perf_counter

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from gototile.grid import SkyGrid

from goto_sim.scheduling import HEATSurvey
from goto_sim.sim import AltAzCache, GOTONode, Simulator

logging.basicConfig(level=logging.INFO)
# Filter out astropy warnings for cleaner output
logging.getLogger("astropy").setLevel(logging.ERROR)


def compare_reserved_fraction(
    simulator: Simulator,
    reserved_fractions: list[float],
    results_dir=None,
    results_prefix="simulation_results_reserve_",
) -> pd.DataFrame:
    """
    Compare different reserve fractions for a list of surveys.

    :param simulator: Simulator object to use for the comparison.
    :param reserved_fractions: List of reserve fractions to test (between 0 and 1).
    :param results_dir: Directory to save the results CSV files. If None, files are not saved.
    :param results_prefix: Prefix for the results files.
    :return: DataFrame with cadence results for each survey and reserve fraction.
    """
    results = []
    for reserve_fraction in reserved_fractions:
        simulator.reserved_fraction = reserve_fraction
        simulator.run()
        if results_dir is not None:
            results_file = os.path.join(
                results_dir, f"{results_prefix}{reserve_fraction:.2f}.csv"
            )
            simulator.save_results(results_file)
        obs_df = pd.DataFrame(simulator.results)
        results.append(obs_df)
    combined_results = pd.concat(
        results, keys=reserved_fractions, names=["reserve_fraction", "index"]
    )
    return combined_results


if __name__ == "__main__":
    simulate_allsky = True
    reserve_times = np.arange(0, 0.95, 0.05)
    cadences_hot = []
    cadences_cold = []
    cadences_all = []
    start_time = Time("2026-01-01T00:00:00")
    stop_time = Time("2027-01-01T00:00:00")
    time_step = 5 * u.min  # minutes
    alt_lim = 30
    cache_file = "../2026_altaz_cache.npz"
    nodes = [
        GOTONode(site="GOTO-N", altlim=alt_lim),
        GOTONode(site="GOTO-S", altlim=alt_lim),
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
    base_heat_survey = HEATSurvey()  # Adjust revisit time here if necessary
    print(f"Base HEAT survey has {len(base_heat_survey.tiles)} tiles.")
    base_cold_survey = base_heat_survey.generate_colds()

    grid: SkyGrid = SkyGrid.from_name("GOTO")
    simulator = Simulator(
        start_time=start_time,
        stop_time=stop_time,
        surveys=[base_heat_survey, base_cold_survey],
        too_fraction=0.06,  # 6% TOO fraction (derived from GRB observations in 2025)
    )
    cache.load_data(cache_file)
    simulator.cache = cache
    results_dir = "sims"
    combined_results = compare_reserved_fraction(
        simulator, reserve_times, results_dir=results_dir
    )
    # Repeat simulations but using an all-sky survey
    if simulate_allsky:
        simulator = Simulator(
            start_time=start_time,
            stop_time=stop_time,
            too_fraction=0.06,
        )
        simulator.cache = cache
        combined_allsky_results = compare_reserved_fraction(
            simulator,
            reserve_times,
            results_dir=results_dir,
            results_prefix="simulation_results_reserve_allsky_",
        )
    end_time = perf_counter()
    print(f"Completed in {end_time - start:.2f} seconds.")
