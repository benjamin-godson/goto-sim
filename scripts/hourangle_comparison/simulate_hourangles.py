import os

import pandas as pd
from astropy.coordinates import Angle

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

if __name__ == "__main__":
    ha_lims = [1, 2, 3, 4, 5, 6]
    heats_hour_limits = [Angle(h, unit=u.hourangle) for h in ha_lims]
    reserve_time = 0.1
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
    base_heat_survey = HEATSurvey()
    print(f"Base HEAT survey has {len(base_heat_survey.tiles)} tiles.")
    base_cold_survey = base_heat_survey.generate_colds()

    # Create extra HEAT surveys with different declination ranges
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
    for hour_angle in heats_hour_limits:
        heat_survey = HEATSurvey(ha_limit=hour_angle)
        cold_survey = heat_survey.generate_colds()
        simulator = Simulator(
            start_time=start_time,
            stop_time=stop_time,
            surveys=[heat_survey, cold_survey],
            too_fraction=0.06,
            reserved_fraction=reserve_time,
        )
        simulator.cache = cache
        print(
            f"Running simulation with HEAT hour angle limit {hour_angle.to(u.hourangle)}..."
        )
        simulator.run()
        print("Simulation complete.")
        print("Saving results...")
        results_file = os.path.join(
            results_dir,
            f"simulation_results_ha_limit_{hour_angle.to(u.hourangle).value:.1f}h.csv",
        )
        simulator.save_results(results_file)
        obs_df = pd.DataFrame(simulator.results)
        cadences_hot.append(
            obs_df[obs_df["tile"].isin(heat_survey.tiles)]
            .groupby("tile")["time"]
            .apply(lambda x: x.diff().mean())
        )
        cadences_cold.append(
            obs_df[obs_df["tile"].isin(cold_survey.tiles)]
            .groupby("tile")["time"]
            .apply(lambda x: x.diff().mean())
        )
        cadences_all.append(
            obs_df.groupby("tile")["time"].apply(lambda x: x.diff().mean())
        )
        print(f"Results saved to {results_file}")
        end = perf_counter()
        print(f"Total simulation time: {end - start:.2f} seconds")
        print(f"Average cadence for HEAT tiles: {cadences_hot[-1].mean()}")
        print(f"Average cadence for COLD tiles: {cadences_cold[-1].mean()}")
        print(f"Average cadence for all tiles: {cadences_all[-1].mean()}")
