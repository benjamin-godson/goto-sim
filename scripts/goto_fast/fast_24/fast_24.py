import pandas as pd

from goto_sim.sim import Simulator
from goto_sim.scheduling import Survey
from goto_sim.utils import load_tilelist
from astropy.time import Time
import astropy.units as u
import os
import logging
from time import perf_counter

logging.basicConfig(level=logging.INFO)
# Filter out astropy warnings for cleaner output
logging.getLogger("astropy").setLevel(logging.ERROR)

if __name__ == "__main__":
    # Create simulator
    start = perf_counter()
    sim = Simulator(
        start_time=Time("2024-02-04T12:00:00"),
        stop_time=Time("2024-02-14T12:00:00"),
        reserved_fraction=0,
        too_fraction=0,
    )
    cache = sim.cache
    cache_file = "fast_24_altaz_cache.npz"
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"Loading cache from {cache_file}...")
        start_load = perf_counter()
        cache.load_data(cache_file)
        end_load = perf_counter()
        print(f"Cache load time: {end_load - start_load:g} seconds")
    else:
        print(f"Creating new cache and saving to {cache_file}...")
        start_cache = perf_counter()
        cache.generate_cache()
        end_cache = perf_counter()
        print(f"Cache generation time: {end_cache - start_cache:.2f} seconds")
        start_write = perf_counter()
        cache.write_data(cache_file)
        end_write = perf_counter()
        print(f"Cache write time: {end_write - start_write:.2f} seconds")

    fast_tiles_n = load_tilelist("fast_24_n_tiles")
    fast_tiles_s = load_tilelist("fast_24_s_tiles")

    north = Survey(
        name="FAST_24_N",
        tiles=fast_tiles_n,  # Example tile indices
        revisit_time=4 * u.hour,
        tels=[2],
        priority="high",
    )
    south = Survey(
        name="FAST_24_S",
        tiles=fast_tiles_s,  # Example tile indices
        revisit_time=4 * u.hour,
        tels=[3, 4],
        priority="high",
    )

    sim.add_survey(north)
    sim.add_survey(south)

    print("Running simulation...")
    start_sim = perf_counter()
    sim.run()
    end_sim = perf_counter()
    print(f"Simulation run time: {end_sim - start_sim:.2f} seconds")
    print("Simulation complete.")
    print("Saving results...")
    sim.save_results("fast_24_simulation_results.csv")
    end = perf_counter()
    print(f"Total simulation time: {end - start:.2f} seconds")
    # Print summary statistics
    obs_df = pd.DataFrame(sim.results)
    total_observations = len(obs_df)
    observations_per_tile = obs_df["tile"].value_counts().mean()
    print(f"Total observations: {total_observations}")
    print(f"Average observations per tile: {observations_per_tile:.2f}")
    average_cadence = obs_df.groupby("tile")["time"].apply(lambda x: x.diff().mean())
    print(
        f"Average cadence (mean time between observations per tile): {average_cadence.mean()} days"
    )
