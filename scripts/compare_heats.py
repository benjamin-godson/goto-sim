"""
Compare different HEAT survey strategies by creating multiple HEAT surveys with different tile selections,
running simulations for each, and comparing the results. As of 25/11/2025 the reserved fraction only affects
the scheduling of non-high priority surveys, all HEAT surveys here are run with high priority so the reserved fraction
only affects the COLD surveys (and hence the array time impact is half what might be expected).
To reproduce the original behaviour, modify the HEATSurvey priority to 'normal' when creating the surveys."""

import os

import numpy as np
import pandas as pd

from goto_sim.sim import Simulator, AltAzCache, GOTONode
from goto_sim.scheduling import HEATSurvey
from goto_sim.utils import save_tilelist
from gototile.grid import SkyGrid
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
import logging
from time import perf_counter

logging.basicConfig(level=logging.INFO)
# Filter out astropy warnings for cleaner output
logging.getLogger("astropy").setLevel(logging.ERROR)

if __name__ == "__main__":
    run_simulation = True
    simulate_base = False
    simulate_extended = False
    simulate_asymmetric = False
    simulate_reduced = True

    # Create simulator
    start = perf_counter()

    # Create HEAT survey
    base_heat_survey = HEATSurvey()
    print(f"Base HEAT survey has {len(base_heat_survey.tiles)} tiles.")
    base_cold_survey = base_heat_survey.generate_colds()

    # Create extra HEAT surveys with different declination ranges
    grid: SkyGrid = SkyGrid.from_name("GOTO")
    tile_coords: SkyCoord = grid.coords
    base_tiles = set(base_heat_survey.tiles)
    grid.plot(
        highlight=base_heat_survey.tiles,
        title="Base HEAT Survey Tiles",
        color="lightgrey",
    )
    base_tile_coords = grid.get_coordinates(base_tiles)
    # Find the galactic latitude of the base tiles
    base_galactic_lats = base_tile_coords.galactic.b.degree
    min_base_lat = 10
    max_base_lat = np.abs(base_galactic_lats).max()
    print(
        "Base HEAT survey covers galactic latitudes from "
        f"{min_base_lat:.2f} to {max_base_lat:.2f} degrees."
    )

    # Add tiles with declinations up to 30 degrees declination and galactic latitudes
    # greater than min_base_lat
    dec_lim = 30
    extended_heats_tiles = []
    for tile, coord in zip(grid.tilenames, tile_coords):
        gal_lat = coord.galactic.b.degree
        if (
            tile not in base_tiles
            and np.abs(gal_lat) >= min_base_lat
            and abs(coord.dec.degree) <= dec_lim
        ):
            extended_heats_tiles.append(tile)
    print(f"Adding {len(extended_heats_tiles)} extra HEAT tiles for extended survey.")
    extended_heat_survey = HEATSurvey(tels=[1, 3])
    extended_heat_survey.add_tiles(extended_heats_tiles)
    print(f"Extended HEAT survey now has {len(extended_heat_survey.tiles)} tiles.")
    file_name = f"extended_heat_tiles_dec{dec_lim}_lat{min_base_lat}.txt"
    print(f"Saving extended HEAT tile list to {file_name}...")
    save_tilelist(extended_heat_survey.tiles, file_name)
    extended_cold_survey = extended_heat_survey.generate_colds()
    grid.plot(
        highlight=extended_heat_survey.tiles,
        title="Extended HEAT Survey Tiles",
        color="lightgrey",
    )
    # plt.show()

    # Create an asymmetric HEATS survey, adding more tiles to the south but not the north
    asymmetric_heats_tiles = []
    for tile, coord in zip(grid.tilenames, tile_coords):
        gal_lat = coord.galactic.b.degree
        if (
            tile not in base_tiles
            and np.abs(gal_lat) >= min_base_lat
            and abs(coord.dec.degree) <= dec_lim
            and coord.dec.degree < 0
        ):
            asymmetric_heats_tiles.append(tile)
    print(
        f"Adding {len(asymmetric_heats_tiles)} extra HEAT tiles for asymmetric survey."
    )
    asymmetric_heat_survey = HEATSurvey(tels=[1, 3])
    asymmetric_heat_survey.add_tiles(asymmetric_heats_tiles)
    print(f"Asymmetric HEAT survey now has {len(asymmetric_heat_survey.tiles)} tiles.")
    file_name_asym = f"asymmetric_heat_tiles_dec{dec_lim}_lat{min_base_lat}.txt"
    print(f"Saving asymmetric HEAT tile list to {file_name_asym}...")
    save_tilelist(asymmetric_heat_survey.tiles, file_name_asym)
    asymmetric_cold_survey = asymmetric_heat_survey.generate_colds()
    grid.plot(
        highlight=asymmetric_heat_survey.tiles,
        title="Asymmetric HEAT Survey Tiles",
        color="lightgrey",
    )
    # plt.show()

    # Create reduced HEATS survey, removing tiles at high declinations
    reduced_heats_tiles = []
    reduced_dec_lim = 15
    for tile, coord in zip(grid.tilenames, tile_coords):
        gal_lat = coord.galactic.b.degree
        if tile in base_tiles and (abs(coord.dec.degree) <= reduced_dec_lim):
            reduced_heats_tiles.append(tile)
    print(
        f"Removing {len(base_heat_survey.tiles) - len(reduced_heats_tiles)} HEAT tiles for reduced survey."
    )
    reduced_heat_survey = HEATSurvey(tels=[1, 3], tiles=reduced_heats_tiles)
    print(f"Reduced HEAT survey now has {len(reduced_heat_survey.tiles)} tiles.")
    file_name_reduced = f"reduced_heat_tiles_dec{dec_lim}_lat{min_base_lat}.txt"
    print(f"Saving reduced HEAT tile list to {file_name_reduced}...")
    save_tilelist(reduced_heat_survey.tiles, file_name_reduced)
    reduced_cold_survey = reduced_heat_survey.generate_colds()
    grid.plot(
        highlight=reduced_heat_survey.tiles,
        title="Reduced HEAT Survey Tiles",
        color="lightgrey",
    )

    start_time = Time("2026-01-01T00:00:00")
    stop_time = Time("2027-01-01T00:00:00")
    time_step = 5 * u.min  # minutes
    altlim = 30
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

    if run_simulation:
        cache_file = "2026_altaz_cache.npz"
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
        print("Running base HEAT simulations...")
        if simulate_base:
            for reserved_fraction in [0.1, 0.25, 0.5]:
                print(
                    f"Running simulation with reserved fraction: {reserved_fraction}..."
                )
                sim = Simulator(
                    start_time=start_time,
                    stop_time=stop_time,
                    reserved_fraction=reserved_fraction,
                )
                sim.cache = cache
                sim.add_survey(base_heat_survey)
                sim.add_survey(base_cold_survey)
                sim.run()
                sim.save_results(
                    f"2026_base_heat_simulation_reserve{reserved_fraction}_results.csv"
                )
                end = perf_counter()
                print(f"Total simulation time: {end - start:.2f} seconds")
                # Print summary statistics
                obs_df = pd.DataFrame(sim.results)
                total_observations = len(obs_df)
                observations_per_tile = obs_df["tile"].value_counts().mean()
                print(f"Total observations (base HEAT): {total_observations}")
                print(
                    f"Average observations per tile (base HEAT): {observations_per_tile:.2f}"
                )
                average_cadence = obs_df.groupby("tile")["time"].apply(
                    lambda x: x.diff().mean()
                )
                print(f"Average cadence (base HEAT): {average_cadence.mean()}")

        if simulate_extended:
            for reserved_fraction in [0.1, 0.25, 0.5]:
                # Now run extended survey
                sim_extended = Simulator(
                    start_time=start_time,
                    stop_time=stop_time,
                    reserved_fraction=reserved_fraction,
                )
                sim_extended.cache = cache
                sim_extended.add_survey(extended_heat_survey)
                sim_extended.add_survey(extended_cold_survey)
                sim_extended.run()
                print("Saving extended HEAT simulation results...")
                sim_extended.save_results(
                    f"2026_extended_heat_simulation_reserve{reserved_fraction}_results.csv"
                )
                # Print summary statistics
                obs_extended_df = pd.DataFrame(sim_extended.results)
                total_observations_extended = len(obs_extended_df)
                observations_per_tile_extended = (
                    obs_extended_df["tile"].value_counts().mean()
                )
                print(
                    f"Total observations (extended HEAT): {total_observations_extended}"
                )
                print(
                    f"Average observations per tile (extended HEAT): {observations_per_tile_extended:.2f}"
                )
                average_cadence_extended = obs_extended_df.groupby("tile")[
                    "time"
                ].apply(lambda x: x.diff().mean())
                print(
                    f"Average cadence (extended HEAT): {average_cadence_extended.mean()}"
                )
        if simulate_asymmetric:
            for reserved_fraction in [0.1, 0.25, 0.5]:
                # Now run asymmetric survey
                sim_asymmetric = Simulator(
                    start_time=start_time,
                    stop_time=stop_time,
                    reserved_fraction=reserved_fraction,
                )
                sim_asymmetric.cache = cache
                sim_asymmetric.add_survey(asymmetric_heat_survey)
                sim_asymmetric.add_survey(asymmetric_cold_survey)
                sim_asymmetric.run()
                print("Saving asymmetric HEAT simulation results...")
                sim_asymmetric.save_results(
                    f"2026_asymmetric_heat_simulation_reserve{reserved_fraction}_results.csv"
                )
                # Print summary statistics
                obs_asymmetric_df = pd.DataFrame(sim_asymmetric.results)
                total_observations_asymmetric = len(obs_asymmetric_df)
                observations_per_tile_asymmetric = (
                    obs_asymmetric_df["tile"].value_counts().mean()
                )
                print(
                    f"Total observations (asymmetric HEAT): {total_observations_asymmetric}"
                )
                print(
                    f"Average observations per tile (asymmetric HEAT): {observations_per_tile_asymmetric:.2f}"
                )
                average_cadence_asymmetric = obs_asymmetric_df.groupby("tile")[
                    "time"
                ].apply(lambda x: x.diff().mean())
                print(
                    f"Average cadence (asymmetric HEAT): {average_cadence_asymmetric.mean()}"
                )

        if simulate_reduced:
            for reserved_fraction in [0.1, 0.25, 0.5]:
                # Now run reduced survey
                sim_reduced = Simulator(
                    start_time=start_time,
                    stop_time=stop_time,
                    reserved_fraction=reserved_fraction,
                )
                sim_reduced.cache = cache
                sim_reduced.add_survey(reduced_heat_survey)
                sim_reduced.add_survey(reduced_cold_survey)
                sim_reduced.run()
                print("Saving reduced HEAT simulation results...")
                sim_reduced.save_results(
                    f"2026_reduced_heat_simulation_reserve{reserved_fraction}_results.csv"
                )
                # Print summary statistics
                obs_reduced_df = pd.DataFrame(sim_reduced.results)
                total_observations_reduced = len(obs_reduced_df)
                observations_per_tile_reduced = (
                    obs_reduced_df["tile"].value_counts().mean()
                )
                print(
                    f"Total observations (reduced HEAT): {total_observations_reduced}"
                )
                print(
                    f"Average observations per tile (reduced HEAT): {observations_per_tile_reduced:.2f}"
                )
                average_cadence_reduced = obs_reduced_df.groupby("tile")["time"].apply(
                    lambda x: x.diff().mean()
                )
                print(
                    f"Average cadence (reduced HEAT): {average_cadence_reduced.mean()}"
                )
    end = perf_counter()
    print(f"Total script time: {end - start:.2f} seconds")
