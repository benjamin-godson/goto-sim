"""
Core module for simulation classes and logic.
"""

from pathlib import Path
from typing import Union

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, HADec
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time
import astropy.units as u
from gototile.grid import SkyGrid

from .scheduling import Survey
from .utils import concat_earth_locations
import logging

logger = logging.getLogger(__name__)


class GOTONode:
    """
    A single GOTO node in the simulation. Contains information about the site as well
    as the sidereal time and tile hour angles for each time step of the simulation.
    """

    def __init__(
        self,
        site: str = None,
        location: EarthLocation = None,
        name: str = None,
        telescopes: int = 2,
        altlim: float = 30.0,
    ):
        """
        Initialize the GOTO node
        """
        if site is None and location is None:
            raise ValueError("Either site or location must be provided")
        if site is None and name is None:
            raise ValueError("Either site or name must be provided")

        if location:
            self.location = location
        else:
            try:
                self.location = self._resolve_site(site)
            except ValueError as e:
                raise e

        self.lat = self.location.lat
        self.lon = self.location.lon
        self.height = self.location.height

        self.name = name if name is not None else site
        self.telescopes = telescopes
        self.horizon = altlim  # degrees

    def __str__(self):
        return (
            f"GOTONode '{self.name}' at ({self.lat.deg:.2f}°, {self.lon.deg:.2f}°), "
            f"height {self.height.to(u.m).value:.1f} m, "
            f"{self.telescopes} telescopes, altlim={self.horizon}°"
        )

    def __repr__(self):
        return f"GOTONode(name={self.name}, location=({self.lat.deg:.2f}°, {self.lon.deg:.2f}°), height={self.height.to(u.m).value:.1f} m, telescopes={self.telescopes}, altlim={self.horizon}°)"

    @staticmethod
    def _resolve_site(site: str) -> EarthLocation:
        """
        Resolve the site name to a location
        :return:
        """
        sites = {
            "goto-north": "lapalma",
            "goto-n": "lapalma",
            "north": "lapalma",
            "n": "lapalma",
            "goto-south": "sso",
            "goto-s": "sso",
            "south": "sso",
        }
        location_name = sites.get(site.lower(), site)
        try:
            location = EarthLocation.of_site(location_name)
        except Exception as e:
            raise ValueError(
                f"Could not resolve site name '{site}': {e}, try goto-n or goto-s"
            )
        return location


class AltAzCache:
    """
    Stores the AltAz and Hour Angle coordinates of each GOTO tile for each time step of a
    simulation.
    The data is stored as a 4D numpy array of shape (n_times, n_nodes, n_tiles, 3),
    where the last dimension contains the altitude and azimuth in degrees.
    Attributes:
        nodes: List of GOTONode objects representing the GOTO nodes.
        times: Array of Time objects representing the time steps.
        n_times: Number of time steps.
        time_step: Time step between each time step as an astropy time Quantity.
        data: 4D numpy array of shape (n_times, n_nodes, n_tiles, 3) containing the
              altitude and azimuth in degrees and hour angle.
        grid: SkyGrid object representing the GOTO sky grid.
        dtype: Data type of the stored data (default: np.float16).

    """

    def __init__(
        self,
        nodes: list[GOTONode] = None,
        times: Time = None,
        start_time: Time = None,
        stop_time: Time = None,
        n_times: int = None,
        time_step: u.Quantity[u.s] = 5 * u.min,
        data: SkyCoord = None,
        grid: SkyGrid = SkyGrid.from_name("GOTO"),
        dtype: np.dtype = np.float16,
    ):
        if nodes is None:
            nodes = [GOTONode("goto-north"), GOTONode("goto-south")]
        self.nodes = nodes
        self.locations = [x.location for x in nodes]
        self.start_time = start_time
        if times is not None:
            self.times = times
            self.n_times = times.shape[0]
        else:
            if start_time is not None:
                if stop_time is not None:
                    self.stop_time = stop_time
                    self.n_times = int(
                        (stop_time - start_time).to_value(u.s)
                        // time_step.to_value(u.s)
                    )
                elif n_times is not None:
                    self.n_times = n_times
                    self.stop_time = self.start_time + time_step * n_times
                else:
                    raise ValueError(
                        "Either times or start_time and either stop_time or n_times must be provided"
                    )
            if time_step > 0:
                self.time_step = time_step
            else:
                raise ValueError("time_step must be > 0")
            self.times: Time = (
                self.start_time + np.arange(self.n_times) * self.time_step
            )

        self.data = data
        self.grid = grid
        self.dtype = dtype
        self.solar_alt: Union[np.array, None] = None

    def __str__(self):
        return (
            f"AltAzCache with {len(self.nodes)} nodes, "
            f"{self.n_times} time steps from {self.start_time.iso} to {self.stop_time.iso}, "
            f"grid: {self.grid.name}, data type: {self.dtype}"
        )

    def __repr__(self):
        return f"AltAzCache(nodes={self.nodes}, n_times={self.n_times}, start_time={self.start_time.iso}, stop_time={self.stop_time.iso}, time_step={self.time_step.to_value(u.s)} seconds, grid={self.grid.name}, dtype={self.dtype})"

    def get_data(self) -> SkyCoord:
        if self.data is None:
            raise ValueError(
                "Cache data has not been generated yet, call generate_cache() first"
            )
        return self.data

    def generate_cache(self):
        """
        Generate the AltAz cache for the given nodes and times.
        Results are stored in self.data
        :return:
        """

        logger.info(
            f"Generating AltAz cache for {self.n_times} time steps between {self.start_time.iso} and {self.stop_time.iso} for {len(self.nodes)} nodes"
        )

        if self.n_times < 20_000:
            locations = concat_earth_locations(self.locations)
            times = self.times
            coords = self.grid.coords
            frame = AltAz(
                location=locations[:, np.newaxis, np.newaxis],
                obstime=times[np.newaxis, np.newaxis, :],
            )
            ha_decframe = HADec(
                location=locations[:, np.newaxis, np.newaxis],
                obstime=times[np.newaxis, np.newaxis, :],
            )
            with erfa_astrom.set(ErfaAstromInterpolator(1 * u.day)):
                altaz: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(frame)
                hadec: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(
                    ha_decframe
                )
            # Saving the whole SkyCoord is a bit inefficient, especially for large n_times
            alts = altaz.alt.deg
            azs = altaz.az.deg
            has = hadec.ha.deg  # Hour angle in degrees
            # Combine both values into an array of size (n_times, n_nodes, n_tiles, 2)
            data = np.stack([alts, azs, has], axis=-1).transpose(2, 0, 1, 3)
        else:
            # For large n_times, we generate the data in chunks to save memory
            chunk_size = 10_000
            n_chunks = (self.n_times + chunk_size - 1) // chunk_size
            logger.info(f"Generating AltAz cache in {n_chunks} chunks to save memory")
            all_data = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, self.n_times)
                times_chunk = self.times[start_idx:end_idx]
                locations = concat_earth_locations([x.location for x in self.nodes])
                coords = self.grid.coords
                frame = AltAz(
                    location=locations[:, np.newaxis, np.newaxis],
                    obstime=times_chunk[np.newaxis, np.newaxis, :],
                )
                ha_decframe = HADec(
                    location=locations[:, np.newaxis, np.newaxis],
                    obstime=times_chunk[np.newaxis, np.newaxis, :],
                )
                with erfa_astrom.set(ErfaAstromInterpolator(1 * u.day)):
                    altaz: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(
                        frame
                    )
                    hadec: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(
                        ha_decframe
                    )
                alts = altaz.alt.deg
                azs = altaz.az.deg
                has = hadec.ha.deg  # Hour angle in degrees
                data_chunk = np.stack([alts, azs, has], axis=-1).transpose(2, 0, 1, 3)
                all_data.append(data_chunk)
            data = np.concatenate(all_data, axis=0)

        self.data = data.astype(self.dtype)
        # Generate solar altitudes for each time and node
        logger.info("Calculating Solar altitudes")
        sun = get_sun(self.times)
        frame = AltAz(
            obstime=self.times[:, np.newaxis], location=locations[np.newaxis :,]
        )
        solar_alt = sun[:, np.newaxis].transform_to(frame).alt.deg
        self.solar_alt = solar_alt
        self._validate()
        logger.info("Cache generation complete")

    def write_data(self, filename: Union[str, Path], overwrite: bool = True) -> None:
        """
        Write the cache to a numpy file, including the times and node locations
        :param filename: Path to the output file
        :param overwrite: Whether to overwrite an existing file
        :return:
        """
        times = self.times
        locations = [x.location for x in self.nodes]
        data = self.get_data()
        solar_alt = self.solar_alt

        # Combine metadata and data into a single npz file
        np.savez_compressed(
            filename,
            times=times.mjd,
            locations_lat=[loc.lat.deg for loc in locations],
            locations_lon=[loc.lon.deg for loc in locations],
            locations_height=[loc.height.to(u.m).value for loc in locations],
            data=data,
            solar_alt=solar_alt,
            overwrite=overwrite,
        )

    def load_data(self, filename: str) -> None:
        """
        Load the cache from a numpy file
        :param filename:
        :return:
        """
        npzfile = np.load(filename)
        mjd_times = npzfile["times"]
        self.times = Time(mjd_times, format="mjd")
        latitudes = npzfile["locations_lat"]
        longitudes = npzfile["locations_lon"]
        heights = npzfile["locations_height"]
        solar_alt = npzfile["solar_alt"]

        # Check that the locations and times match the existing ones
        if len(latitudes) != len(self.nodes):
            raise ValueError(
                "Number of locations in file does not match number of nodes"
            )
        for i, node in enumerate(self.nodes):
            if (
                not u.isclose(node.location.lat.deg, latitudes[i])
                or not u.isclose(node.location.lon.deg, longitudes[i])
                or not u.isclose(node.location.height.to(u.m).value, heights[i])
            ):
                raise ValueError(
                    f"Location of node {node.name} does not match location in file"
                )
        if len(self.times) != len(mjd_times):
            raise ValueError("Number of times in file does not match existing times")
        if not np.allclose(self.times.mjd, mjd_times):
            raise ValueError("Times in file do not match existing times")

        data = npzfile["data"]
        # Check that the data shape matches
        expected_shape = (self.n_times, len(self.nodes), self.grid.ntiles, 3)
        if data.shape != expected_shape:
            raise ValueError(
                f"Data shape in file {data.shape} does not match expected shape {expected_shape}"
            )
        self.solar_alt = solar_alt
        self.data = data.astype(self.dtype)
        self._validate()

    def _validate(self) -> bool:
        """
        Validate that the cache data has the correct shape, and expected properties
        :return:
        """
        data = self.get_data()
        expected_shape = (self.n_times, len(self.nodes), self.grid.ntiles, 3)
        if data.shape != expected_shape:
            raise ValueError(
                f"Cache data shape {data.shape} does not match expected shape {expected_shape}"
            )
        # Check that solar altaz has the expected shape
        if self.solar_alt is not None:
            expected_sun_shape = (self.n_times, len(self.nodes))
            if self.solar_alt.shape != expected_sun_shape:
                raise ValueError(
                    f"Solar altaz shape {self.solar_alt.shape} does not match expected shape {expected_sun_shape}"
                )

        # Check that the last time matches the expected stop time within one time step
        last_time = self.times[-1]
        if not np.isclose(
            (last_time - self.stop_time).to_value(u.s),
            0,
            atol=1.1 * self.time_step.to_value(u.s),
        ):
            raise ValueError(
                f"Last time {last_time.iso} is more than one timestep from expected stop time {self.stop_time.iso}"
            )
        return True

    def _estimate_size(self):
        n_nodes = len(self.nodes)
        n_tiles = self.grid.ntiles
        n_times = self.n_times
        size_bytes = n_nodes * n_tiles * n_times * 2 * np.dtype(self.dtype).itemsize
        return size_bytes

    @property
    def alt(self) -> np.ndarray:
        """
        Get the altitudes from the data cache
        :return:
        """
        data = self.get_data()
        return data[..., 0]

    @property
    def az(self) -> np.ndarray:
        """
        Get the azimuths from the data cache
        :return:
        """
        data = self.get_data()
        return data[..., 1]

    @property
    def ha(self) -> np.ndarray:
        """
        Get the hour angles from the data cache
        :return:
        """
        data = self.get_data()
        return data[..., 2]


class Simulator:
    """
    A single simulation of an observing run for GOTO.
    Attributes:
        nodes: List of GOTONode objects representing the GOTO nodes.
        start_time: Start time of the simulation as an astropy Time object.
        stop_time: End time of the simulation as an astropy Time object.
        time_step: Time step between each time step as an astropy time Quantity.
        cache: AltAzCache object containing the precomputed AltAz data for the simulation.
        twilight_limit: Maximum sun altitude (degrees) for observations to be allowed.
        altlim: Minimum altitude (degrees) for observations to be allowed.
        reserved_fraction: Fraction of time reserved for non-survey observations.
        results: List of scheduled observations generated by the simulation.
    """

    def __init__(
        self,
        nodes: list[GOTONode] = None,
        start_time: Time = None,
        stop_time: Time = None,
        time_step: u.Quantity[u.s] = 5 * u.min,
        surveys: list[Survey] = None,
        cache: AltAzCache = None,
        twilight_limit: float = -12.0,
        altlim: float = 30.0,
        reserved_fraction=0.1,
        too_fraction=0,
    ):
        """
        Initialize the simulator
        """
        if nodes is None:
            nodes = [
                GOTONode(site="GOTO-N", altlim=altlim),
                GOTONode(site="GOTO-S", altlim=altlim),
            ]
        self.nodes = nodes
        if start_time is None:
            logger.debug("Start time not specified, using current time")
            start_time = Time.now()
        if stop_time is None:
            logger.debug("End time not specified, using start_time + 24 hours")
            stop_time = start_time + 24 * u.hour
        self.start_time = start_time
        self.stop_time = stop_time
        self.time_step = time_step
        self.surveys = surveys
        if cache is None:
            cache = AltAzCache(
                start_time=start_time,
                stop_time=stop_time,
                nodes=nodes,
                time_step=time_step,
            )
        self.cache = cache
        self.locations = [node.location for node in cache.nodes]
        self.times: Time = self.cache.times
        self.twilight_limit = twilight_limit  # degrees
        self.reserved_fraction = reserved_fraction
        self.too_fraction = too_fraction
        self.results = []

    def __str__(self):
        return (
            f"Simulator with {len(self.nodes)} nodes from {self.start_time.iso} to "
            f"{self.stop_time.iso} with time step {self.time_step.to_value(u.s)} seconds "
            f"and surveys: "
            f"{', '.join([survey.name for survey in self.surveys]) if self.surveys else 'None'}"
        )

    def __repr__(self):
        return f"Simulator(nodes={self.nodes}, start_time={self.start_time.iso}, stop_time={self.stop_time.iso}, time_step={self.time_step.to_value(u.s)} seconds, surveys={self.surveys}, twilight_limit={self.twilight_limit} degrees, reserved_fraction={self.reserved_fraction})"

    def load_cached_data(self, filename: str):
        """
        Load cached AltAz data from a file
        :param filename:
        :return:
        """
        self.cache.load_data(filename)

    def add_survey(self, survey: Survey):
        """
        Add a survey to the simulator
        :param survey:
        """
        if self.surveys is None:
            self.surveys = []
        self.surveys.append(survey)

    def run(self):
        """
        Run the simulation, returning a list of scheduled observations
        :return:
        """
        self._setup()
        solar_alts = self.cache.solar_alt

        tels_per_node = np.fromiter([node.telescopes for node in self.nodes], dtype=int)
        # What index do we need each node to start from in the telescope list
        tel_start_indices = np.cumsum(np.insert(tels_per_node, 0, 0))[:-1]
        total_tels = tels_per_node.sum()

        logger.info(
            f"Running simulation with {total_tels} telescopes & {len(self.nodes)} nodes"
        )

        # For each telescope, create a mask of valid tiles based on the surveys assigned
        tel_tile_masks = np.zeros((total_tels, self.cache.grid.ntiles), dtype=bool)
        # For each telescope create an hour angle limit based on the surveys assigned
        tel_ha_limit = np.full(total_tels, 180)  # degrees
        # And determine which telescopes are affected by reserved time
        tel_reserved_mask = np.zeros(total_tels, dtype=bool)
        # Set minimum time between revisits for each tile based on surveys
        tile_revisit_times = np.zeros(self.cache.grid.ntiles)  # in seconds
        if self.surveys is not None:
            for survey in self.surveys:
                # Update revisit times
                if survey.revisit_time is not None:
                    for tile in survey.indices:
                        tile_revisit_times[tile] = survey.revisit_time.to_value(u.s)
                for tel in survey.tels:
                    tel_tile_masks[tel - 1] |= np.isin(
                        self.cache.grid.tilenames, survey.tiles
                    )
                    # Update hour angle limit
                    if survey.ha_limit is not None:
                        tel_ha_limit[tel - 1] = min(
                            tel_ha_limit[tel - 1], survey.ha_limit.deg
                        )
                if survey.priority != "high":
                    for tel in survey.tels:
                        tel_reserved_mask[tel - 1] = True
        else:
            tel_tile_masks[:, :] = True  # All tiles are valid if no surveys assigned
            tel_reserved_mask[:] = (
                True  # Reserved time affects all tels if no surveys assigned
            )

        tiles = self.cache.grid.tilenames
        observations = []
        last_observation_per_tile = np.zeros(len(tiles))  # MJD of last observation
        obs_count = np.zeros(len(tiles))
        for t_i, time in enumerate(self.times):
            # Random check for ToO time, which takes precedence over all surveys
            too_time = False
            reserve_time = False
            if np.random.rand() < self.too_fraction:
                logger.debug(f"Time {time.iso}: ToO time, skipping observations")
                too_time = True

            # Random check for reserved time
            if np.random.rand() < self.reserved_fraction:
                logger.debug(f"Time {time.iso}: Reserved time, skipping observations")
                reserve_time = True
            else:
                reserve_time = False
            # Loop over each node
            for n_i, node in enumerate(self.nodes):
                # Check for daytime
                sun_alt = solar_alts[t_i, n_i]
                if sun_alt > self.twilight_limit:
                    logger.debug(
                        f"Time {time.iso}: Node {node.name}: Daylight (Sun alt {sun_alt:.2f}°)"
                    )
                    continue
                # Check tile visibility
                tile_alts = self.cache.alt[t_i, n_i]
                tile_azis = self.cache.az[t_i, n_i]
                tile_has = self.cache.ha[t_i, n_i]
                visibility_mask = tile_alts > node.horizon  # degrees
                # TODO: Smarter tile selection logic based on hour angle, airmass, etc.
                for tel in range(tels_per_node[n_i]):
                    # Apply survey tile mask
                    tel_idx = tel_start_indices[n_i] + tel
                    if too_time:
                        logger.debug(
                            f"Time {time.iso}: Node {node.name}, Tel {tel_idx + 1}: ToO time, skipping observation"
                        )
                        observations.append(
                            {
                                "time": time.mjd,
                                "tile": None,
                                "target": "ToO",
                                "altitude": None,
                                "azimuth": None,
                                "hour_angle": None,
                                "node": node.name,
                                "telescope": tel_idx + 1,
                                "sun_alt": sun_alt,
                            }
                        )
                        continue
                    if reserve_time and tel_reserved_mask[tel_idx]:
                        logger.debug(
                            f"Time {time.iso}: Node {node.name}, Tel {tel_idx + 1}: Reserved time, skipping observation"
                        )
                        observations.append(
                            {
                                "time": time.mjd,
                                "tile": None,
                                "target": "Reserved",
                                "altitude": None,
                                "azimuth": None,
                                "hour_angle": None,
                                "node": node.name,
                                "telescope": tel_idx + 1,
                                "sun_alt": sun_alt,
                            }
                        )
                        continue

                    # Apply hour angle limit
                    ha_mask = np.abs(tile_has) <= tel_ha_limit[tel_idx]
                    validity_mask = visibility_mask & ha_mask
                    # Apply survey tile mask
                    validity_mask = validity_mask & tel_tile_masks[tel_idx]
                    # Apply revisit time mask
                    if np.any(tile_revisit_times > 0):
                        time_since_last_obs = (
                            time.mjd - last_observation_per_tile
                        ) * 86400.0  # seconds
                        revisit_mask = (time_since_last_obs >= tile_revisit_times) | (
                            last_observation_per_tile == 0
                        )
                        validity_mask = validity_mask & revisit_mask

                    # Select the valid tiles with the least observations so far
                    candidate_tiles = np.where(validity_mask)[0]
                    if len(candidate_tiles) == 0:
                        logger.debug(
                            f"Time {time.iso}: Node {node.name}, Tel {tel_start_indices[n_i] + tel + 1}: No visible tiles above horizon"
                        )
                        continue
                    # TODO: Improve tile selection strategy
                    # Find the tile(s) with the least observations
                    least_obs_tiles = candidate_tiles[
                        np.where(
                            obs_count[candidate_tiles]
                            == np.min(obs_count[candidate_tiles])
                        )
                    ]
                    # Tiebreak based on altitude
                    best_tile_idx = least_obs_tiles[
                        np.argmax(tile_alts[least_obs_tiles])
                    ]
                    observed_tile_idx = best_tile_idx
                    # Update observation count
                    obs_count[observed_tile_idx] += 1
                    obs = {
                        "time": time.mjd,
                        "tile": tiles[observed_tile_idx],
                        "target": tiles[observed_tile_idx],
                        "altitude": tile_alts[observed_tile_idx],
                        "azimuth": tile_azis[observed_tile_idx],
                        "hour_angle": tile_has[observed_tile_idx],
                        "node": node.name,
                        "telescope": tel_start_indices[n_i] + tel + 1,  # 1-indexed
                        "sun_alt": sun_alt,
                    }
                    observations.append(obs)
                    last_observation_per_tile[observed_tile_idx] = time.mjd
                    logger.debug(
                        f"Time {time.iso}: Node {node.name}, Tel {tel_start_indices[n_i] + tel + 1}: Tile {obs['tile']} at Alt {obs['altitude']:.2f}° (Sun alt {sun_alt:.2f}°)"
                    )
        self.results = observations
        logger.info(f"Simulation complete, total observations: {len(observations)}")

    def save_results(self, filename: Union[str, Path]):
        """
        Save the simulation run results to a CSV file
        :param filename:
        :return:
        """
        import pandas as pd

        if not self.results:
            logger.warning("No results to save")
            return

        df = pd.DataFrame(self.results)
        with open(filename, "w") as f:
            # Write metadata as comments
            f.write("# Simulation Results\n")
            f.write(f"# Start Time: {self.start_time.iso}\n")
            f.write(f"# Stop Time: {self.stop_time.iso}\n")
            f.write(f"# Time Step: {self.time_step.to_value(u.s)} seconds\n")
            f.write(f"# Nodes: {', '.join([node.name for node in self.nodes])}\n")
            f.write(f"# Twilight Limit: {self.twilight_limit} degrees\n")
            f.write(f"# Reserved Fraction: {self.reserved_fraction}\n")
            f.write(f"# ToO Fraction: {self.too_fraction}\n")
            f.write("# Surveys:\n")
            if self.surveys is not None:
                for survey in self.surveys:
                    f.write(
                        f"#   {survey.name} Survey with {len(survey.tiles)} tiles on Telescopes: {', '.join(map(str, survey.tels))}\n"
                    )
            else:
                f.write("#   No surveys assigned\n")
            # Write data
            df.to_csv(f, index=False)
        logger.info(f"Simulation results saved to {filename}")

    def _setup(self):
        """
        Set up the simulation by calculating solar positions and preparing telescope arrays
        :return:
        """
        # Cache AltAz data if not already cached
        if self.cache.data is None:
            logger.info("No AltAz data cache, generating new one")
            self.cache.generate_cache()

        # Find sun's position at each time step
        # logger.debug("Calculating Solar positions")
        # sun = get_sun(self.times)
        # locations = concat_earth_locations(self.locations)
        # frame = AltAz(
        #    obstime=self.times[:, np.newaxis], location=locations[np.newaxis :,]
        # )
        # self.sun_altaz = sun[:, np.newaxis].transform_to(frame)

        # Populate telescope array
        self.tels = (
            np.arange(
                np.fromiter([node.telescopes for node in self.nodes], dtype=int).sum()
            )
            + 1
        )  # 1-indexed


class HEATSCOLDSimulator(Simulator):
    """
    A simulator for the HEATSCOLD survey strategy.
    """

    def __init__(
        self,
        nodes: list[GOTONode] = None,
        start_time: Time = None,
        stop_time: Time = None,
        time_step: u.Quantity[u.s] = 5 * u.min,
        cache: AltAzCache = None,
    ):
        super().__init__(nodes, start_time, stop_time, time_step, cache)

    def run(self):
        """
        Run the HEATSCOLD simulation
        :return:
        """
        super().run()
        # Implement HEATSCOLD-specific logic here
        pass
