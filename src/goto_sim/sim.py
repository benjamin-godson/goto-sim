"""
Core module for simulation classes and logic.
"""

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, erfa_astrom
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.time import Time, TimeDelta
import astropy.units as u
from gototile.grid import SkyGrid

from .utils import concatenate_earth_locations
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
        pass

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
    Stores the AltAz coordinates of each GOTO tile for each time step of a
    simulation.
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
            self.times: Time = self.start_time + np.arange(self.n_times) * self.time_step

        self.data = data
        self.grid = grid
        self.dtype = dtype

    def get_data(self) -> SkyCoord:
        if self.data is None:
            raise ValueError(
                "Cache data has not been generated yet, call generate_cache() first"
            )
        return self.data

    def generate_cache(self):
        if self.n_times < 20_000:
            locations = concatenate_earth_locations(self.locations)
            times = self.times
            coords = self.grid.coords
            frame = AltAz(
                location=locations[:, np.newaxis, np.newaxis],
                obstime=times[np.newaxis, np.newaxis, :],
            )
            with erfa_astrom.set(ErfaAstromInterpolator(1 * u.day)):
                altaz: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(frame)
            # Saving the whole SkyCoord is a bit inefficient, especially for large n_times
            alts = altaz.alt.deg
            azs = altaz.az.deg
            # Combine both values into an array of size (n_times, n_nodes, n_tiles, 2)
            data = np.stack([alts, azs], axis=-1).transpose(2, 0, 1, 3)
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
                locations = concatenate_earth_locations(
                    [x.location for x in self.nodes]
                )
                coords = self.grid.coords
                frame = AltAz(
                    location=locations[:, np.newaxis, np.newaxis],
                    obstime=times_chunk[np.newaxis, np.newaxis, :],
                )
                with erfa_astrom.set(ErfaAstromInterpolator(1 * u.day)):
                    altaz: SkyCoord = coords[np.newaxis, :, np.newaxis].transform_to(
                        frame
                    )
                alts = altaz.alt.deg
                azs = altaz.az.deg
                data_chunk = np.stack([alts, azs], axis=-1).transpose(2, 0, 1, 3)
                all_data.append(data_chunk)
            data = np.concatenate(all_data, axis=0)

        self.data = data.astype(self.dtype)
        self._validate()

    def write_data(self, filename: str, overwrite: bool = True) -> None:
        """
        Write the cache to a numpy file, including the times and node locations
        :param filename:
        :return:
        """
        times = self.times
        locations = [x.location for x in self.nodes]
        data = self.get_data()

        # Combine metadata and data into a single npz file
        np.savez_compressed(
            filename,
            times=times.mjd,
            locations_lat=[loc.lat.deg for loc in locations],
            locations_lon=[loc.lon.deg for loc in locations],
            locations_height=[loc.height.to(u.m).value for loc in locations],
            data=data,
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
        expected_shape = (self.n_times, len(self.nodes), self.grid.ntiles, 2)
        if data.shape != expected_shape:
            raise ValueError(
                f"Data shape in file {data.shape} does not match expected shape {expected_shape}"
            )
        self.data = data.astype(self.dtype)
        self._validate()

    def _validate(self) -> bool:
        """
        Validate that the cache data has the correct shape, and expected properties
        :return:
        """
        data = self.get_data()
        expected_shape = (self.n_times, len(self.nodes), self.grid.ntiles, 2)
        if data.shape != expected_shape:
            raise ValueError(
                f"Cache data shape {data.shape} does not match expected shape {expected_shape}"
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


class Simulator:
    """
    A single simulation of an observing run for GOTO.
    """

    def __init__(
        self,
        nodes: list[GOTONode] = None,
        start_time: Time = None,
        end_time: Time = None,
        time_step: u.Quantity[u.s] = 5 * u.min,
        cache: AltAzCache = None,
    ):
        """
        Initialize the simulator
        """
        if nodes is None:
            nodes = [
                GOTONode(site="goto-north"),
                GOTONode(site="goto-south"),
            ]
        self.nodes = nodes
        if start_time is None:
            logger.debug("Start time not specified, using current time")
            start_time = Time.now()
        if end_time is None:
            logger.debug("End time not specified, using start_time + 24 hours")
            end_time = start_time + 24 * u.hour
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        if cache is None:
            cache = AltAzCache(
            start_time=start_time, stop_time=end_time, nodes=nodes, time_step=time_step
        )
        self.cache = cache
        self.times: Time = self.cache.times

    def load_cached_data(self, filename: str):
        """
        Load cached AltAz data from a file
        :param filename:
        :return:
        """
        self.cache.load_data(filename)

    def run(self):
        """
        Run the simulation
        :return:
        """
        if self.cache.data is None:
            logger.info("No AltAz data cache, generating new one")
            self.cache.generate_cache()
        pass
