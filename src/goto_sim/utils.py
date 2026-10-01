from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from astropy.coordinates import AltAz, Angle, EarthLocation
from astropy.time import Time
from gototile.grid import SkyGrid


def hour_angle_to_altitude(ha: Angle, dec: Angle, lat: Angle) -> Angle:
    """
    Convert hour angle and declination to altitude
    :param ha: Hour angle in degrees
    :param dec: Declination in degrees
    :param lat: Latitude in degrees
    :return: Altitude in degrees
    """
    alt = np.arcsin(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(ha))
    return Angle(alt)


def generate_altitude_cache(grid: SkyGrid, times: Time, location: EarthLocation):
    """
    Generate an altitude cache for a given sky grid, times and location
    :param grid: SkyGrid object
    :param times: Array of times
    :param location: EarthLocation object
    :return: Altitude cache as a 2D numpy array (degrees)
    """
    coords = grid.coords
    frame = AltAz(obstime=times[:, np.newaxis], location=location, pressure=0)
    altaz = coords.transform_to(frame)
    # Return altitude values in degrees; caller can compute azimuth separately if needed
    return altaz.alt.deg


def concat_earth_locations(locations: Iterable[EarthLocation]) -> EarthLocation:
    """
    Combine a sequence EarthLocation objects into a single EarthLocation object
    :param locations: iterable of EarthLocation objects
    :return: EarthLocation object
    """
    latitudes = [x.lat for x in locations]
    longitudes = [x.lon for x in locations]
    heights = [x.height for x in locations]
    return EarthLocation.from_geodetic(lat=latitudes, lon=longitudes, height=heights)


def load_tilelist(filename: str | os.PathLike) -> list[str]:
    """
    Load a list of tile names from a text file. Accepts str or Path-like objects.
    :param filename: Path to the text file
    :return: List of tile names
    """
    path = Path(filename)
    # Open using Path.open for Path-like safety
    with path.open("r", encoding="utf-8") as f:
        tiles = [line.strip() for line in f if line.strip()]
    return tiles


def save_tilelist(tiles: Iterable[str], filename: str | os.PathLike):
    """
    Save a list of tile names to a text file. Accepts str or Path-like objects.
    :param tiles: List (or iterable) of tile names
    :param filename: Path to the text file
    """
    path = Path(filename)
    # Ensure parent directory exists
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tile in tiles:
            f.write(f"{tile}\n")
