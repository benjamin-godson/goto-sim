import numpy as np
from astropy.coordinates import Angle, EarthLocation, AltAz
from gototile.grid import SkyGrid
from astropy.time import Time


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
    :return: Altitude cache as a 2D numpy array
    """
    coords = grid.coords
    frame = AltAz(obstime=times[:, np.newaxis], location=location, pressure=0)
    altaz = coords.transform_to(frame)


def concat_earth_locations(locations: list[EarthLocation]) -> EarthLocation:
    """
    Combine a sequence EarthLocation objects into a single EarthLocation object
    :param locations: list of EarthLocation objects
    :return: EarthLocation object
    """
    latitudes = [x.lat for x in locations]
    longitudes = [x.lon for x in locations]
    heights = [x.height for x in locations]
    return EarthLocation.from_geodetic(lat=latitudes, lon=longitudes, height=heights)
