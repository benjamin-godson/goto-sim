
import numpy as np
import pytest
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, Longitude, TETE, \
    Latitude
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
import astropy.units as u
from astropy.time import Time

from goto_sim.utils import generate_altitude_cache
from time import perf_counter

def test_hour_angle_to_altitude():
    """
    Test the hour_angle_to_altitude function.
    """
    from astropy.coordinates import Angle
    from goto_sim.utils import hour_angle_to_altitude
    obstime = Time('2025-01-01T00:00:00')
    location = EarthLocation(lon=0, lat=45, height=0)
    lst = obstime.sidereal_time('apparent', longitude=location.lon)
    coords = np.meshgrid(np.linspace(0, 365, 25),
                         np.linspace(-89, 89, 25))
    ra = Angle(coords[0].flatten(), unit='deg')
    dec = Angle(coords[1].flatten(), unit='deg')
    coords = SkyCoord(ra=ra, dec=dec, frame='icrs')
    current_coords = coords.transform_to(TETE(obstime=obstime,
                                              location=location))
    ha = lst - current_coords.ra
    az_frame = AltAz(obstime=obstime,
                     location=location,
                     pressure=0)
    altaz = coords.transform_to(az_frame)
    expected_alt = altaz.alt.degree
    alt = hour_angle_to_altitude(ha, dec, location.lat)
    np.testing.assert_allclose(alt.degree, expected_alt)

def test_altitude_cache():
    """
    Test the generate_altitude_cache function.
    """
    from goto_sim.utils import generate_altitude_cache
    from gototile.grid import SkyGrid
    lapalma = EarthLocation.of_site('lapalma')
    sso = EarthLocation.of_site('sso')
    longitudes = Longitude([lapalma.lon, sso.lon])
    latitudes = Latitude([lapalma.lat, sso.lat])
    heights = [lapalma.height, sso.height]
    locations = EarthLocation.from_geodetic(longitudes, latitudes, heights)
    times = Time('2025-01-01T00:00:00') + np.arange(12*24*30*12) * 5 * u.min
    grid = SkyGrid.from_name('GOTO')
    coords = grid.coords
    frame = AltAz(
        location=locations[:, np.newaxis, np.newaxis],
        obstime=times[np.newaxis, np.newaxis, :],
    )
    start = perf_counter()
    with erfa_astrom.set(ErfaAstromInterpolator(1*u.day)):
        altaz = coords[np.newaxis, :, np.newaxis].transform_to(frame)
    end = perf_counter()
    print(f"Direct (approx) transform took {end - start:.2f} seconds")
    approx_alts = altaz.alt.degree
    start = perf_counter()
    #altaz = coords[np.newaxis, :, np.newaxis].transform_to(frame)
    end = perf_counter()
    print(f"Direct (exact) transform took {end - start:.2f} seconds")
    #exact_alts = altaz.alt.degree
    # Find the mean, max, and std difference between the approx and exact
    #diff = np.abs(approx_alts - exact_alts)
    #print(f"Mean difference: {np.mean(diff):.6f} degrees")
    #print(f"Max difference: {np.max(diff):.6f} degrees")
    #print(f"Std difference: {np.std(diff):.6f} degrees")




