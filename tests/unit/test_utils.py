from time import perf_counter

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, Latitude, Longitude
from astropy.coordinates.erfa_astrom import ErfaAstromInterpolator, erfa_astrom
from astropy.time import Time
from gototile.grid import SkyGrid


def test_altitude_cache():
    """
    Test the generate_altitude_cache function.
    """

    lapalma = EarthLocation.of_site("lapalma")
    sso = EarthLocation.of_site("sso")
    longitudes = Longitude([lapalma.lon, sso.lon])
    latitudes = Latitude([lapalma.lat, sso.lat])
    heights = [lapalma.height, sso.height]
    locations = EarthLocation.from_geodetic(longitudes, latitudes, heights)
    times = Time("2025-01-01T00:00:00") + np.arange(12 * 24) * 5 * u.min
    grid = SkyGrid.from_name("GOTO")
    coords = grid.coords
    frame = AltAz(
        location=locations[:, np.newaxis, np.newaxis],
        obstime=times[np.newaxis, np.newaxis, :],
    )
    start = perf_counter()
    with erfa_astrom.set(ErfaAstromInterpolator(1 * u.day)):
        altaz = coords[np.newaxis, :, np.newaxis].transform_to(frame)
    end = perf_counter()
    print(f"Direct (approx) transform took {end - start:.2f} seconds")
    approx_alts = altaz.alt.degree
    start = perf_counter()
    altaz = coords[np.newaxis, :, np.newaxis].transform_to(frame)
    end = perf_counter()
    print(f"Direct (exact) transform took {end - start:.2f} seconds")
    exact_alts = altaz.alt.degree
    # Find the mean, max, and std difference between the approx and exact
    diff = np.abs(approx_alts - exact_alts)
    print(f"Mean difference: {np.mean(diff):.6f} degrees")
    print(f"Max difference: {np.max(diff):.6f} degrees")
    print(f"Std difference: {np.std(diff):.6f} degrees")


def test_concat_earth_location():
    from goto_sim.utils import concat_earth_locations

    lp = EarthLocation.of_site("lapalma")
    sso = EarthLocation.of_site("sso")
    combined = concat_earth_locations([lp, sso])
    assert isinstance(combined, EarthLocation)
    assert u.isclose(combined[0].lon, lp.lon)
    assert u.isclose(combined[0].lat, lp.lat)
    assert u.isclose(combined[0].height, lp.height)

    assert u.isclose(combined[1].lon, sso.lon)
    assert u.isclose(combined[1].lat, sso.lat)
    assert u.isclose(combined[0].height, lp.height)
