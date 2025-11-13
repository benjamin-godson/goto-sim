"""
A test module that tests your example module.

Some people prefer to write tests in a test file for each function or
method/ class. Others prefer to write tests for each module. That decision
is up to you. This test example provides a single test for the example.py
module.
"""
import numpy as np
from astropy.coordinates import SkyCoord

from goto_sim.sim import GOTONode, Simulator, AltAzCache
from astropy.time import Time
import astropy.units as u

def test_create_node():
    """
    Create a GOTONode node
    """
    node = GOTONode(site='goto-north', name='Test Node')
    assert node is not None
    assert node.name == 'Test Node'

    node = GOTONode(site='goto-south', name='Test Node')
    assert node is not None
    assert node.name == 'Test Node'

    node = GOTONode(site='goto-south')
    assert node is not None
    assert node.name == 'goto-south'

def test_create_simulator():
    """
    Test that the default simulator constructor works as expected.
    """
    sim = Simulator()
    assert isinstance(sim, Simulator)

def test_generate_altaz_cache():

    cache = AltAzCache(start_time=Time.now(), stop_time=Time.now() + 24*u.hour)
    assert isinstance(cache, AltAzCache)

    cache = AltAzCache(start_time=Time("2026-01-01T00:00:00"),
                       stop_time=Time("2026-01-02T00:00:00"))
    assert isinstance(cache, AltAzCache)

    cache.generate_cache()
    cache.write_data('test_altaz_cache.npz', overwrite=True)
    loaded_cache = AltAzCache(start_time=Time("2026-01-01T00:00:00"),
                              stop_time=Time("2026-01-02T00:00:00"))
    loaded_cache.load_data('test_altaz_cache.npz')
    assert np.array_equal(cache.alt, loaded_cache.alt)
    assert np.array_equal(cache.az, loaded_cache.az)
    assert np.array_equal(cache.times.mjd, loaded_cache.times.mjd)

    assert np.isclose(loaded_cache.alt, cache.alt).all()
    assert np.isclose(loaded_cache.az, cache.az).all()
    assert np.isclose(loaded_cache.times.mjd, cache.times.mjd).all()


