"""
A test module that tests your example module.

Some people prefer to write tests in a test file for each function or
method/ class. Others prefer to write tests for each module. That decision
is up to you. This test example provides a single test for the example.py
module.
"""

from time import perf_counter, perf_counter_ns

import numpy as np
from astropy.coordinates import SkyCoord

from goto_sim.sim import GOTONode, Simulator, AltAzCache
from astropy.time import Time
import astropy.units as u


def test_create_node():
    """
    Create a GOTONode node
    """
    node = GOTONode(site="goto-north", name="Test Node")
    assert node is not None
    assert node.name == "Test Node"

    node = GOTONode(site="goto-south", name="Test Node")
    assert node is not None
    assert node.name == "Test Node"

    node = GOTONode(site="goto-south")
    assert node is not None
    assert node.name == "goto-south"


def test_create_simulator():
    """
    Test that the default simulator constructor works as expected.
    """
    sim = Simulator()
    assert isinstance(sim, Simulator)


def test_generate_altaz_cache():
    cache = AltAzCache(start_time=Time.now(), stop_time=Time.now() + 24 * u.hour)
    assert isinstance(cache, AltAzCache)

    cache = AltAzCache(
        start_time=Time("2026-01-01T00:00:00"), stop_time=Time("2026-01-02T00:00:00")
    )
    assert isinstance(cache, AltAzCache)

    cache.generate_cache()
    cache.write_data("test_altaz_cache.npz", overwrite=True)
    loaded_cache = AltAzCache(
        start_time=Time("2026-01-01T00:00:00"), stop_time=Time("2026-01-02T00:00:00")
    )
    loaded_cache.load_data("test_altaz_cache.npz")
    assert np.array_equal(cache.alt, loaded_cache.alt)
    assert np.array_equal(cache.az, loaded_cache.az)
    assert np.array_equal(cache.times.mjd, loaded_cache.times.mjd)

    assert np.isclose(loaded_cache.alt, cache.alt).all()
    assert np.isclose(loaded_cache.az, cache.az).all()
    assert np.isclose(loaded_cache.times.mjd, cache.times.mjd).all()


def test_rank_tiles():
    cache = AltAzCache(
        start_time=Time("2026-01-01T00:00:00"), stop_time=Time("2026-02-01T00:00:00")
    )
    cache.generate_cache()
    tilenames = cache.grid.tilenames
    start = perf_counter()
    for t, time in enumerate(cache.times):
        for n, node in enumerate(cache.nodes):
            alts = cache.alt[t, n]
            sorted_indices = np.argsort(alts)
            _ = [tilenames[i] for i in sorted_indices[::-1]]
    end = perf_counter()
    print(
        f"Ranking {cache.grid.ntiles} tiles for {cache.n_times} timesteps and {len(cache.nodes)} nodes took {end - start:g} seconds"
    )

def test_run_simulator():
    sim = Simulator()
    sim.run()

def bench_generate_cache(n_times: int = 1000):
    """
    Benchmark the altitude cache generation for a given number of time steps.
    :param n_times: Number of time steps to simulate
    """
    cache = AltAzCache(start_time=Time.now(), n_times=n_times)
    start = perf_counter_ns()
    cache.generate_cache()
    end = perf_counter_ns()
    return end - start
