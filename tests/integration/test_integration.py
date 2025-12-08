from astropy.coordinates import Angle

from goto_sim.sim import Simulator
from goto_sim.scheduling import Survey, HEATSurvey


def test_integration_simulator_survey():
    """
    Test integration between Simulator and Survey classes.
    """
    sim = Simulator()
    survey = Survey(
        name="Integration Test Survey", tiles=["T0001", "T0002"], tels=[1, 2]
    )

    # Add survey to simulator
    sim.add_survey(survey)

    # Check that the survey is correctly added
    assert survey in sim.surveys
    assert survey.tiles == ["T0001", "T0002"]
    assert (survey.tels == [1, 2]).all()

    sim = Simulator()
    survey = Survey(name="Only Tel 3 Survey", tiles=sim.cache.grid.tilenames, tels=[3])
    sim.add_survey(survey)
    sim.run()

    for obs in sim.results:
        assert obs["telescope"] == 3

    # Assign odd tiles to telescope 1/3 and even tiles to telescope 2/4
    sim = Simulator()

    odd_tiles = [tile for i, tile in enumerate(sim.cache.grid.tilenums) if i % 2 == 0]
    odd_tiles = ["T" + str(tile).zfill(4) for tile in odd_tiles]
    survey_odd = Survey(name="Odd Tiles Survey", tiles=odd_tiles, tels=[1, 3])
    sim.add_survey(survey_odd)
    even_tiles = [tile for i, tile in enumerate(sim.cache.grid.tilenums) if i % 2 == 1]
    even_tiles = ["T" + str(tile).zfill(4) for tile in even_tiles]
    survey_even = Survey(name="Even Tiles Survey", tiles=even_tiles, tels=[2, 4])
    sim.add_survey(survey_even)

    sim.run()

    for obs in sim.results:
        if obs["tile"] is None:
            continue
        tile_num = int(obs["tile"][1:])
        if tile_num % 2 == 0:
            assert obs["telescope"] in [2, 4]
        else:
            assert obs["telescope"] in [1, 3]


def test_integration_heat_survey():
    """
    Test integration between Simulator and HEATSurvey classes.
    """

    sim = Simulator()
    heat_survey = HEATSurvey(tels=[1, 3], ha_limit=Angle(2, "hour"))
    cold_survey = heat_survey.generate_colds()

    sim.add_survey(heat_survey)
    sim.add_survey(cold_survey)

    sim.run()

    heat_tiles = set(heat_survey.tiles)
    cold_tiles = set(cold_survey.tiles)

    for obs in sim.results:
        tile = obs["tile"]
        if tile is None:
            continue
        telescope = obs["telescope"]
        if tile in heat_tiles:
            assert telescope in heat_survey.tels
        elif tile in cold_tiles:
            assert telescope in cold_survey.tels
        else:
            assert False, f"Observed tile {tile} not in HEAT or COLD surveys."


def test_reserve_time():
    """
    Test that the Simulator respects reserved time for non-high priority observations.
    """
    sim = Simulator(reserved_fraction=1)
    heats = HEATSurvey()
    cold = heats.generate_colds()
    sim.add_survey(heats)
    sim.add_survey(cold)
    sim.run()
    # With 100% reserved time, there should be no observations from COLD survey
    for obs in sim.results:
        if obs["tile"] is None:
            continue
        assert obs["tile"] in heats.tiles


def test_too_time():
    """
    Test that the Simulator respects TOO time fraction.
    """
    too_fraction = 1.0
    sim = Simulator(too_fraction=too_fraction)
    heats = HEATSurvey()
    cold = heats.generate_colds()
    sim.add_survey(heats)
    sim.add_survey(cold)
    sim.run()
    # With 100% TOO time, there should be no observations from HEAT or COLD surveys
    for obs in sim.results:
        if obs["tile"] is None:
            continue
        assert False, f"Observed tile {obs['tile']} despite 100% TOO time."

    too_fraction = 0.5
    sim = Simulator(too_fraction=too_fraction, reserved_fraction=0)
    sim.add_survey(heats)
    sim.add_survey(cold)
    sim.run()
    total_observations = len(sim.results)
    too_observations = sum(1 for obs in sim.results if obs["tile"] is None)
    observed_too_fraction = too_observations / total_observations
    assert abs(observed_too_fraction - too_fraction) < 0.1, (
        f"Observed TOO fraction {observed_too_fraction:.2f} deviates from expected {too_fraction:.2f}"
    )


def test_hourangle_limit():
    """
    Test that the HEATSurvey respects hour angle limit.
    """
    sim = Simulator()
    for ha in [1, 3, 6]:
        ha_limit = Angle(ha, "hour")
        heat_survey = HEATSurvey(tels=[1, 3], ha_limit=ha_limit)
        cold_survey = heat_survey.generate_colds()
        sim.add_survey(heat_survey)
        sim.add_survey(cold_survey)
        sim.run()

        for obs in sim.results:
            if obs["telescope"] in heat_survey.tels:
                ha = obs["hour_angle"]
                assert abs(ha) <= ha_limit.deg, (
                    f"Observed hour angle {ha} exceeds limit {ha_limit} for HEAT survey."
                )
