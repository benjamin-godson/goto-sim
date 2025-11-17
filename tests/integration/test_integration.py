from goto_sim.sim import Simulator
from goto_sim.scheduling import Survey


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
        tile_num = int(obs["tile"][1:])
        if tile_num % 2 == 0:
            assert obs["telescope"] in [2, 4]
        else:
            assert obs["telescope"] in [1, 3]


def test_integration_heat_survey():
    """
    Test integration between Simulator and HEATSurvey classes.
    """
    from goto_sim.scheduling import HEATSurvey

    sim = Simulator()
    heat_survey = HEATSurvey(tels=[1, 2])
    cold_survey = heat_survey.generate_colds()

    sim.add_survey(heat_survey)
    sim.add_survey(cold_survey)

    sim.run()

    heat_tiles = set(heat_survey.tiles)
    cold_tiles = set(cold_survey.tiles)

    for obs in sim.results:
        tile = obs["tile"]
        telescope = obs["telescope"]
        if tile in heat_tiles:
            assert telescope in heat_survey.tels
        elif tile in cold_tiles:
            assert telescope in cold_survey.tels
        else:
            assert False, f"Observed tile {tile} not in HEAT or COLD surveys."
