import astropy.units as u
import numpy as np

from goto_sim.scheduling import HEATSurvey, Survey


def test_create_survey():
    """
    Test creating a Survey object.
    """
    survey = Survey(name="Test Survey")
    assert survey.name == "Test Survey"
    assert survey.tiles == []
    assert survey.revisit_time == 1 * u.day
    assert (survey.tels == np.array([1, 2, 3, 4])).all()

    survey.add_tiles("T0001")
    assert survey.tiles == ["T0001"]

    survey.add_tiles(["T0002", "T0003"])
    assert survey.tiles == ["T0001", "T0002", "T0003"]
    assert (survey.indices == np.array([0, 1, 2])).all()

    survey.add_tiles(["T0002", "T0003"])
    assert survey.tiles == ["T0001", "T0002", "T0003"]
    assert (survey.indices == np.array([0, 1, 2])).all()

    survey.remove_tiles("T0002")
    assert survey.tiles == ["T0001", "T0003"]

    survey.remove_tiles(["T0001", "T0003"])
    assert survey.tiles == []


def test_survey_tel_mask():
    """
    Test that the survey telescope mask is created correctly.
    """
    survey = Survey(name="Test Survey", tels=[1, 3])
    expected_mask = np.array([True, False, True, False])
    assert (survey.tel_mask == expected_mask).all()
    assert (survey.tels == [1, 3]).all()

    survey = Survey(name="Test Survey", tels=np.array([2, 4]))
    expected_mask = np.array([False, True, False, True])
    assert (survey.tel_mask == expected_mask).all()
    assert (survey.tels == [2, 4]).all()


def test_survey_tile_verification():
    """
    Test that the survey tile verification raises an error for invalid tiles.
    """
    survey = Survey(name="Test Survey")
    try:
        survey.add_tiles("INVALID_TILE")
    except ValueError as e:
        assert str(e) == "Invalid tile names in survey: ['INVALID_TILE']"
    else:
        assert False, "ValueError not raised for invalid tile."

    try:
        survey.add_tiles(["T0001", "INVALID_TILE"])
    except ValueError as e:
        assert str(e) == "Invalid tile names in survey: ['INVALID_TILE']"
    else:
        assert False, "ValueError not raised for invalid tile."

    survey = Survey(name="Test Survey")
    survey.add_tiles(["T0002", "T0003"])
    assert survey.tiles == ["T0002", "T0003"]


def test_heats_survey():
    """
    Test that HEATS survey works as expected.
    """

    heats = HEATSurvey()
    cold = heats.generate_colds()
    heats_tiles = set(heats.tiles)
    cold_tiles = set(cold.tiles)
    all_tiles = set(heats.grid.tilenames)

    assert (heats.tels == [1, 3]).all()
    assert (cold.tels == [2, 4]).all()

    assert heats_tiles.isdisjoint(cold_tiles)
    assert heats_tiles.union(cold_tiles) == all_tiles
    assert set(np.append(heats.tels, cold.tels)) == {1, 2, 3, 4}
