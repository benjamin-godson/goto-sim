import numpy as np
import astropy.units as u

from goto_sim.scheduling import Survey


def test_create_survey():
    """
    Test creating a Survey object.
    """
    survey = Survey(name="Test Survey")
    assert survey.name == "Test Survey"
    assert survey.tiles == []
    assert survey.revisit_time == 1 * u.day
    assert (survey.tel_mask == np.array([1, 2, 3, 4])).all()

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
