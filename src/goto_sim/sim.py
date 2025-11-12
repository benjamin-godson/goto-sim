"""
Core module for simulation classes and logic.
"""
from astropy.coordinates import EarthLocation

class GOTONode:
    """
    A single GOTO node in the simulation. Contains information about the site as well
    as the sidereal time and tile hour angles for each time step of the simulation.
    """
    def __init__(self, site: str, name: str = None, telescopes: int = 2):
        """
        Initialize the GOTO node
        """
        try:
            self.location = self._resolve_site(site)
        except ValueError as e:
            raise e
        self.name = name if name is not None else site
        self.telescopes = telescopes
        pass

    @staticmethod
    def _resolve_site(self, site: str):
        """
        Resolve the site name to a location
        :return:
        """
        sites = {'goto-north': 'lapalma',
                 'goto-n': 'lapalma',
                 'north': 'lapalma',
                 'n': 'lapalma',
                 'goto-south': 'sso',
                 'goto-s': 'sso',
                 'south': 'sso'}
        location_name = sites.get(site.lower(), site)
        try:
            location = EarthLocation.of_site(location_name)
        except Exception as e:
            raise ValueError(f"Could not resolve site name '{site}': {e},"
                             f" try goto-n or goto-s")
        return location

class Simulator:
    """
    A single simulation of an observing run for GOTO.
    """

    def __init__(self, nodes: list[GOTONode] = None, start_time=None, end_time=None,):
        """
        Initialize the simulator
        """
        if nodes is None:
            self.nodes = [GOTONode(site='goto-north'),
                          GOTONode(site='goto-south')]
        else:
            self.nodes = nodes

    def run(self):
        """
        Run the simulation
        :return:
        """
        pass

    def _get_tile_hour_angles(self):
        """
        Calculate the hour angles for all tiles in the sky
        :return:
        """
        for node in self.nodes:
            pass

    def _time_step(self):
        """
        Advance the simulation by one time step. This will be run in a loop
        until the simulation is complete.
        :return:
        """
