"""
Tools to support scheduling in the GOTO simulator. Allowing for creation of surveys.
"""

from typing import Union

import numpy as np
import astropy.units as u
from gototile.grid import SkyGrid


class Survey:
    """
    A survey is a collection of tiles to be observed according to some strategy.
    :param name: Name of the survey.
    :param tiles: List of tile names (e.g. T0001, T0002, etc.) to include in the survey.
    :param revisit_time: Time between revisits to the same tile.
    :param tels: List of telescopes to use for this survey. If None, all telescopes are used.
    :param tel_mask: Boolean mask array indicating which telescopes to use. If None, all telescopes are used.
    :param grid: SkyGrid object defining the tile grid. Defaults to GOTO grid.
    """

    def __init__(
        self,
        name: str,
        tiles: Union[None, list[str]] = None,
        revisit_time: u.Quantity[u.day] = 1 * u.day,
        tels: Union[None, np.ndarray, list] = None,
        grid: SkyGrid = SkyGrid.from_name("GOTO"),
    ):
        self.name = name
        if tiles is not None:
            self.tiles = tiles
        else:
            self.tiles = []
        self._generate_indices()
        self.revisit_time = revisit_time
        if tels is not None:
            self.tels = np.array(tels)
        else:
            # TODO: Currently hardcoded for 4 telescopes
            self.tels = np.array([1, 2, 3, 4])

        # TODO: Currently hardcoded for 4 telescopes
        tel_mask = np.zeros(4, dtype=bool)
        for tel in self.tels:
            tel_mask[tel - 1] = True
        self.tel_mask = tel_mask

        self.grid = grid

    def add_tiles(self, tile: Union[str, list[str]]):
        """
        Add a tile to the survey. Can be a single tile name (e.g. T0993) or a list of
        tile names.
        :param tile: Tile or list of tiles to add to the survey.
        """
        if isinstance(tile, list):
            self.tiles.extend(tile)
        else:
            self.tiles.append(tile)
        self._remove_duplicates()
        self._generate_indices()

    def remove_tiles(self, tile: Union[str, list[str]]):
        """
        Remove a tile from the survey. Can be a single tile name (e.g. T0993) or a list of
        tile names.
        :param tile: Tile or list of tiles to remove from the survey.
        """
        if isinstance(tile, str):
            tile = [tile]
        for t in tile:
            if t in self.tiles:
                self.tiles.remove(t)
        self._generate_indices()

    def load_tiles(self, filename: str):
        """
        Load tiles from a text file. Each line in the file should contain a single tile name.
        :param filename: Path to the text file containing tile names.
        """
        with open(filename, "r") as f:
            for line in f:
                tile = line.strip()
                if tile:
                    self.tiles.append(tile)
        self._remove_duplicates()
        self._generate_indices()

    def get_tiles(self):
        """
        Get the list of tiles in the survey.
        """
        return self.tiles

    def _remove_duplicates(self):
        """
        Remove duplicate tiles from the survey.
        """
        self.tiles = list(set(self.tiles))
        self.tiles.sort()

    def _generate_indices(self):
        """
        Generate indices for the tiles in the survey.
        """
        self.indices = np.array([int(tile[1:]) - 1 for tile in self.tiles])


class HEATS(Survey):
    """
    Convenience for generating survey for GOTO-HEATS.
    """

    def __init__(
        self,
        name: str = "HEATS Survey",
        revisit_time: u.Quantity[u.day] = 1 * u.day,
        tels: Union[None, np.ndarray, list] = None,
        grid: SkyGrid = SkyGrid.from_name("GOTO"),
    ):
        super().__init__(
            name=name, tiles=[], revisit_time=revisit_time, tels=tels, grid=grid
        )

    def generate_colds(self) -> Survey:
        """
        Generate a complementary survey for the COLD tiles.
        :return: Survey object containing the COLD tiles.
        """
        cold_tiles = [
            f"T{str(i).zfill(4)}"
            for i in range(self.grid.ntiles)
            if i not in self.indices
        ]
        cold_survey = Survey(
            name="COLD Survey",
            tiles=cold_tiles,
            revisit_time=self.revisit_time,
            tels=self.tels,
            grid=self.grid,
        )
        return cold_survey
