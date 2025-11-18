"""
Tools to support scheduling in the GOTO simulator. Allowing for creation of surveys.
"""

from typing import Union

import numpy as np
import astropy.units as u
from gototile.grid import SkyGrid
from importlib import resources

import logging

from goto_sim.utils import load_tilelist

logger = logging.getLogger(__name__)


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
        self.grid = grid

        if tiles is not None:
            self.tiles = tiles
        else:
            self.tiles = []

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
        self._verify_tiles()
        self._generate_indices()

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
        self._verify_tiles()
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
            logger.warning(f"Tile {t} not in survey; cannot remove.")
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
        self._verify_tiles()
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
        self.tile_nums = np.array([int(tile[1:]) for tile in self.tiles])
        self.indices = self.tile_nums - 1

    def _verify_tiles(self):
        """
        Verify that all tiles in the survey are valid tile names in the grid.
        """
        valid_tiles = set(self.grid.tilenames)
        invalid_tiles = [tile for tile in self.tiles if tile not in valid_tiles]
        if invalid_tiles:
            raise ValueError(f"Invalid tile names in survey: {invalid_tiles}")


class HEATSurvey(Survey):
    """
    Convenience for generating survey for GOTO-HEATS and COLD surveys.
    """

    def __init__(
        self,
        name: str = "HEATS Survey",
        revisit_time: u.Quantity[u.day] = 1 * u.day,
        tels: Union[None, np.ndarray, list] = None,
        tiles: Union[None, list[str]] = None,
        grid: SkyGrid = SkyGrid.from_name("GOTO"),
    ):
        if tels is None:
            tels = [1, 3]
        if tiles is None:
            with resources.as_file(
                resources.files("goto_sim").joinpath("data/tile_lists/HEATS.txt")
            ) as p:
                tiles = load_tilelist(str(p))
        super().__init__(
            name=name, tiles=tiles, revisit_time=revisit_time, tels=tels, grid=grid
        )

    def generate_colds(self) -> Survey:
        """
        Generate a complementary survey for the COLD tiles.
        :return: Survey object containing the COLD tiles.
        """
        cold_tiles = [
            f"T{str(i + 1).zfill(4)}"
            for i in range(self.grid.ntiles)
            if i not in self.indices
        ]
        cold_tels = [tel for tel in range(1, 5) if tel not in self.tels]
        cold_survey = Survey(
            name="COLD Survey",
            tiles=cold_tiles,
            revisit_time=self.revisit_time,
            tels=cold_tels,
            grid=self.grid,
        )
        return cold_survey
