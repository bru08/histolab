# encoding: utf-8

# ------------------------------------------------------------------------
# Copyright 2020 All Histolab Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import csv
import os
from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from .exceptions import LevelError
from .scorer import Scorer
from .slide import Slide
from .tile import Tile
from .types import CoordinatePair
from .util import (
    lru_cache,
    region_coordinates,
    regions_from_binary_mask,
    scale_coordinates,
)

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Tiler(Protocol):
    """General tiler object"""

    level: int
    tile_size: int

    @lru_cache(maxsize=100)
    def box_mask(self, slide: Slide) -> np.ndarray:
        """Return binary mask, at thumbnail level, of the box for tiles extraction.

        The mask pixels set to True correspond to the tissue box.

        Parameters
        ----------
        slide : Slide
            The Slide from which to extract the extraction mask

        Returns
        -------
        np.ndarray
            Extraction mask at thumbnail level
        """

        return slide.biggest_tissue_box_mask

    @abstractmethod
    def extract(self, slide: Slide):
        raise NotImplementedError

    # ------- implementation helpers -------

    def _tile_filename(
        self, tile_wsi_coords: CoordinatePair, tiles_counter: int, ind_grid: int = None
    ) -> str:
        """Return the tile filename according to its 0-level coordinates and a counter.

        Parameters
        ----------
        tile_wsi_coords : CoordinatePair
            0-level coordinates of the slide the tile has been extracted from.
        tiles_counter : int
            Counter of extracted tiles.
        ind_grid : int, optional
            If the tile is extracted from a grid arrangment, is the index of the extracted
            tile among all the possible grid-tiles 

        Returns
        -------
        str
            Tile filename, according to the format
            `{prefix}tile_{tiles_counter}_gridtile_{ind_grid}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}"
            "-{y_br_wsi}{suffix}`
        """

        x_ul_wsi, y_ul_wsi, x_br_wsi, y_br_wsi = tile_wsi_coords
        tile_filename = (
            f"{self.prefix}tile_{tiles_counter}_gridtile_{ind_grid}_level{self.level}_{x_ul_wsi}-{y_ul_wsi}"
            f"-{x_br_wsi}-{y_br_wsi}{self.suffix}"
        )

        return tile_filename


class GridTiler(Tiler):
    """Extractor of tiles arranged in a grid, at the given level, with the given size.

    Arguments
    ---------
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    partial: float, optional
        Fraction of the grid tiles that will be extracted choosing randomly from the available 
        tiles in the grid
    maximum: maximum number of tiles that will be extracted regardless the number of tiles
        computed with the partial parameter 
    reference_folder: str, optional 
        if not None this folder is checked, and if it contains tiles the new tiles extracted 
        will not include these tiles 
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        level: int = 0,
        check_tissue: bool = True,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
        partial: float = 1,
        maximum: int = 100,
        reference_folder: str = None 
    ):
        self.tile_size = tile_size
        self.level = level
        self.check_tissue = check_tissue
        self.pixel_overlap = pixel_overlap
        self.prefix = prefix
        self.suffix = suffix
        self.partial = partial
        self.maximum = maximum
        self.ref_fold= reference_folder

    def extract(self, slide: Slide):
        """Extract tiles arranged in a grid and save them to disk, following this
        filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        """

        if self.level not in slide.levels:
            raise LevelError(
                f"Level {self.level} not available. Number of available levels: "
                f"{len(slide.levels)}"
            )
        if not (0 <= self.partial <=1 ):
             raise ValueError(f"The partial parameter must be between 0 and 1, current value: {self.partial}")
        
        #if self.maximum > self.tissue_mask_tiles_count(slide) and self.partial!=1:
        #     raise ValueError(f"The maximum number of tiles in output, {self.maximum}, is greater than the maximum number of grid tiles,filtered by the mask filter {self.tissue_mask_tiles_count(slide)}.")
        
        if self.partial == 1:
            grid_tiles = self._grid_all_tiles_generator(slide)
        else:
            grid_tiles = self._grid_partial_tiles_generator(slide)

        tiles_counter = 0

        for tiles_counter, (tile, tile_wsi_coords, ind) in enumerate(grid_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter, ind)
            #tile_filename = slide.name + "_" + tile_filename
            #full_tile_path = os.path.join(slide.processed_path, "tiles", tile_filename)
            full_tile_path = os.path.join(slide.processed_path, tile_filename)
            tile.save(full_tile_path)
            print(f"\t Tile {tiles_counter} saved: {tile_filename}")

        print(f"{tiles_counter} Grid Tiles have been saved.")

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise LevelError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    # ------- implementation helpers -------

    def _grid_coordinates_from_bbox_coordinates(
        self, bbox_coordinates: CoordinatePair, slide: Slide
    ) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within a tissue box.

        Parameters
        ----------
        bbox_coordinates: CoordinatePair
            Coordinates of the tissue box from which to calculate the coordinates.
        slide : Slide
            Slide from which to calculate the coordinates.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        tile_w_lvl, tile_h_lvl = self.tile_size

        n_tiles_row = self._n_tiles_row(bbox_coordinates)
        n_tiles_column = self._n_tiles_column(bbox_coordinates)

        for i in range(n_tiles_row):
            for j in range(n_tiles_column):
                x_ul_lvl = bbox_coordinates.x_ul + tile_w_lvl * j - self.pixel_overlap
                y_ul_lvl = bbox_coordinates.y_ul + tile_h_lvl * i - self.pixel_overlap

                x_ul_lvl = np.clip(x_ul_lvl, bbox_coordinates.x_ul, None)
                y_ul_lvl = np.clip(y_ul_lvl, bbox_coordinates.y_ul, None)

                x_br_lvl = x_ul_lvl + tile_w_lvl
                y_br_lvl = y_ul_lvl + tile_h_lvl

                tile_wsi_coords = scale_coordinates(
                    reference_coords=CoordinatePair(
                        x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl
                    ),
                    reference_size=slide.level_dimensions(level=self.level),
                    target_size=slide.level_dimensions(level=0),
                )
                yield tile_wsi_coords

    def _grid_coordinates_generator(self, slide: Slide) -> CoordinatePair:
        """Generate Coordinates at level 0 of grid tiles within the tissue.

        Parameters
        ----------
        slide : Slide
            Slide from which to calculate the coordinates. Needed to calculate the
            tissue area.

        Yields
        -------
        Iterator[CoordinatePair]
            Iterator of tiles' CoordinatePair
        """
        """
        box_mask = self.box_mask(slide)

        regions = regions_from_binary_mask(box_mask)
        # ----at the moment there is only one region----
        for region in regions:
            bbox_coordinates_thumb = region_coordinates(region)
            bbox_coordinates = scale_coordinates(
                bbox_coordinates_thumb,
                box_mask.shape[::-1],
                slide.level_dimensions(self.level),
            )
            print(bbox_coordinates_thumb)
        yield from self._grid_coordinates_from_bbox_coordinates(
                bbox_coordinates, slide
            )
        """
        tissue_mask = slide.tissue_mask
        bbox_coordinates = scale_coordinates(
                CoordinatePair(0, 0, *tissue_mask.shape[::-1]),
                tissue_mask.shape[::-1],
                slide.level_dimensions(self.level),
            )
        yield from self._grid_coordinates_from_bbox_coordinates(
            bbox_coordinates, slide
        )
        

        



    def gross_tiles_count(self, slide: Slide) -> int:
        """
        Count the element in the grid coordinate generator
        
        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Return
        -------
        n_tiles
            maximum possible number of tiles to be extracted from the slide
        """
        return sum(1 for _ in self._grid_coordinates_generator(slide))

    def net_tiles_count(self, slide: Slide) -> int:
        """
        Count the element in the grid tiles generator
        (slower as it may have to apply operations on images)
        
        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Return
        -------
        n_tiles
            maximum possible number of tiles to be extracted from the slide
        """
        return sum(1 for _ in self._grid_tiles_generator(slide))

    def tissue_mask_tiles_count(self, slide: Slide) -> int:
        """
        Count the element in the grid coordinate generator
        
        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Return
        -------
        n_tiles
            maximum possible number of tiles to be extracted from the slide
        """
        tissue_mask = slide.tissue_mask
        counter = 0
        for x in self._grid_coordinates_generator(slide):
            x = scale_coordinates(
                x,
                slide.dimensions,
                tissue_mask.shape[:2][::-1]
            )
            if np.mean(tissue_mask[x.y_ul:x.y_br, x.x_ul:x.x_br]) > .8:
                counter += 1
        return counter

    def _grid_all_tiles_generator(self, slide: Slide, check: bool = None) -> Tuple[Tile, CoordinatePair, int]: 
        """Generator of the possible tiles arranged in a grid

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        check : bool
            If it is set false the self.has_enough_tissue method is ignored. Useful if we 
            want to plot the extraction_plot for checking which tiles will be extracted without 
            filtering the not valid tiles
        Yields
        -------
        Tile
            Extracted tile
        CoordinatePair
            Coordinates of the slide at level 0 from which the tile has been extracted
        int
            index of the tile returned 
        """

        grid_coordinates_generator = self._grid_coordinates_generator(slide)
        for ind,coords in enumerate(grid_coordinates_generator):

            try:
                tile = slide.extract_tile(coords, self.level)
            except ValueError:
                continue
            if check == None or check==True:
                if not self.check_tissue or tile.has_enough_tissue():
                    yield tile, coords, ind
            else:
                yield tile, coords, ind

    def already_extracted(self):
        """
        Returns the grid index tiles already estracted in the self.ref_fold folder

        Parameters
        ----------
        self 

        Returns
        -------
        numpy.array
            index tiles already estracted in the self.ref_fold folder
        """
        if not os.path.isdir(self.ref_fold):
            raise ValueError(f"The reference folder {self.ref_fold} does not exist")

        tiles_index=[int(x.split("_")[4]) for x in os.listdir(self.ref_fold) if os.path.splitext(x)[1] == self.suffix]
        if len(tiles_index)==0:
            print(f"Warning: the reference folder {self.ref_fold} is empty" )
        """
        present=list()
        for name in tiles_name:
            present.append(int(name.split("_")[3]))
        """
        return np.asarray(tiles_index)

    def tissue_mask_tiles_index(self,slide: Slide):
        """
        Compute the possible tiles index of the tiles arranged in a grid, and that are 
        not excluded by the mask filter 

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Returns
        -------
        indexes
            List of all the possible indexes of tiles that can be extracted using _grid_partial_tiles_generator
            
        """
        tissue_mask = slide.tissue_mask
        li_ind=[]
        for i,x in enumerate(self._grid_coordinates_generator(slide)):
            x = scale_coordinates(
                x,
                slide.dimensions,
                tissue_mask.shape[:2][::-1]
            )
            if np.mean(tissue_mask[x.y_ul:x.y_br, x.x_ul:x.x_br]) > .8:
                li_ind.append(i)
        
        return np.asarray(li_ind)


    def partial_grid_ntiles(self, slide: Slide):
        """Return the target number of tiles in output when self.partial<1

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles

        Return
        -------
        int
            Return the target number of tiles in output when self.partial<1
        """
        if int(self.tissue_mask_tiles_count(slide) * self.partial) < self.maximum:
            n_tiles_target = int(self.tissue_mask_tiles_count(slide) * self.partial)
        else:
            n_tiles_target = self.maximum
        return n_tiles_target

    def _grid_partial_tiles_generator(self, slide: Slide, check: bool = None) -> Tuple[Tile, CoordinatePair, int]: 
        """
        Generator of a fraction of all the possible valid tiles arranged in a grid. 
        The fraction is defined by 'self.partial', and the tiles are chosen 
        randomly among the possible tiles. 
        Whether the number of expected tiles would be greater than self.maximum, the target 
        number of tiles will be self.maximum. 
        Let's note that the tiles could be in minor number than what expected if the number 
        of valid tiles are minor then the target number of tiles. 

        
        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        check : bool
            If it is set false the self.has_enough_tissue method is ignored. Useful if we 
            want to plot the extraction_plot for checking which tiles will be extracted without 
            filtering the not valid tiles

        Yields
        -------
        Tile
            Extracted tile
        CoordinatePair
            Coordinates of the slide at level 0 from which the tile has been extracted
        int
            index of the tile returned, referred to all the possible grid tiles in the box-tissue
        """
            
        #initialize the possible index among the ones counted in the mask
        possible_ind = self.tissue_mask_tiles_index(slide)
        #print(len(possible_ind))
        if self.ref_fold is not None:
            to_exclude = self.already_extracted()
            #print(to_exclude, type(to_exclude))
            possible_ind = np.setdiff1d(possible_ind, to_exclude)
        #print(len(possible_ind))
        n_t_target = self.partial_grid_ntiles(slide)       
        n_t_out= 0
        while possible_ind.size > 0 and n_t_out < n_t_target: 
            n_ext = n_t_target - n_t_out

            if possible_ind.size > n_ext:
                random_ind = np.random.choice(possible_ind, n_ext, replace=False)
            else:
                #if I don't have anymore enough possible index respect to the expected number
                #of tiles, it doesn't make sense the random extraction:
                random_ind = possible_ind

            for ind,coords in enumerate(self._grid_coordinates_generator(slide)):
                if n_t_out >= n_t_target:
                    break
                if ind in random_ind:
                    try:
                        tile = slide.extract_tile(coords, self.level)
                    except ValueError:
                        continue
                    
                    if check == None or check==True:
                        if not self.check_tissue or tile.has_enough_tissue():
                            n_t_out=n_t_out + 1   
                            yield tile, coords, ind
                    else:
                        n_t_out=n_t_out + 1   
                        yield tile, coords, ind
                    """
                    
                    if not self.check_tissue or tile.has_enough_tissue():
                        n_t_out=n_t_out + 1                           
                        yield tile, coords, ind   
                    """
                #update the possible index, erasing what I have already tried                             
                possible_ind = np.setdiff1d(possible_ind,random_ind)
        if n_t_out < n_t_target:
            print(f"Warning: the number of output tiles is {n_t_out}, minor than the target tiles number {n_t_target}" )
                    
                
    def _n_tiles_column(self, bbox_coordinates: CoordinatePair) -> int:
        """Return the number of tiles which can be extracted in a column.

        Parameters
        ----------
        bbox_coordinates : CoordinatePair
            Coordinates of the tissue box

        Returns
        -------
        int
            Number of tiles which can be extracted in a column.
        """
        return (bbox_coordinates.y_br - bbox_coordinates.y_ul) // (
            self.tile_size[1] - self.pixel_overlap
        )

    def _n_tiles_row(self, bbox_coordinates: CoordinatePair) -> int:
        """Return the number of tiles which can be extracted in a row.

        Parameters
        ----------
        bbox_coordinates : CoordinatePair
            Coordinates of the tissue box

        Returns
        -------
        int
            Number of tiles which can be extracted in a row.
        """
        return (bbox_coordinates.x_br - bbox_coordinates.x_ul) // (
            self.tile_size[0] - self.pixel_overlap
        )

    def extraction_plot(self, slide: Slide, check_t: bool = None):
        """Generate diagnostic plot to visualize tiles to be extracted
        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        check_t : bool
            If it is set false the self.has_enough_tissue method in the generator is ignored. 
            Useful if we want to plot the extraction_plot for checking which tiles will be 
            extracted without filtering the not valid tiles
        """
        if self.level not in slide.levels:
            raise LevelError(
                f"Level {self.level} not available. Number of available levels: "
                f"{len(slide.levels)}"
            )
        if not (0 <= self.partial <=1 ):
             raise ValueError(f"The partial parameter must be between 0 and 1, current value: {self.partial}")
        
        #if (self.maximum > self.tissue_mask_tiles_count(slide)) and (self.partial!=1):
        #    raise ValueError(f"The maximum number of tiles in output, {self.maximum}, is greater than the maximum number of grid tiles,filtered by the mask filter {self.tissue_mask_tiles_count(slide)}.")
        
        if self.partial == 1:
            grid_tiles = self._grid_all_tiles_generator(slide, check=check_t)
        else:
            grid_tiles = self._grid_partial_tiles_generator(slide, check=check_t )
 
            
        thumb = np.copy(slide._resample()[1])
        for (_, coord, _) in grid_tiles:
            x = scale_coordinates(
                coord,
                slide.dimensions,
                (thumb.shape[1],
                thumb.shape[0])
            )
            l_width = 3 # max(1, ((x[2]-x[1])//(10)))
            border_color =  (0,255,0)  
            #("CoordinatePair", ("x_ul", "y_ul", "x_br", "y_br")) 
            thumb[x[1]:x[1]+l_width,x[0]:x[2],:] = border_color
            # # top margin
            thumb[x[1]:x[3],x[0]-l_width:x[0],:] = border_color
            # #right margin
            thumb[x[3]:x[3]+l_width,x[0]:x[2], :] = border_color
            # # bottom margin
            thumb[x[1]:x[3],x[2]-l_width:x[2],:] = border_color
        return thumb
    

class RandomTiler(Tiler):
    """Extractor of random tiles from a Slide, at the given level, with the given size.

    Arguments
    ---------
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    n_tiles : int
        Maximum number of tiles to extract.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    seed : int, optional
        Seed for RandomState. Must be convertible to 32 bit unsigned integers. Default
        is 7.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    max_iter : int, optional
        Maximum number of iterations performed when searching for eligible (if
        ``check_tissue=True``) tiles. Must be grater than or equal to ``n_tiles``.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        n_tiles: int,
        level: int = 0,
        seed: int = 7,
        check_tissue: bool = True,
        prefix: str = "",
        suffix: str = ".png",
        max_iter: int = int(1e4),
    ):

        super().__init__()

        self.tile_size = tile_size
        self.n_tiles = n_tiles
        self.max_iter = max_iter
        self.level = level
        self.seed = seed
        self.check_tissue = check_tissue
        self.prefix = prefix
        self.suffix = suffix

    def extract(self, slide: Slide):
        """Extract random tiles and save them to disk, following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        """

        np.random.seed(self.seed)

        random_tiles = self._random_tiles_generator(slide)

        tiles_counter = 0
        for tiles_counter, (tile, tile_wsi_coords) in enumerate(random_tiles):
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            tile.save(os.path.join(slide.processed_path, "tiles", tile_filename))

            print(f"\t Tile {tiles_counter} saved: {tile_filename}")
        print(f"{tiles_counter+1} Random Tiles have been saved.")

    @property
    def level(self) -> int:
        return self._valid_level

    @level.setter
    def level(self, level_: int):
        if level_ < 0:
            raise LevelError(f"Level cannot be negative ({level_})")
        self._valid_level = level_

    @property
    def max_iter(self) -> int:
        return self._valid_max_iter

    @max_iter.setter
    def max_iter(self, max_iter_: int = int(1e4)):
        if max_iter_ < self.n_tiles:
            raise ValueError(
                f"The maximum number of iterations ({max_iter_}) must be grater than or"
                f" equal to the maximum number of tiles ({self.n_tiles})."
            )
        self._valid_max_iter = max_iter_

    @property
    def tile_size(self) -> Tuple[int, int]:
        return self._valid_tile_size

    @tile_size.setter
    def tile_size(self, tile_size_: Tuple[int, int]):
        if tile_size_[0] < 1 or tile_size_[1] < 1:
            raise ValueError(f"Tile size must be greater than 0 ({tile_size_})")
        self._valid_tile_size = tile_size_

    # ------- implementation helpers -------

    def _random_tile_coordinates(self, slide: Slide) -> CoordinatePair:
        """Return 0-level Coordinates of a tile picked at random within the box.

        Parameters
        ----------
        slide : Slide
            Slide from which calculate the coordinates. Needed to calculate the box.

        Returns
        -------
        CoordinatePair
            Random tile Coordinates at level 0
        """
        box_mask = self.box_mask(slide)
        tile_w_lvl, tile_h_lvl = self.tile_size

        x_ul_lvl = np.random.choice(np.where(box_mask)[1])
        y_ul_lvl = np.random.choice(np.where(box_mask)[0])

        # Scale tile dimensions to thumbnail dimensions
        tile_w_thumb = (
            tile_w_lvl * box_mask.shape[1] / slide.level_dimensions(self.level)[0]
        )
        tile_h_thumb = (
            tile_h_lvl * box_mask.shape[0] / slide.level_dimensions(self.level)[1]
        )

        x_br_lvl = x_ul_lvl + tile_w_thumb
        y_br_lvl = y_ul_lvl + tile_h_thumb

        tile_wsi_coords = scale_coordinates(
            reference_coords=CoordinatePair(x_ul_lvl, y_ul_lvl, x_br_lvl, y_br_lvl),
            reference_size=box_mask.shape[::-1],
            target_size=slide.dimensions,
        )

        return tile_wsi_coords

    def _random_tiles_generator(self, slide: Slide) -> Tuple[Tile, CoordinatePair]:
        """Generate Random Tiles within a slide box.

        Stops if:
        * the number of extracted tiles is equal to ``n_tiles`` OR
        * the maximum number of iterations ``max_iter`` is reached

        Parameters
        ----------
        slide : Slide
            The Whole Slide Image from which to extract the tiles.

        Yields
        ------
        tile : Tile
            The extracted Tile
        coords : CoordinatePair
            The level-0 coordinates of the extracted tile
        """

        iteration = valid_tile_counter = 0

        while True:

            tile_wsi_coords = self._random_tile_coordinates(slide)
            try:
                tile = slide.extract_tile(tile_wsi_coords, self.level)
            except ValueError:
                iteration -= 1
                continue

            if not self.check_tissue or tile.has_enough_tissue():
                yield tile, tile_wsi_coords
                valid_tile_counter += 1
            iteration += 1

            if self.max_iter and iteration >= self.max_iter:
                break

            if valid_tile_counter >= self.n_tiles:
                break


class ScoreTiler(GridTiler):
    """Extractor of tiles arranged in a grid according to a scoring function.

    The extraction procedure is the same as the ``GridTiler`` extractor, but only the
    first ``n_tiles`` tiles with the highest score are saved.

    Arguments
    ---------
    scorer : Scorer
        Scoring function used to score the tiles.
    tile_size : Tuple[int, int]
        (width, height) of the extracted tiles.
    n_tiles : int, optional
        The number of tiles to be saved. Default is 0, which means that all the tiles
        will be saved (same exact behaviour of a GridTiler). Cannot be negative.
    level : int, optional
        Level from which extract the tiles. Default is 0.
    check_tissue : bool, optional
        Whether to check if the tile has enough tissue to be saved. Default is True.
    pixel_overlap : int, optional
       Number of overlapping pixels (for both height and width) between two adjacent
       tiles. If negative, two adjacent tiles will be strided by the absolute value of
       ``pixel_overlap``. Default is 0.
    prefix : str, optional
        Prefix to be added to the tile filename. Default is an empty string.
    suffix : str, optional
        Suffix to be added to the tile filename. Default is '.png'
    """

    def __init__(
        self,
        scorer: Scorer,
        tile_size: Tuple[int, int],
        n_tiles: int = 0,
        level: int = 0,
        check_tissue: bool = True,
        pixel_overlap: int = 0,
        prefix: str = "",
        suffix: str = ".png",
    ):
        self.scorer = scorer
        self.n_tiles = n_tiles

        super().__init__(tile_size, level, check_tissue, pixel_overlap, prefix, suffix)

    def extract(self, slide: Slide, report_path: str = None):
        """Extract grid tiles and save them to disk, according to a scoring function and
        following this filename pattern:
        `{prefix}tile_{tiles_counter}_level{level}_{x_ul_wsi}-{y_ul_wsi}-{x_br_wsi}-{y_br_wsi}{suffix}`

        Save a CSV report file with the saved tiles and the associated score.

        Parameters
        ----------
        slide : Slide
            Slide from which to extract the tiles
        report_path : str, optional
            Path to the CSV report. If None, no report will be saved
        """
        highest_score_tiles, highest_scaled_score_tiles = self._highest_score_tiles(
            slide
        )

        tiles_counter = 0
        filenames = []

        for tiles_counter, (score, tile_wsi_coords) in enumerate(highest_score_tiles):
            tile = slide.extract_tile(tile_wsi_coords, self.level)
            tile_filename = self._tile_filename(tile_wsi_coords, tiles_counter)
            #tile.save(os.path.join(slide.processed_path, "tiles", tile_filename))
            tile.save(os.path.join(slide.processed_path, tile_filename))
            filenames.append(tile_filename)
            print(f"\t Tile {tiles_counter} - score: {score} saved: {tile_filename}")

        if report_path:
            self._save_report(
                report_path, highest_score_tiles, highest_scaled_score_tiles, filenames
            )

        print(f"{tiles_counter+1} Grid Tiles have been saved.")

    # ------- implementation helpers -------

    def _highest_score_tiles(self, slide: Slide) -> List[Tuple[float, CoordinatePair]]:
        """Calculate the tiles with the highest scores and their extraction coordinates.

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.

        Returns
        -------
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for the
            tiles with the highest score. Each tuple represents a tile.
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the scaled score between 0 and 1 and the
            extraction coordinates for the tiles with the highest score. Each tuple
            represents a tile.

        Raises
        ------
        ValueError
            If ``n_tiles`` is negative.
        """
        all_scores = self._scores(slide)
        scaled_scores = self._scale_scores(all_scores)

        sorted_tiles_by_score = sorted(all_scores, key=lambda x: x[0], reverse=True)
        sorted_tiles_by_scaled_score = sorted(
            scaled_scores, key=lambda x: x[0], reverse=True
        )
        if self.n_tiles < 0:
            raise ValueError(f"'n_tiles' cannot be negative ({self.n_tiles})")

        if self.n_tiles > 0:
            highest_score_tiles = sorted_tiles_by_score[: self.n_tiles]
            highest_scaled_score_tiles = sorted_tiles_by_scaled_score[: self.n_tiles]
        else:
            highest_score_tiles = sorted_tiles_by_score
            highest_scaled_score_tiles = sorted_tiles_by_scaled_score

        return highest_score_tiles, highest_scaled_score_tiles

    def _save_report(
        self,
        report_path: str,
        highest_score_tiles: List[Tuple[float, CoordinatePair]],
        highest_scaled_score_tiles: List[Tuple[float, CoordinatePair]],
        filenames: List[str],
    ) -> None:
        """Save to ``filename`` the report of the saved tiles with the associated score.

        The CSV file

        Parameters
        ----------
        report_path : str
            Path to the report
        highest_score_tiles : List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for the
            tiles with the highest score. Each tuple represents a tile.
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the scaled score between 0 and 1 and the
            extraction coordinates for the tiles with the highest score. Each tuple
            represents a tile.
        filenames : List[str]
            List of the tiles' filename
        """

        header = ["filename", "score", "scaled_score"]
        rows = [
            dict(zip(header, values))
            for values in zip(
                filenames,
                np.array(highest_score_tiles)[:, 0],
                np.array(highest_scaled_score_tiles)[:, 0],
            )
        ]

        with open(report_path, "w+", newline="") as filename:
            writer = csv.DictWriter(
                filename, fieldnames=header, lineterminator=os.linesep
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _scale_scores(
        scores: List[Tuple[float, CoordinatePair]]
    ) -> List[Tuple[float, CoordinatePair]]:
        """Scale scores between 0 and 1.

        Parameters
        ----------
        scores : List[Tuple[float, CoordinatePair]]
            Scores to be scaled

        Returns
        -------
        List[Tuple[float, CoordinatePair]])
            Scaled scores
        """
        scores_ = np.array(scores)[:, 0]
        coords = np.array(scores)[:, 1]
        scores_scaled = (scores_ - np.min(scores_)) / (
            np.max(scores_) - np.min(scores_)
        )

        return list(zip(scores_scaled, coords))

    def _scores(self, slide: Slide) -> List[Tuple[float, CoordinatePair]]:
        """Calculate the scores for all the tiles extracted from the ``slide``.

        Parameters
        ----------
        slide : Slide
            The slide to extract the tiles from.

        Returns
        -------
        List[Tuple[float, CoordinatePair]]
            List of tuples containing the score and the extraction coordinates for each
            tile. Each tuple represents a tile.
        """
        if next(self._grid_tiles_generator(slide), None) is None:
            raise RuntimeError(
                "No tiles have been generated. This could happen if `check_tissue=True`"
            )

        grid_tiles = self._grid_tiles_generator(slide)
        scores = []

        for tile, tile_wsi_coords in grid_tiles:
            score = self.scorer(tile)
            scores.append((score, tile_wsi_coords))

        return scores
