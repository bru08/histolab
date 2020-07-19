import numpy as np
import cv2
import skimage
import PIL
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

from ..filters import image_filters as imf
from ..filters import morphological_filters as mof


class TileNuclei:

    def __init__(
        self, nuclei_th: int = 5, selem_radius: int = 2,
        min_area: int = 20, max_area: int = 100,
        ):
            # TODO add dependency on magnification power linking expected nuclei size and zoom
            """Define a tile interesting if there are enough nuclei

            Parameter
            ---------
            nuclei_th : int
                Minimum number of nuclei to accept
            selem_radius: int
                Disk radius for structuring element used in the opening operation
                used to refine nuclei count
            min_area: int
                min number of pixel in a connected component to be included in
                nuclei count
            max_area: int
                max number of pixel in a connected component to be included in
                nuclei count

            Return
            ------
            enough_nuclei : bool
            """
            self.nuclei_th = nuclei_th
            self.selem_ = skimage.morphology.disk(selem_radius, dtype=np.uint8)
            self.max_area = max_area
            self.min_area = min_area
            self_nuclei_count_ = None

    def count_nuclei(self, image: PIL.Image.Image) -> int:
        """Count nuclei-like objects inside the given image

        Parameter
        ---------
        image: PIL.Image

        Return
        ------
        n_count: int
            number of nuclei detected
        """
        x = imf.RgbToGrayscale()(image)
        x = np.array(x) / 255
        x = x < 0.3
        x = cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, self.selem_)
        cc_stats = cv2.connectedComponentsWithStats(x)
        areas = [y[-1] for y in cc_stats[2]]
        n_count = np.sum((np.array(areas)<100) & (np.array(areas)>20)) > 10
        return n_count

    def nuclei_plot(self, image: PIL.Image.Image) -> int:
        x = imf.RgbToGrayscale()(image)
        x = np.array(x) / 255
        x = x < 0.3
        x = cv2.morphologyEx(x.astype(np.uint8), cv2.MORPH_OPEN, self.selem_)
        cc_stats = cv2.connectedComponentsWithStats(x)
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[1].imshow(cc_stats[1])

    def __call__(self, image: PIL.Image.Image) -> bool:
        return self.count_nuclei(image) > self.nuclei_th
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class FindTileMarker:
    
    def __init__(
        self, 
        hue1: float = 0.1, hue2: float = 0.85,
        sat_min: float = 0.05, sat_max: float = 0.5,
        bg_saturation_th=None
        ):
    
            self.h1 = hue1
            self.h2 = hue2
            self.min_sat = sat_min
            self.max_sat = sat_max
            self.bg_sat = bg_saturation_th if bg_saturation_th else self.min_sat

    def background_and_marker(self, image: PIL.Image.Image) -> Tuple[float, float]:
        hsv_array = np.array(image.convert("HSV")) / 255
        hue, sat = hsv_array[:, :, :2].reshape(-1, 2).T
        hue_condition = (hue < self.h1) | (hue > self.h2)
        sat_condition = (sat > self.min_sat) & (sat < self.max_sat)
        count_marker = np.sum(hue_condition & sat_condition)
        count_bg = np.sum(sat < self.bg_sat)
        self.marker_pc = 1e2 * count_marker / (hue.shape[0] - count_bg)
        self.bg_pc = 1e2 * count_bg / hue.shape[0]
        return self.marker_pc, self.bg_pc

    def polar_hs_marker(self, image: PIL.Image.Image):
        image = skimage.transform.rescale(np.array(image), (1/4,1/4,1))
        hsv_array = rgb2hsv(image)
        colors = image.reshape(-1,3)
        hue, sat = hsv_array[:, :, :2].reshape(-1, 2).T
        hue_angle = hue * 2 * np.pi
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(image)
        ax2 = fig.add_subplot(1,2,2, projection="polar")
        ax2.scatter(hue_angle, sat, color=colors, s=1)
        # angular sector
        a1 = 2 * np.pi * self.h1
        a2 = 2 * np.pi * self.h2
        a_space = (np.linspace(self.h2, 1+self.h1) % 1) * 2 * np.pi
        l1 = np.ones_like(a_space) * self.min_sat
        l2 = np.ones_like(a_space) * self.max_sat
        lr = np.ones_like(a_space) * self.bg_sat
        ax2.plot(a_space, l1, c="orange")
        ax2.plot(a_space, l2, c="orange")
        ax2.plot([a_space[0]] * 2, [self.min_sat, self.max_sat], c="orange")
        ax2.plot([a_space[-1]] * 2, [self.min_sat, self.max_sat], c="orange")
        # ignored central circular area
        ac_space = np.linspace(0, 1) * 2 * np.pi
        ax2.fill_between(ac_space, np.zeros_like(lr), lr, alpha=0.3, facecolor="red")
        ax1.set_title(
            f"Marker: {self.marker_pc:.2f}%, Background:{self.bg_pc:.2f}%")
    def __call__(self, image: PIL.Image.Image) -> float:
        marker_pc, bg_pc = self.background_and_marker(image)
        return marker_pc / bg_pc
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

