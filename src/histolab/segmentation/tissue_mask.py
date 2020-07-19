import os, sys
from typing import Any, List, Union

import skimage
import numpy as np
import PIL
from sklearn import cluster
import hdbscan

from ..filters import image_filters as imf
from ..filters import morphological_filters as mof


class HSDCFilter:
    """
    HSDCFilter: Hue-Saturation Density cluster filter.
    Take the WSI at low resolution, large scale objects were artifacts are
    easily recognisable and separable from the main tissue region.
    Transform the image in HSV space, to obtain the H-S coordinates.
    Take the first (eventually the second) density based cluster in terms of size,
    in the H-S space.
    Use pixel from the eligible clusters to build a prototype mask, which is
    morpohlogically refined and upscaled to the desired resolution.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, max_low_side=200, clu_method="dbscan"):
      
        self.max_low_side = max_low_side
        self.clu_method = clu_method.lower()

        self.low_res_ = None
        self.low_res_msk_ = None
        self.ni = None
        self.upscaled_mask_ = None
        self.hs_ = None
        self.hs_colors_ = None
        self.ok_cluster_masks_ = []
        self.ok_clusters_ = []

    def fit_predict(self, image):
        self.ni = self.max_low_side / max(image.size)
        new_w = int(image.size[0] * self.ni)
        new_h = int(image.size[1] * self.ni)
        self.low_res_ = image.resize((new_w, new_h), PIL.Image.BILINEAR)
        self.low_res_msk_ = self.extract_tissue(self.low_res_)
        self.upscaled_mask_ = self.mask_reconstruction(image, self.low_res_msk_) > 0.5
        self.refined_mask_ = self.refine_mask(self.upscaled_mask_)
        return self.refined_mask_

    @staticmethod
    def refine_mask(mask):
        msk = np.copy(mask)
        
        filters = Compose([
            RemoveSmallHolesRelative(),
            RemoveSmallObjectsRelative(),
        ])

        return filters(msk)
    
    @staticmethod
    def mask_reconstruction(image, ds_mask):
        return skimage.transform.resize(
            ds_mask, (image.size[1], image.size[0]), anti_aliasing=False)
    

    def extract_tissue(self, image: PIL.Image.Image) -> np.array:
        """
        Take as input an RGB image as a 3d numpy array.
        It project the data on HSV colorspace and filter based on HS values or
        perform custering to find the bigger /denser structures which should belong
        to relevant areas: i.e. tissue.
        Parameter
        ---------
        image: float np.ndarray (w x h x ch)
        ax: to output a diagnostic plot using matplotlib
        **kwargs: other arguments for clustering methods (HDBSCAN)
        Return
        ------
        img_res: np.ndarray: RGB image masked from the background
        mask_tot: np.ndarray w x h
            points that are ok

        """
        fg_mask = self.bg_segmentation_mask(image)
        hsv_img = skimage.img_as_float(np.array(image.convert("HSV")))
        image = skimage.img_as_float(np.array(image))
        hsv_pix = hsv_img[fg_mask, :]
        self.hs_colors_ = image[fg_mask, :]
        self.hs_ = hsv_pix[:, :2]

        if self.clu_method == "hdbscan":
            self.clusterer = hdbscan.HDBSCAN(allow_single_cluster=True)
        elif self.clu_method == "dbscan":
            self.clusterer = cluster.DBSCAN(eps=0.005, min_samples=10)
        else:
            raise Exception(f"Clustering method {self.clu_method} not valid.")

        self.c_labels = self.clusterer.fit_predict(self.hs_)

        lab_u, cnts = np.unique(self.c_labels, return_counts=True)
        max_size_clu = np.argmax(cnts)

        max_clu = lab_u[max_size_clu]
        mask_max_clu = self.c_labels == max_clu
        self.ok_cluster_masks_.append(mask_max_clu)
        self.ok_clusters_.append(max_clu)
        img_res = np.copy(image)

        mask_tot = fg_mask
        mask_tot[mask_tot == True] = mask_max_clu

        img_res[~mask_tot, :] = 0

        assert self.hs_.shape[0] == self.hs_colors_.shape[0]

        return mask_tot
    
    @staticmethod
    def bg_segmentation_mask(image: PIL.Image.Image) -> np.array:
        filters = Compose(
            [
                RgbToGrayscale(),
                OtsuThreshold(),
                RemoveSmallObjectsRelative(),
                RemoveSmallHolesRelative(),
            ]
        )
        return filters(image)

    def plot_colors_hs(self, ax=None):
        """
        Provide diagnostic plot for color in the hue-saturation space.
        Optionally a matplotlib ax can be provided to plot on.

        Parameter
        ---------
        ax : matplotlib ax
        """
        if ax is None: fig, ax = plt.subplots()

        ax.scatter(*self.hs_.T, color=self.hs_colors_, s=1)
        ax.set_ylim((-.1,1.1))
        ax.set_xlim((-.1,1.1))
        ax.set_xlabel("H")
        ax.set_ylabel("S")
        ax.set_title("H-S scatterplot with real colors")
        if ax is None: plt.show()

    def plot_cluster_hs(self, ax=None):
        """
        Provide a diagnostic plot for the selected clusters from points
        in the H-S space.

        Parameter
        ---------
        ax : matplotlib ax
        """
        if ax is None: fig, ax = plt.subplots()
        # all acceptable points
        msk = np.repeat(False, self.ok_cluster_masks_[0].shape[0])
        for elem in self.ok_cluster_masks_:
            msk = (msk) | (elem)
        ax.scatter(
            *self.hs_[msk, :].T, label="Tissue clusters",
            s=1,c="orange", alpha=0.7)
        ax.scatter(
            *self.hs_[~msk, :].T, label="Other",
            s=1,c="blue", alpha=0.7)
        ax.set_title("Diagnostic plot: selected/found clusters")
        ax.set_ylim((- .1, 1.1))
        ax.set_xlim((- .1, 1.1))
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")

        if ax is None: plt.show()

    def __repr__(self) -> str:
        name = "{}\nClustering method: {}\nMax image size: {}".format(
            self.__class__.__name__, self.clu_method, self.max_low_side
        )
        return name

    def __call__(self, img: PIL.Image.Image, **kwargs) -> np.array:
        return self.fit_predict(img, **kwargs)


def main_tissue_areas_mask_filters() -> imf.Compose:
    """Return a filters composition to get a binary mask of the main tissue regions.

    Returns
    -------
    imf.Compose
        Filters composition
    """
    filters = imf.Compose(
        [
            imf.RgbToGrayscale(),
            imf.OtsuThreshold(),
            mof.BinaryDilation(),
            mof.RemoveSmallHoles(),
            mof.RemoveSmallObjects(),
        ]
    )
    return filters

def strict_fg_mask_filters() -> imf.Compose:
    """Return a filters composition to get a binary mask for the foreground

    Returns
    -------
    imf.Compose
        Filters composition
    """
    filters = imf.Compose(
        [
            imf.RgbToGrayscale(),
            imf.OtsuThreshold(),
            mof.RemoveSmallObjectsRelative(),
            mof.RemoveSmallHolesRelative(),
        ]
    )
    return filters

def tissue_texture_filters() -> imf.Compose:
    """Return a filters composition to get a binary mask for the foreground

    Returns
    -------
    imf.Compose
        Filters composition
    """
    filters = imf.Compose(
        [
            imf.RgbToGrayscale(),
            imf.CannyEdges(0,None,None),
            mof.BinaryDilation(3),
            mof.BinaryErosion(6),
            mof.BinaryClosing(3, 5),
            mof.RemoveSmallObjectsRelative(1e-3),
            mof.RemoveSmallHolesRelative(1e-3),
        ]
    )
    return filters
