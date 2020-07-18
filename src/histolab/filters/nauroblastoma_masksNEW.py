
import numpy as np

from matplotlib import pylab as plt


from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
from histomicstk.preprocessing.color_normalization.\
    deconvolution_based_normalization import deconvolution_based_normalization
from histomicstk.preprocessing.color_deconvolution.\
    color_deconvolution import color_deconvolution_routine, stain_unmixing_routine
import histolab
from histolab.filters import image_filters_functional
from histolab.filters import image_filters
from histolab.filters import morphological_filters_functional

from histolab.slide import Slide
from histolab import util
import os
import numpy as np
import ntpath
import PIL.ImageOps
import skimage.filters as sk_filters
import scipy.ndimage.morphology
import skimage.morphology
import matplotlib.pyplot as plt
from PIL import Image    

#NEW MASKS FOR DATASET PDL_1 AND CD_141 THAT ANALYZE ALSO THE PRESENCE OF NUCLEI AND CUT OFF TISSUE PARTS WITHOUT THEM

#this function returns the boolean mask
#               DOESN'T WORK FOR CD_3 DATASET

#n.b: do not use it with PDL_5 slide. 
##    doesn't work with patient 73 but it seems that we must cut off it, so no problem.
##some slides are totally deleted, it seems that it is not a bad thing because they lack hematoxylin nuclei (to be better controlled)

#RESULTS ARE IN GOOGLE DRIVE, TO BE CHECKED IN A BETTER WAY IF THESE MASKS DELETE TOO MUCH

def masks_new (pil_image, name):
    assert (name!='PDL_5'), print("this function doesn't work for it!")
    arra=np.asarray(pil_image)
    ext="png"
    mask_out, _ = get_tissue_mask(arra, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
    mask_out=mask_out.astype(bool)

    if (name=='PDL_3') or (name=='PDL_7'):
        contr=~mask_out
        arra1=np.array(util.apply_mask_image(arra,contr))
        green_ch=image_filters_functional.green_channel_filter(arra1)
        mask1=util.apply_mask_image(arra1,green_ch)
        gray=PIL.ImageOps.grayscale(mask1)
        otsu_thresh = sk_filters.threshold_otsu(np.array(gray))
        otsu=np.array(gray)>otsu_thresh
        dilation=scipy.ndimage.morphology.binary_dilation(otsu, skimage.morphology.disk(3))
        remove=morphological_filters_functional.remove_small_objects(dilation,min_size=5000)
        first=np.logical_and(green_ch,contr)
        mask_out1=np.logical_and(first,remove)
        arra2=np.array(util.apply_mask_image(arra,mask_out1))
        mask_out, _ = get_tissue_mask(arra2, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
        mask_out=mask_out.astype(bool)
        mask_out=np.logical_and(mask_out,mask_out1)
        
    return mask_out

#NEW MASKS FOR DATASET CD_3  THAT ANALYZE ALSO THE PRESENCE OF NUCLEI AND CUT OFF TISSUE PARTS WITHOUT THEM
#COLOR NORMALIZATION
#returns the boolean mask and the rgb normalized np.array


def CD_3_masks_new(pil_image,name):
    assert ((name!='42') and (name!='43') and (name!='102')), print('not readable')
    arra=np.asarray(pil_image)
    ext="png"
# color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
    stain_unmixing_routine_params = {
        'stains': ['hematoxylin', 'eosin'],
        'stain_unmixing_method': 'macenko_pca',
    }
# TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
# for macenco (obtained using rgb_separate_stains_macenko_pca()
# and reordered such that columns are the order:
# Hamtoxylin, Eosin, Null
    W_target = np.array([
        [0.5807549,  0.08314027,  0.08213795],
        [0.71681094,  0.90081588,  0.41999816],
        [0.38588316,  0.42616716, -0.90380025]
        ])
    tissue_rgb_normalized = deconvolution_based_normalization(
            arra, W_target=W_target,
            stain_unmixing_routine_params=stain_unmixing_routine_params)
    mask_out, _ = get_tissue_mask(tissue_rgb_normalized, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
    mask_out=mask_out.astype(bool)
    mask=util.apply_mask_image(tissue_rgb_normalized,mask_out)
    if (name=='66') or (name=='73') or (name=='74') or (name=='78') or (name=='80') or (name=='83') or (name=='101'):
        arra1=np.array(mask)
        mask_out1, _ = get_tissue_mask(arra1, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
        mask_out1=mask_out1.astype(bool)
        mask_out=np.logical_and(mask_out,mask_out1)
    return mask_out, tissue_rgb_normalized