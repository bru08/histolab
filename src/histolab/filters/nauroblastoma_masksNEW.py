
import numpy as np

from matplotlib import pylab as plt


from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask)
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
#DOESNT' WORK FOR CD_3 DATASET

def masks_new (pil_image, name):
    arra=np.asarray(pil_image)
    #
    mask_out, _ = get_tissue_mask(arra, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
    mask_out=mask_out.astype(bool)

    mask=util.apply_mask_image(arra,mask_out)
    if (PDL_slide.name=='PDL_3') or (PDL_slide.name=='PDL_5') or (PDL_slide.name=='PDL_7'):
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
        mask_out=np.logical_and(first,remove)
        arra2=np.array(util.apply_mask_image(arra,mask_out))
        mask_out, _ = get_tissue_mask(arra2, deconvolve_first=True,n_thresholding_steps=1, sigma=1.5, min_size=30)
        mask_out=mask_out.astype(bool)
        
    mask=util.apply_mask_image(arra,mask_out)
    
    mask_path = os.path.join("masks", f"{PDL_slide.name}.{ext}")
    mask.save(mask_path)
