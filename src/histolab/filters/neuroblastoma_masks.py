import histolab
from histolab.slide import Slide
from histolab.filters import image_filters_functional
from histolab.filters import image_filters
from histolab.filters import morphological_filters_functional
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

#these functions need only the directory path of svs files.
#---scaled thumbnails of the svs images and put them in the foder "thumbnails" inside the folder "processed_DATASET_NAME"
#---masked png images and put them in the folder "masks_DATASET_NAME"

def PDL_1_masks (svsdirectory):
    #SVSDIR= '/datadisk/OBPG_NB_TIF/OBPG/PDL_1'
    SVSDIR=svsdirectory
    svs_files=os.listdir(SVSDIR)
    svs_files
    ext="png"
    os.makedirs('masks_PDL_1') 
    for svs in svs_files:
        PDL_path=f"{SVSDIR}/{svs}"
        PDL_slide = Slide(PDL_path, processed_path='processed_PDL_1')
        print(f"Slide name: {PDL_slide.name}")
        print(f"Dimensions at level 0: {PDL_slide.dimensions}")
        print(f"Dimensions at level 1: {PDL_slide.level_dimensions(level=1)}")
        print(f"Dimensions at level 2: {PDL_slide.level_dimensions(level=2)}")
        PDL_slide.save_thumbnail()
        print(f"Thumbnails saved at: {PDL_slide.thumbnail_path}") 

        arra=PDL_slide.resampled_array(scale_factor=32)

        green_ch=image_filters_functional.green_channel_filter(arra)
        mask=util.apply_mask_image(arra,green_ch)
        gray=PIL.ImageOps.grayscale(mask)
        otsu_thresh = sk_filters.threshold_otsu(np.array(gray))
        otsu=np.array(gray)>otsu_thresh
        if (PDL_slide.name=='PDL_2')  or (PDL_slide.name=='PDL_8') or (PDL_slide.name=='PDL_11') or (PDL_slide.name=='PDL_12') or (PDL_slide.name=='PDL_29') or (PDL_slide.name=='PDL_30') or (PDL_slide.name=='PDL_34') or (PDL_slide.name=='PDL_38') or (PDL_slide.name=='PDL_41') or (PDL_slide.name=='PDL_59')or (PDL_slide.name=='PDL_62') or (PDL_slide.name=='PDL_64') or (PDL_slide.name=='PDL_65')or (PDL_slide.name=='PDL_76')or (PDL_slide.name=='PDL_77')or (PDL_slide.name=='PDL_82')or (PDL_slide.name=='PDL_86')or (PDL_slide.name=='PDL_87')or (PDL_slide.name=='PDL_102')or (PDL_slide.name=='PDL_104'):
            otsu=morphological_filters_functional.remove_small_objects(otsu,min_size=8000)
            otsu=scipy.ndimage.morphology.binary_fill_holes(otsu)

        dilation=scipy.ndimage.morphology.binary_dilation(otsu, skimage.morphology.disk(3))
        remove=morphological_filters_functional.remove_small_objects(dilation,min_size=5000)
        mask1=util.apply_mask_image(arra,remove)
        mask_path = os.path.join('masks_PDL_1', f"{PDL_slide.name}.{ext}")
        if (PDL_slide.name=='PDL_67') or (PDL_slide.name=='PDL_68') or (PDL_slide.name=='PDL_70') or (PDL_slide.name=='PDL_72') or (PDL_slide.name=='PDL_73') or (PDL_slide.name=='PDL_75') or (PDL_slide.name=='PDL_83') or (PDL_slide.name=='PDL_93') or (PDL_slide.name=='PDL_95') or (PDL_slide.name=='PDL_99'):
            green_ch1=image_filters_functional.green_channel_filter(np.array(mask1))
            mask1=util.apply_mask_image(np.array(mask1),green_ch1)
            gray1=PIL.ImageOps.grayscale(mask1)  
            gray1.save(mask_path)
            otsu_thresh1 = sk_filters.threshold_multiotsu(np.array(gray1))
            otsu1=(np.array(gray1))<otsu_thresh1[1]
            remove2=morphological_filters_functional.remove_small_objects(otsu1,min_size=10000)#new
            dilation1=scipy.ndimage.morphology.binary_dilation(remove2, skimage.morphology.disk(3))
            remove3=morphological_filters_functional.remove_small_objects(dilation1,min_size=5000)
            holes2=scipy.ndimage.morphology.binary_fill_holes(remove3)
            mask2=util.apply_mask_image(np.array(mask1),holes2)  
            mask2.save(mask_path)
        else:
            mask1.save(mask_path)


def CD_141_masks (svsdirectory):
    #SVSDIR= '/datadisk/OBPG_NB_TIF/OBPG/CD_141'
    SVSDIR=svsdirectory
    svs_files=os.listdir(SVSDIR)
    svs_files
    ext="png"
    os.makedirs('masks_CD_141') 
    for svs in svs_files:
        PDL_path=f"{SVSDIR}/{svs}"
        PDL_slide = Slide(PDL_path, processed_path='processed_CD_141')
        print(f"Slide name: {PDL_slide.name}")
        PDL_slide.save_thumbnail()
        print(f"Thumbnails saved at: {PDL_slide.thumbnail_path}")  
        arra=PDL_slide.resampled_array(scale_factor=32)

        green_ch=image_filters_functional.green_channel_filter(arra)
        mask=util.apply_mask_image(arra,green_ch)#qui ritorna una pil image
        gray=PIL.ImageOps.grayscale(mask)
        otsu_thresh = sk_filters.threshold_otsu(np.array(gray))
        otsu=np.array(gray)>otsu_thresh
        if (PDL_slide.name=='141_74'):
            remove1=morphological_filters_functional.remove_small_objects(otsu,min_size=1000)
        if (PDL_slide.name=='141_30') or (PDL_slide.name=='141_78') or (PDL_slide.name=='141_82'):
            remove1=morphological_filters_functional.remove_small_objects(otsu,min_size=12000)
        else:
            remove1=morphological_filters_functional.remove_small_objects(otsu,min_size=8000)
        dilation=scipy.ndimage.morphology.binary_dilation(remove1, skimage.morphology.disk(3))
        remove=morphological_filters_functional.remove_small_objects(dilation,min_size=5000)
        mask1=util.apply_mask_image(arra,remove)
        mask_path = os.path.join('masks_CD_141', f"{PDL_slide.name}.{ext}")
        if  (PDL_slide.name=='141_68') or (PDL_slide.name=='141_72') or (PDL_slide.name=='141_73') or (PDL_slide.name=='141_70') or (PDL_slide.name=='141_78') or (PDL_slide.name=='141_80') or (PDL_slide.name=='141_83')  or (PDL_slide.name=='141_93') or (PDL_slide.name=='141_95') or (PDL_slide.name=='141_99') or (PDL_slide.name=='141_100') or (PDL_slide.name=='141_101'):
            if (PDL_slide.name=='141_80') or (PDL_slide.name=='141_83') or (PDL_slide.name=='141_93') or (PDL_slide.name=='141_95') or(PDL_slide.name=='141_99') or (PDL_slide.name=='141_100')or (PDL_slide.name=='141_101') :
                green_ch=image_filters_functional.green_channel_filter(np.array(mask1))
                mask1=util.apply_mask_image(np.array(mask1),green_ch)
            gray1=PIL.ImageOps.grayscale(mask1)  
            gray1.save(mask_path)
            otsu_thresh1 = sk_filters.threshold_multiotsu(np.array(gray1))
            otsu1=(np.array(gray1))<otsu_thresh1[1]
            if (PDL_slide.name=='141_72') or (PDL_slide.name=='141_73') or (PDL_slide.name=='141_78') or (PDL_slide.name=='141_83') or (PDL_slide.name=='141_99') or (PDL_slide.name=='141_101'):
                remove2=morphological_filters_functional.remove_small_objects(otsu1,min_size=12000)
            else:
                remove2=morphological_filters_functional.remove_small_objects(otsu1,min_size=8000)
            dilation1=scipy.ndimage.morphology.binary_dilation(remove2, skimage.morphology.disk(3))
            remove3=morphological_filters_functional.remove_small_objects(dilation1,min_size=5000)
            if (PDL_slide.name=='141_101'):
                remove3=scipy.ndimage.morphology.binary_fill_holes(remove3)
            if (PDL_slide.name=='141_93') or (PDL_slide.name=='141_100'):
                remove3=scipy.ndimage.morphology.binary_fill_holes(remove3)
                remove3=scipy.ndimage.morphology.binary_dilation(remove3, skimage.morphology.disk(10))
                remove3=scipy.ndimage.morphology.binary_fill_holes(remove3)
                remove3=morphological_filters_functional.remove_small_objects(remove3,min_size=10000)
                
            mask2=util.apply_mask_image(np.array(mask1),remove3)
            mask2.save(mask_path)
            
        else:
            mask1.save(mask_path)

def CD_3_masks(svsdirectory):
    SVSDIR=svsdirectory
    svs_files=os.listdir(SVSDIR)
    svs_files
    ext="png"
    os.makedirs('masks_CD_3') 
    for svs in svs_files:
        PDL_path=f"{SVSDIR}/{svs}"
        PDL_slide = Slide(PDL_path, processed_path='processed_CD_3')
        print(f"Slide name: {PDL_slide.name}")
        if (PDL_slide.name=='42') or (PDL_slide.name=='43') or (PDL_slide.name=='102'):
            continue
        print(f"Dimensions at level 0: {PDL_slide.dimensions}")
        PDL_slide.save_thumbnail()
        print(f"Thumbnails saved at: {PDL_slide.thumbnail_path}") 
      
        arra=PDL_slide.resampled_array(scale_factor=32)
        green_ch=image_filters_functional.green_channel_filter(arra)
        mask=util.apply_mask_image(arra,green_ch)#qui ritorna una pil image
        gray=PIL.ImageOps.grayscale(mask)
        otsu_thresh = sk_filters.threshold_otsu(np.array(gray))
        otsu=np.array(gray)>otsu_thresh
        if (PDL_slide.name=='11') or (PDL_slide.name=='18') or (PDL_slide.name=='21') or (PDL_slide.name=='23') or (PDL_slide.name=='24') or (PDL_slide.name=='26') or (PDL_slide.name=='35') or (PDL_slide.name=='46')or (PDL_slide.name=='61') or (PDL_slide.name=='90'):
            otsu=morphological_filters_functional.remove_small_objects(otsu,min_size=15000)#new
        dilation=scipy.ndimage.morphology.binary_dilation(otsu, skimage.morphology.disk(3))
        remove=morphological_filters_functional.remove_small_objects(dilation,min_size=5000)
        mask1=util.apply_mask_image(arra,remove)
        mask_path = os.path.join('masks_CD_3', f"{PDL_slide.name}.{ext}")
        if (PDL_slide.name=='67') or (PDL_slide.name=='70') or (PDL_slide.name=='73') or (PDL_slide.name=='74') or (PDL_slide.name=='78') or (PDL_slide.name=='80') or (PDL_slide.name=='81') or (PDL_slide.name=='93') or (PDL_slide.name=='95') or (PDL_slide.name=='99') or (PDL_slide.name=='100') or (PDL_slide.name=='103') or (PDL_slide.name=='11') or (PDL_slide.name=='21') or (PDL_slide.name=='34'):
            green_ch1=image_filters_functional.green_channel_filter(np.array(mask1))
            mask1=util.apply_mask_image(np.array(mask1),green_ch1)
            gray1=PIL.ImageOps.grayscale(mask1)  
            gray1.save(mask_path)
            otsu_thresh1 = sk_filters.threshold_multiotsu(np.array(gray1))
            otsu1=(np.array(gray1))<otsu_thresh1[1]
            remove2=morphological_filters_functional.remove_small_objects(otsu1,min_size=10000)#new
            dilation1=scipy.ndimage.morphology.binary_dilation(remove2, skimage.morphology.disk(3))
            remove3=morphological_filters_functional.remove_small_objects(dilation1,min_size=5000)
            holes2=scipy.ndimage.morphology.binary_fill_holes(remove3)
            mask2=util.apply_mask_image(np.array(mask1),holes2)  
            mask2.save(mask_path)
        else:
            mask1.save(mask_path)
    
    




