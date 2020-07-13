import os
import numpy as np
import ntpath
import skimage.filters as sk_filters
import scipy.ndimage.morphology
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import skimage.morphology as sk_morphology
import skimage.color as sk_color
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import cv2

#INPUT: the rgb array, the name of the tile (without extensions) and the directory where saving the H-S plots (Hue-Saturation). 
#RETURN: SCORE that is the percentage of marker in that tile.
#        H-S plots of each tile

def scoring(rgb_array,name, H_S_plot_directory):
    ext="png"
    path = os.path.join(H_S_plot_directory,f"{name}.{ext}")
    rgb=rgb_array
    hsv=sk_color.rgb2hsv(rgb)
    pixel_colors = rgb.reshape((np.shape(rgb)[0]*np.shape(rgb)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    h, s, v = cv2.split(hsv)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.scatter(h.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    plt.savefig(path)
    x=h.flatten()
    totsize=len(x)
    y=s.flatten()
    true=x<0.1
    x=x[true]
    y=y[true]
    true2=y>0.2
    x=x[true2]
    y=y[true2]
    size=len(x)
    SCORE=(size/totsize)*100
    return SCORE

    #the same function whitout plotting the H-S diagrams (that are very expensive!)

def scoring_light(rgb_array,name): #without saving H-S plot
    rgb=rgb_array
    hsv=sk_color.rgb2hsv(rgb)
    h, s, v = cv2.split(hsv)
    x=h.flatten()
    totsize=len(x)
    y=s.flatten()
    true=x<0.1
    x=x[true]
    y=y[true]
    true2=y>0.2
    x=x[true2]
    y=y[true2]
    size=len(x)
    SCORE=(size/totsize)*100
    return SCORE

    #the same function having as input the PIL image

 def scoring_light_PIL(PIL_image,name): #without saving H-S plot
    rgb = np.asarray(PIL_image)
    hsv=sk_color.rgb2hsv(rgb)
    h, s, v = cv2.split(hsv)
    x=h.flatten()
    totsize=len(x)
    y=s.flatten()
    true=x<0.1
    x=x[true]
    y=y[true]
    true2=y>0.2
    x=x[true2]
    y=y[true2]
    size=len(x)
    SCORE=(size/totsize)*100
    return SCORE   