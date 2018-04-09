from skimage import morphology
from skimage import measure
from skimage import io
from skimage.transform import resize
# import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib
from PIL import Image
from skimage.exposure import equalize_hist
from scipy import ndimage
from skimage import filters
import matplotlib.pyplot as plt



def normalising0to1(img):
    range1=np.max(img)-np.min(img)
    img=img-np.min(img)
    img=img/range1
    return img
def plotHistogram(img):
    plt.hist(img.flatten(),100)

def histogramEqualisation(img):
    return equalize_hist(img)

def medianFilter(img):
    img=ndimage.median_filter(img,3)
    return img
def dilation(img):
    img=morphology.dilation(img)
    return img



file_list = glob("../data/Nodules/" + "*.jpg")


output_dir = "../data/Nodules_preProcessed_1/"
img_id = 0
for img_file in file_list[1:2]:

    img=Image.open(img_file)
    plt.subplot(3,3,1)
    plt.title("Original Image")
    plt.imshow(img,cmap="gray")

    img=normalising0to1(img)
    plt.subplot(3,3,2)
    plt.title("Normalised image")
    plt.imshow(img,cmap="gray")

    plt.subplot(3,3,3)
    plotHistogram(img)
    plt.title("Original Image")

    eq=equalize_hist(img)
    plt.subplot(3,3,4)
    plt.title("Histogram equlised image")
    plt.imshow(eq,cmap="gray")

    plt.subplot(3,3,5)
    plotHistogram(eq)
    plt.title("After equalisation")

    medfilt=medianFilter(eq)
    plt.subplot(3,3,6)
    plt.title("After median filter")
    plt.imshow(medfilt,cmap="gray")

    dilated=dilation(medfilt)
    plt.subplot(3,3,7)
    plt.title("After dilation ")
    plt.imshow(dilated,cmap="gray")

    plt.show()

            # io.imsave(output_dir+subset_number+"_"+str(img_id)+".jpg", nodule_slice)
            # img_id+=1

