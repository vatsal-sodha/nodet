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

from random import randint

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

def OTSUsegmentation(img):
    thresh=filters.threshold_otsu(img)
    otsu=img>thresh
    otsu=morphology.erosion(otsu,np.ones([6,6]))
    otsu=morphology.dilation(otsu)
    return otsu

def minimumSegmentation(img):
    thresh=filters.threshold_minimum(img)
    minimum=img>thresh
    minimum=morphology.dilation(minimum)
    minimum=morphology.erosion(minimum,np.ones([4,4]))
    return minimum

def cropImage(img):
    width, height = img.shape
    cropx=50
    cropy=50

    temp=img[cropy:height-cropy,cropx:width-cropx]
    return temp



file_list = glob("../data/Nodules/" + "*.jpg")


output_dir = "../data/nodules_preProcessed_1/"
img_id = 0
# temp=randint(0,525)
for img_file in file_list[0:10]:

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

    minimum=minimumSegmentation(medfilt)
    print(minimum.shape)
    plt.subplot(3,3,8)
    plt.title("After minimum segmentation")
    plt.imshow(minimum,cmap="gray")

    cropedImage=cropImage(dilated)
    print(cropedImage.shape)
    plt.subplot(3,3,9)
    plt.title("After Cropping")
    plt.imshow(cropedImage,cmap="gray")

    plt.show()
    print(img_file)
    # print(output_dir+img_file)
    filename=img_file.split("/")[3]
    # io.imsave(output_dir+filename, cropedImage)
            # img_id+=1

