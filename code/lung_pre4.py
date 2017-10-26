from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy import ndimage
from skimage import filters
# from scipy.signal import medfilt
# import np_image
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

def getregionprops(image):
    labels = measure.label(image)

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)


    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)

    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)

    #mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    return mask

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

file_list = glob("../data/subset0/" + "*.mhd")

#print(file_list)
for img_file in file_list[0:1]:
	itk_img=sitk.ReadImage(img_file)
	img_array=sitk.GetArrayFromImage(itk_img)
	imgs_to_process=img_array.astype(np.float64)


	for i in range(0,1):
		img=imgs_to_process[100]
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

		otsu=OTSUsegmentation(medfilt)
		plt.subplot(3,3,7)
		plt.title("After Otsu segmentation")
		plt.imshow(otsu,cmap="gray")
		
		mask=getregionprops(otsu)
		img_fin=mask*img
		plt.subplot(3,3,8)
		plt.title("final Image")
		plt.imshow(img_fin,cmap="gray")
		
		plt.show()



