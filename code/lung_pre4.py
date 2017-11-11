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

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

from scipy.ndimage.morphology import binary_fill_holes
import math
from skimage.filters import threshold_otsu

def getregionprops(image):
    labels = measure.label(image)

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)


    good_labels = []
    for prop in regions:
        B = prop.bbox
        print(B)
        print("     ")
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)

    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([5,5])) # one last dilation
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
	otsu=img<thresh
	otsu=morphology.erosion(otsu,np.ones([7,7]))
	#otsu=morphology.dilation(otsu,np.ones([4,4]))
	#otsu=np.where(otsu==1,0,1)
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
	print(img_file)
	itk_img=sitk.ReadImage(img_file)
	img_array=sitk.GetArrayFromImage(itk_img)
	imgs_to_process=img_array.astype(np.float64)


	for i in range(0,1):
		img=imgs_to_process[215]
		plt.subplot(4,3,1)
		plt.title("Original Image")
		plt.imshow(img,cmap="gray")

		img=normalising0to1(img)
		normalised_image=img
		plt.subplot(4,3,2)
		plt.title("Normalised image")
		plt.imshow(img,cmap="gray")

		plt.subplot(4,3,3)
		plotHistogram(img)
		plt.title("Original Image")

		eq=equalize_hist(img)
		plt.subplot(4,3,4)
		plt.title("Histogram equlised image")
		plt.imshow(eq,cmap="gray")

		plt.subplot(4,3,5)
		plotHistogram(eq)
		plt.title("After equalisation")

		medfilt=medianFilter(eq)
		plt.subplot(4,3,6)
		plt.title("After median filter")
		plt.imshow(medfilt,cmap="gray")

		otsu=OTSUsegmentation(medfilt)
		plt.subplot(4,3,7)
		plt.title("After Otsu segmentation")
		plt.imshow(otsu,cmap="gray")	
		
		mask=getregionprops(otsu)
		

		plt.subplot(4,3,8)
		plt.title("mask")
		plt.imshow(mask,cmap="gray")

		mask = binary_fill_holes(mask)
		plt.subplot(4,3,9)
		plt.title("After filling holes")
		plt.imshow(mask,cmap="gray")

		img_fin=mask*normalised_image
		plt.subplot(4,3,10)
		plt.title("final Image")
		plt.imshow(img_fin,cmap="gray")

		img_fin2=mask*medfilt
		plt.subplot(4,3,11)
		plt.title("final Image")
		plt.imshow(img_fin2,cmap="gray")

		plt.show()

		plt.subplot(2,2,1)
		plt.title("final Image")
		plt.imshow(img_fin,cmap="gray")

		plt.subplot(2,2,2)
		plt.title("final Image")
		plt.imshow(img_fin2,cmap="gray")

		blur_radius = 1.0
		threshold = 0.5

		thresh = threshold_otsu(img_fin)
		print(thresh)

		thresh = threshold_otsu(img_fin2)
		print(thresh)

		# smooth the image (to remove small objects)
		imgf = ndimage.gaussian_filter(img_fin, blur_radius)
		labeled, nr_objects = ndimage.label(imgf > threshold)
		print("Number of objects is ",nr_objects)
		plt.subplot(2,2,3)
		plt.title("regions")
		plt.imshow(labeled,cmap="gray")

		# find connected components
		imgf = ndimage.gaussian_filter(img_fin2, blur_radius)
		labeled, nr_objects = ndimage.label(imgf > threshold) 
		regions = measure.regionprops(labeled)
		print("Number of objects is ",nr_objects)

		plt.subplot(2,2,4)
		plt.title("regions")
		plt.imshow(labeled,cmap="gray")
		
		plt.show()

		fig, ax = plt.subplots()
		ax.imshow(labeled, cmap="gray")

		for props in regions:
		    minr, minc, maxr, maxc = props.bbox
		    bx = (minc, maxc, maxc, minc, minc)
		    by = (minr, minr, maxr, maxr, minr)
		    if (maxr-minr >50) or (maxc - minc > 50):
		    	continue
		    ax.plot(bx, by, '-b', linewidth=1.5)
		    
		    cy = (int) (minr + (maxr-minr)/2)
		    cx = (int)(minc + (maxc-minc)/2)

		    #crop=labeled[cy-25:cy+25,cx-25:cx+25]

		    #ax.plot((cx-25,cx+25,cx+25,cx-25,cx-25),(cy-25,cy-25,cy+25,cy+25,cy-25),'-r',linewidth=2.0)

		ax.axis((0, 512, 512, 0))
		plt.show()



