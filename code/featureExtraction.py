
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
import math
from PIL import Image

from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

from skimage import restoration
from skimage import img_as_float
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.morphology import skeletonize,skeletonize_3d,medial_axis,thin

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

from skimage.morphology import convex_hull_image

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def normalising0to1(img):
    range1=np.max(img)-np.min(img)
    img=img-np.min(img)
    img=img/range1
    return img

def medianFilter(img):
	 img=ndimage.median_filter(img,3)
	 return img

def getRegionMetricRow(fname = "../data/subset0_candidates/subset0_positive_12111_107806.jpg"):
	seg = load_image(fname)
	seg = normalising0to1(seg)
	# plt.title("Image after normalising0to1")
	# plt.imshow(seg,cmap="gray")    
	# plt.show()

	#metrics
	totalArea = 0.0
	Ecc = 0.0
	EquivlentDiameter = 0.0
	weightedX = 0.0
	weightedY = 0.0
	numNodes = 0.0
	Rectangularity = 0.0
	MeanIntensity = 0.0
	Perimeter = 0.0
	Circularity = 0.0
	Elongation = 0.0
	EulerNumber = 0.0

	f1=fname.split("/")
	fileName="".join(f1[3])
	f2=f1[3].split("_")
	classLabel=f2[1]
	if classLabel == "positive":
		classLabel=1
	elif classLabel == "negative":
		classLabel=0

	original = seg
	regions, labels, numberOfRegions = getRegionFromMap(seg)

	required_label = 0
	for i in range(30,35):
		for j in range(30,35):
			if labels[i,j]!=0:
				required_label = labels[i,j]
				break

	#print("label: "+str(required_label))
	if required_label == 0 :
		print("No nodules found")
		return ([fileName, totalArea,Ecc,EquivlentDiameter, weightedX, weightedY, Rectangularity, 
			MeanIntensity, Circularity, Elongation, EulerNumber, classLabel])

	nodule_region = 0

	for region in regions:
		B = region.bbox
		if region.label == required_label:
			print(str(B[0])+" "+str(B[1])+" "+str(B[2])+" "+str(B[3])+" ")
			#region should not have the area of 64x64 
			if region.area>= 0.7*64*64:
				break
			
			if B[0] == 0 or B[1] == 0 or B[2] == 64 or B[3]==64:

				#if B[0] == 0 and B[1] == 0 and B[2] == 64 and B[3]==64:
					#break				
				mask = np.where(labels == region.label, 1, 0)
				segmented_region = original * mask
				segmented_region_bw = np.where(segmented_region==0,1,0)
				km=morphology.erosion(segmented_region_bw,np.ones([3,3]))
				#img_fill_holes = ndimage.binary_fill_holes(km)
				chull=convex_hull_image(km)
				chull=morphology.erosion(chull,np.ones([3,3]))

				seg_nodule = segmented_region * chull
				seg_nodule = morphology.dilation(seg_nodule, np.ones([2,2]))
				# plt.title("segmented nodule")
				# plt.imshow(seg_nodule,cmap="gray")
				# plt.show()

				threshold = 0.44

				labeled, nr_objects = ndimage.label(seg_nodule > threshold)
				new_regions = measure.regionprops(labeled)

				if nr_objects == 0:
					break
				max_area=0
				for new_region in new_regions:
					if new_region.area >= max_area:
						nodule_region = new_region
				break

			nodule_region = region
			newB = nodule_region.bbox
			# if newB[0] == 0 or newB[1] == 0 or newB[2] == 64 or newB[3]==64:
			# 	break
			break

		else:
			continue

	if nodule_region == 0:
		print("No nodules found")
		return ([fileName, totalArea,Ecc,EquivlentDiameter, weightedX, weightedY, Rectangularity, 
			MeanIntensity, Circularity, Elongation, EulerNumber, classLabel])
	region = nodule_region
	B = region.bbox
	totalArea = region.area
	Perimeter = region.perimeter
	Ecc = region.eccentricity
	EquivlentDiameter = region.equivalent_diameter
	weightedX += region.centroid[0]
	weightedY += region.centroid[1]
	numNodes += 1
	Rectangularity = region.extent
	#MeanIntensity = region.mean_intensity
	Circularity = (Perimeter*Perimeter)/(4*math.pi*totalArea)
	width = B[3] - B[1]
	height = B[2] - B[0]
	Elongation = min(height,width) / max(height, width)
	EulerNumber = region.euler_number
	return ([fileName, totalArea,Ecc,EquivlentDiameter, weightedX, weightedY, Rectangularity, 
		MeanIntensity, Circularity, Elongation, EulerNumber, classLabel])



def getRegionFromMap(slice_npy):
    #print("mean",np.mean(slice_npy))
    #thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)

    #plt.title("Ihresholding")
    #plt.imshow(thr,cmap="gray")
    #plt.show()

    #label_image = measure.label(thr)
    #labels = label_image.astype(int)
    #regions = measure.regionprops(labels)

    blur_radius = 0.5
    threshold = 0.44

    # eq=equalize_hist(slice_npy)
    # plt.subplot(4,3,4)
    # plt.title("Histogram equlised image")
    # plt.imshow(eq,cmap="gray")

    #eq = np.where(eq>0.44,eq,0)

    imgf = ndimage.gaussian_filter(slice_npy, blur_radius)
    #plt.title("after gaussian_filter")
    #plt.imshow(imgf,cmap="gray")
    #plt.show()

    median = medianFilter(imgf)
    # plt.title("after median_filter")
    # plt.imshow(median,cmap="gray")
    # plt.show()

    #test(median)


    labeled, nr_objects = ndimage.label(median > threshold)
    #print("no of objects", nr_objects)
    regions = measure.regionprops(labeled)

    # plt.title("regions")
    # plt.imshow(labeled,cmap="gray")
    # plt.show()

    return regions, labeled, nr_objects

def OTSUsegmentation(img):
	thresh=filters.threshold_otsu(img)
	otsu=np.where(img<0.44,0,1)
	#otsu=morphology.erosion(otsu,np.ones([3,3]))
	otsu=morphology.erosion(otsu,np.ones([3,3]))
	#otsu=np.where(otsu==1,0,1)
	return otsu





features=getRegionMetricRow()
print(features)


