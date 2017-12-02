
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

from PIL import Image

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

def getRegionMetricRow(fname = "../../data0/subset0_candidates/subset0_positive_93273_134671.jpg"):
    seg = load_image(fname)
    seg = normalising0to1(seg)
    # plt.title("Image after normalising0to1")
    # plt.imshow(seg,cmap="gray")
        
    # plt.show()
    #metrics
    #path should be in the form of ../../data0/subset0_candidates/subset0_positive_93273_134671.jpg
    f1=fname.split("/")
    fileName="".join(f1[4])
    f2=f1[4].split("_")
    classLabel=f2[1]
    if classLabel == "positive":
    	classLabel=1
    elif classLabel == "negative":
    	classLabel=0
    	
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512 
    areas = []
    eqDiameters = []
    regions = getRegionFromMap(seg)
    for region in regions:
        if region.area > maxAllowedArea:
            continue
        totalArea += region.area
        areas.append(region.area)
        avgEcc += region.eccentricity
        avgEquivlentDiameter += region.equivalent_diameter
        eqDiameters.append(region.equivalent_diameter)
        weightedX += region.centroid[0]*region.area
        weightedY += region.centroid[1]*region.area
        numNodes += 1
    weightedX = weightedX / totalArea 
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = avgEquivlentDiameter / numNodes
    stdEquivlentDiameter = np.std(eqDiameters)
    maxArea = max(areas)
    list1=[fileName,avgArea,maxArea,avgEcc,avgEquivlentDiameter, 
        stdEquivlentDiameter, weightedX, weightedY,classLabel]
    #exclude numNode feature
    return list1
    # return np.array([fileName,avgArea,maxArea,avgEcc,avgEquivlentDiameter, 
        # stdEquivlentDiameter, weightedX, weightedY,classLabel])



def getRegionFromMap(slice_npy):
    print("mean",np.mean(slice_npy))
    #thr = np.where(slice_npy > np.mean(slice_npy),0.,1.0)

    #plt.title("Ihresholding")
    #plt.imshow(thr,cmap="gray")
    #plt.show()

    #label_image = measure.label(thr)
    #labels = label_image.astype(int)
    #regions = measure.regionprops(labels)

    blur_radius = 1.0
    threshold = 0.5

    imgf = ndimage.gaussian_filter(slice_npy, blur_radius)

    # plt.title("after gaussian_filter")
    # plt.imshow(imgf,cmap="gray")
    # plt.show()

    labeled, nr_objects = ndimage.label(imgf > threshold)
    print("no of objects", nr_objects)
    regions = measure.regionprops(labeled)

    # plt.title("regions")
    # plt.imshow(labeled,cmap="gray")
    # plt.show()

    return regions


features=getRegionMetricRow()
print(features)


