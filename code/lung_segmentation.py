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
from sklearn.cluster import KMeans
from random import randint

def meanNormalisation(img):
	mean = np.mean(img)
	std=np.std(img)
	img=img-mean
	img = img/std
	return img

def normalisation(img):
# Find the average pixel value near the lungs
        # to renormalize washed out images
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    return img
def plotHistogram(img):
    plt.hist(img.flatten(),100)

def k_means(img):
    middle = img[100:400,100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(kmeans.cluster_centers_.flatten())
    thresh_img = np.where(img<threshold,1.0,0.0)
    
    return thresh_img

def erosion_dilation(img):
 #
        # I found an initial erosion helful for removing graininess from some of the regions
        # and then large dialation is used to make the lung region 
        # engulf the vessels and incursions into the lung cavity by 
        # radio opaque tissue
        #
    eroded = morphology.erosion(img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    return dilation
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

    # mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    return mask
def normalisingMaskedImage(img,mask):
	new_mean = np.mean(img[mask>0])
	new_std = np.std(img[mask>0])
	old_min=np.min(img)
	img[img==old_min] = new_mean-1.2*new_std
	img = img-new_mean
	img = img/new_std
	return img
file_list = glob("../data/Nodules/" + "*.jpg")

output_dir = "../data/Lungs_without_resizing/"
temp=randint(0,525)

for img_file in file_list:

    img=Image.open(img_file)
    # print(type(img))
    # plt.subplot(2,2,1)
    # plt.title("Original Image")
    # plt.imshow(img,cmap="gray")

    # plt.subplot(3,3,2)
    # plotHistogram(img)
    # plt.title("Original Image")

    img=meanNormalisation(img)
    # print(type(img))
    normalised_image=normalisation(img)
    # plt.subplot(3,3,3)
    # plt.title("Normalised image")
    # plt.imshow(normalised_image,cmap="gray")

    # plt.subplot(3,3,4)
    # plotHistogram(normalised_image)
    # plt.title("Normalised image")

    threshold_image=k_means(normalised_image)
    # plt.subplot(3,3,5)
    # plt.title("After K-means")
    # plt.imshow(threshold_image,cmap="gray")

    eroded=erosion_dilation(threshold_image)
    # plt.subplot(2,2,2)
    # plt.title("After erosion and dilation")
    # plt.imshow(eroded,cmap="gray")

    mask=getregionprops(eroded)
    # img_fin=mask*img
    # plt.subplot(2,2,3)
    # plt.title("After mask")
    # plt.imshow(img_fin,cmap="gray") 

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    img_fin=mask*img
    # plt.subplot(2,2,3)
    # plt.title("After mask with dilation")
    # plt.imshow(img_fin,cmap="gray") 

    img=mask*img
    img=normalisingMaskedImage(img,mask)
    # mean = np.mean(img)
    # img = img - mean
    # min = np.min(img)
    # max = np.max(img)
    # img = img/(max-min)
    # plt.subplot(2,2,4)
    # plt.title("After masked image normalisation")
    # plt.imshow(img,cmap="gray")

    #make image bounding box  (min row, min col, max row, max col)
    # labels = measure.label(mask)
    # regions = measure.regionprops(labels)
    #     #
    #     # Finding the global min and max row over all regions
    #     #
    # min_row = 512
    # max_row = 0
    # min_col = 512
    # max_col = 0
    # for prop in regions:
    #     B = prop.bbox
    #     if min_row > B[0]:
    #     	min_row = B[0]
    #     if min_col > B[1]:
    #         min_col = B[1]
    #     if max_row < B[2]:
    #         max_row = B[2]
    #     if max_col < B[3]:
    #         max_col = B[3]
    # width = max_col-min_col
    # height = max_row - min_row
    # if width > height:
    # 	max_row=min_row+width
    # else:
    #     max_col = min_col+height
    #     # 
    #     # cropping the image down to the bounding box for all regions
    #     # (there's probably an skimage command that can do this in one line)
    #     # 
    # img = img[min_row:max_row,min_col:max_col]
    # mask =  mask[min_row:max_row,min_col:max_col]
    # if max_row-min_row <5 or max_col-min_col<5:  # skipping all images with no god regions
    #     pass
    # else:
    #         # moving range to -1 to 1 to accomodate the resize function
    #     mean = np.mean(img)
    #     img = img - mean
    #     min = np.min(img)
    #     max = np.max(img)
    #     img = img/(max-min)
    #     new_img = resize(img,[512,512])
    # plt.subplot(2,2,4)
    # plt.title("After resizing")
    # plt.imshow(new_img,cmap="gray")    
    # plt.show()
    filename=img_file.split("/")[3]
    print(filename)
    io.imsave(output_dir+filename, img)
    