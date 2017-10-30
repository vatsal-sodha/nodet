from sklearn.cluster import KMeans
from skimage import morphology,measure
from skimage.transform import resize
# import np_image
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from skimage import data, img_as_float
from skimage import segmentation
from skimage.filters import sobel,roberts
from scipy import ndimage as ndi
from skimage import exposure

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

# import cv2
def weiner_filter(img):
    astro = color.rgb2gray(img)
    from scipy.signal import convolve2d as conv2
    psf = np.ones((5, 5)) / 25
    astro = conv2(astro, psf, 'same')
    astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)
    
    deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
    
    return deconvolved

def k_means(img):
    middle = img[100:400,100:400]
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(kmeans.cluster_centers_.flatten())
    thresh_img = np.where(img<threshold,1.0,0.0)
    
    return thresh_img

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

def chan_vase(image):
    # Feel free to play around with the parameters to see how they impact the result
    cv = segmentation.chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

    return cv[0]


file_list = glob("../data/subset0/" + "*.mhd")

# print file_list[0]
for img_file in file_list[0:1]:
	itk_img=sitk.ReadImage(img_file)
	img_array=sitk.GetArrayFromImage(itk_img)
	imgs_to_process=img_array.astype(np.float64)

	for i in range(0,1):
		img=imgs_to_process[100]
		plt.subplot(2,2,1)
		plt.title("Original Image")
		plt.imshow(img,cmap="gray")
		mean=np.mean(img)
		std=np.std(img)
		img=img-mean
		img=img/std
		middle=img[100:400,100:400]
		mean=np.mean(middle)
		max=np.max(img)
		min=np.min(img)
		
		img[img==max]=mean
		img[img==min]=mean
		
		#plt.subplot(3,2,2)
		#plt.title("Image after filter")
		#plt.imshow(img,cmap="gray")
			
		wf=weiner_filter(img)
		plt.subplot(2,2,2)
		plt.title("Weiner Filter")
		plt.imshow(wf,cmap='gray')
		
		
		threshold_image=k_means(wf)
		erosion = morphology.erosion(threshold_image,np.ones([4,4]))
		dilated = morphology.dilation(erosion,np.ones([10,10]))
		plt.subplot(2,2,3)
		plt.title("K means")
		plt.imshow(dilated,cmap="gray")

		mask = getregionprops(dilated)
		
		img_fin = mask*img
		plt.subplot(2,2,4)
		plt.title("final Image")
		plt.imshow(img_fin,cmap="gray")
		plt.show()
		#plt.savefig('myfilename.png')

		img_fin = rgb2gray(img_fin)

		blobs_doh = blob_doh(img_fin, max_sigma=30, threshold=.01)

		blobs_list = [blobs_doh]
		colors = ['red']
		titles = ['Determinant of Hessian']
		sequence = zip(blobs_list, colors, titles)

		#fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
		#ax = axes.ravel()


		#plt.imshow(img)
		#plt.show()
		for idx, (blobs, color, title) in enumerate(sequence):
			#ax[idx].set_title(title)
			#ax[idx].imshow(img_fin, interpolation='nearest')
			print(len(blobs))
			for blob in blobs:
				y, x, r = blob
				#c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
				#ax[idx].add_patch(c)
				img2 = img[int(x)-50:int(x)+50,int(y)-50:int(y)+50]
				plt.imshow(img2,cmap="gray")
				#print(img2.shape)
				plt.show()
			#ax[idx].set_axis_off()

		#plt.tight_layout()
		#plt.show()
        






