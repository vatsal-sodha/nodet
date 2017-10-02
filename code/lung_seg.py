from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
# import np_image
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
# import cv2

file_list = glob("../../subset0/" + "*.mhd")

# print file_list[0]
for img_file in file_list[0:1]:
    itk_img=sitk.ReadImage(img_file)
    img_array=sitk.GetArrayFromImage(itk_img)
    imgs_to_process=img_array.astype(np.float64)

    for i in range(0,1):
        img=imgs_to_process[100]
        plt.subplot(3,2,1)
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

        plt.subplot(3,2,2)
        plt.title("Image after filter")
        plt.imshow(img,cmap="gray")

        # Using Kmeans to separate foreground (radio-opaque tissue)
        # and background (radio transparent tissue ie lungs)
        # Doing this only on the center of the image to avoid 
        # the non-tissue parts of the image as much as possible

        kmeans=KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
        centers=sorted(kmeans.cluster_centers_.flatten())
        threshold=np.mean(kmeans.cluster_centers_.flatten())
        thresh_img=np.where(img<threshold,1.0,0.0)
        plt.subplot(3,2,3)
        plt.title("After Clustering")
        plt.imshow(thresh_img,cmap="gray")
        plt.axhline(y=centers[0],color="g")

        eroded = morphology.erosion(thresh_img,np.ones([4,4]))
        dilation = morphology.dilation(eroded,np.ones([10,10]))
        plt.subplot(3,2,4)
        plt.title("After morphology and erosion")
        plt.imshow(dilation,cmap="gray")
        labels = measure.label(dilation)

        # print labels.shape
        # print labels
        label_vals = np.unique(labels)
        # print label_vals
        regions = measure.regionprops(labels)
        # print regions

        good_labels = []
        for prop in regions:
            B = prop.bbox
            # print type(B[0])
            # print type(prop.intensity_image)
            # plt.imshow(prop.intensity_image,cmap="gray")
            # plt.show()
            # print prop.centroid
            # print B[0:4]
            if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
                good_labels.append(prop.label)

        mask = np.ndarray([512,512],dtype=np.int8)
        mask[:] = 0

        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)

        plt.subplot(3,2,5)
        plt.title("Mask")
        plt.imshow(mask,cmap="gray")

        # plt.show()

        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        # mask = morphology.erosion(mask,np.ones([25,25]))
        imgs_to_process[337] = mask

        # plt.imshow(imgs_to_process[337],cmap="gray")

        # plt.show()

        img_fin = imgs_to_process[337]*img
        plt.subplot(3,2,6)
        plt.title("final Image")
        plt.imshow(img_fin,cmap="gray")

        plt.show()


# for img_file in file_list[0:1]:
# 	itk_img = sitk.ReadImage(img_file)
# 	img_array = sitk.GetArrayFromImage(itk_img)
# 	# I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
# 	imgs_to_process = img_array.astype(np.float64)

# 	# print "on image", img_array
# 	# print len(imgs_to_process)
# 	for i in range(0,1):
#         img=imgs_to_process[337]
#         plt.subplot(3,2,1)
#         plt.title("Original Image")
#         plt.imshow(img,cmap="gray")
#         mean=np.mean(img)
#         std = np.std(img)
#         img = img-mean
#         img = img/std
#         middle = img[100:400,100:400] 
#         mean = np.mean(middle)  
#         max = np.max(img)
#         min = np.min(img)
#         # To improve threshold finding, I'm moving the 
#         # underflow and overflow on the pixel spectrum
#         img[img==max]=mean
#         img[img==min]=mean

#         plt.subplot(3,2,2)
#         plt.title("Image after filter")
#         plt.imshow(img,cmap="gray")

#         # plt.show()
#         # Using Kmeans to separate foreground (radio-opaque tissue)
#         # and background (radio transparent tissue ie lungs)
#         # Doing this only on the center of the image to avoid 
#         # the non-tissue parts of the image as much as possible
        



#         kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
#         centers = sorted(kmeans.cluster_centers_.flatten())
#         # print kmeans.cluster_centers_
#         # print centers
#         threshold = np.mean(kmeans.cluster_centers_.flatten())
#         thresh_img = np.where(img<threshold,1.0,0.0)

#         # print threshold
#         # print centers
#         plt.subplot(3,2,3)
#         plt.title("After clustering")
#         plt.imshow(thresh_img,cmap="gray")
#         plt.axhline(y=centers[0],color="g")
#         # plt.show()

#         eroded = morphology.erosion(thresh_img,np.ones([4,4]))
#         dilation = morphology.dilation(eroded,np.ones([10,10]))

#         plt.subplot(3,2,4)
#         plt.title("After morphology and erosion")
#         plt.imshow(dilation,cmap="gray")
#         # plt.axhline(y=centers[0],color="g")
#         # plt.show()

#         labels = measure.label(dilation)

#         # print labels.shape
#         # print labels
#         label_vals = np.unique(labels)
#         # print label_vals
#         regions = measure.regionprops(labels)
#         # print regions

#         good_labels = []
#         for prop in regions:
#             B = prop.bbox
#             # print type(B[0])
#             # print type(prop.intensity_image)
#             # plt.imshow(prop.intensity_image,cmap="gray")
#             # plt.show()
#             # print prop.centroid
#             # print B[0:4]
#             if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
#                 good_labels.append(prop.label)

#         mask = np.ndarray([512,512],dtype=np.int8)
#         mask[:] = 0

#         for N in good_labels:
#         	mask = mask + np.where(labels==N,1,0)

#         plt.subplot(3,2,5)
#         plt.title("Mask")
#        	plt.imshow(mask,cmap="gray")

#         # plt.show()

#         mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
#         # mask = morphology.erosion(mask,np.ones([25,25]))
#         imgs_to_process[337] = mask

#         # plt.imshow(imgs_to_process[337],cmap="gray")

#         # plt.show()

#         img_fin = imgs_to_process[337]*img
#         plt.subplot(3,2,6)
#         plt.title("final Image")
#         plt.imshow(img_fin,cmap="gray")

#         plt.show()



        '''
        print img_fin.shape
        new_mean = np.mean(img[mask>0])  
        new_std = np.std(img[mask>0])
        old_min = np.min(img)
        img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
        img = img-new_mean
        img = img/new_std
        plt.imshow(img,cmap="gray")
        plt.show()

        img_fin = imgs_to_process[337]*img

        plt.imshow(img_fin,cmap="gray")

        plt.show()'''






