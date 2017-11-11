from sklearn.cluster import KMeans
from skimage import morphology,measure
from skimage.transform import resize
# import np_image
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from skimage import data, img_as_float
from skimage import segmentation
from skimage.filters import sobel,roberts
from scipy import ndimage as ndi
from skimage import exposure

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




#file_list = glob("../../subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.333145094436144085379032922488" + ".mhd")
file_list = ["../../../subset0"+"*.mhd"]
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

# The locations of the nodes
df_node = pd.read_csv("../data/annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()

# print file_list[0]
for img_file in file_list:
    print img_file


    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]

    print node_x
    print node_y
    print node_z
    print diam

    itk_img=sitk.ReadImage(img_file)
    img_array=sitk.GetArrayFromImage(itk_img)
    imgs_to_process=img_array.astype(np.float64)
    center = np.array([node_x,node_y,node_z])   # nodule center
    origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)

    print "Vcenter Co-ordinates"

    print v_center[0]
    print v_center[1]
    print v_center[2]

    for i in range(0,1):
        img=imgs_to_process[int(v_center[2])]
        #plt.subplot(2,2,1)
        #plt.title("Original Image")
        #plt.imshow(img,cmap="gray")
        n1 = img.shape[0]
        n2 = img.shape[1]
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
        #plt.subplot(2,2,2)
        #plt.title("Weiner Filter")
        #plt.imshow(wf,cmap='gray')
        
        
        threshold_image=k_means(wf)
        erosion = morphology.erosion(threshold_image,np.ones([4,4]))
        dilated = morphology.dilation(erosion,np.ones([10,10]))
        #plt.subplot(2,2,3)
        #plt.title("K means")
        #plt.imshow(dilated,cmap="gray")

        mask = getregionprops(dilated)
        
        img_fin = mask*img
        '''plt.subplot(2,2,4)
        plt.title("Final Image")
        plt.imshow(img_fin,cmap="gray")
        plt.show()'''

        plt.imshow(img_fin,cmap="gray")
        plt.show()

        c1 = v_center[0]
        c2 = v_center[1]
        rad = int(diam)

        y,x = np.ogrid[-c2:n2-c2, -c1:n1-c1]
        mask = x*x + y*y <= rad*rad

        array = np.ones((n1,n2))
        array[mask] = 255

        plt.imshow(array,cmap="gray")
        plt.show()

        nodule = array*img_fin

        plt.imshow(nodule,cmap="gray")
        plt.show()