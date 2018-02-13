import numpy as np
import tensorflow as tf
import math
from scipy import ndimage
from glob import glob
from PIL import Image
import random
from skimage import io
import matplotlib.pyplot as plt

def load_image( infilename ) :
    #loads image from the directory
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" )
    return data

def normalising0to1(img):
    #normalising the image
    range1=np.max(img)-np.min(img)
    img=img-np.min(img)
    img=img/range1
    return img

def rotate_images(X_img):
    X_rotate = []
    for i in range(3):
    	rotated_img = np.rot90(X_img,(i+1))
    	X_rotate.append(rotated_img)
    	
    X_rotate = np.array(X_rotate, dtype = np.float32)
    return X_rotate

def flip_images(X_img):
    X_flip = []
    for i in range(2):
    	flip_img = np.flip(X_img,i)
    	X_flip.append(flip_img)
    X_flip.append(np.transpose(X_img))
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip


file_list = glob("../data/subset9_candidates/" + "*.jpg")

count = 0
for img_file in file_list:
	f1=img_file.split("/")
	fileName="".join(f1[3])
	f2=f1[3].split("_")
	classLabel=f2[1]


	if classLabel == "positive":
		count= count+1

		img = load_image(img_file)
		img = normalising0to1(img)
		
		try:
			flipped_images = flip_images(img)
		except:
			print("----Error: ",img_file)

		try:
			rotated_images = rotate_images(img)
		except:
			print("----Error: ",img_file)

		for i in range(3):
			filename = img_file[:-4]+"-"+str(i)+".jpg"
			#io.imsave(filename, rotated_images[i])
			print(filename)
			#plt.imshow(rotated_images[i], cmap='gray')
			#plt.show()

		for i in range(3):
			filename = img_file[:-4]+"-"+str(i+3)+".jpg"
			#io.imsave(filename, flipped_images[i])
			print(filename)
			#plt.imshow(flipped_images[i], cmap='gray')
			#plt.show()
print("Done!")
print("Count: ",count)	
    #io.imsave(filename, crop_nodule)
