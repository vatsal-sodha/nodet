#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

import numpy as np
import csv
from glob import glob
import pandas as pd

file_list = sorted(glob("../data/Lungs_without_resizing/" + "*.jpg"))
df = pd.read_csv('../data/WorldToVoxel.csv')
print(df.head())
output_dir = "../data/nodules_preProcessed5.csv"

nodule_images = []
classname = []
labels = ['index','filename','width','height','class','xmin','ymin','xmax','ymax']


# for img_file in file_list:
# 	#print(img_file)
# 	# print(classname)
# 	#nodule_images.append(img_file)
# 	classname.append("nodule")

for i in range(0,10):
	if i==0:
		for j in range(0,67):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==1:
		for j in range(0,61):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==2:
		for j in range(0,56):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==3:
		for j in range(0,65):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==4:
		for j in range(0,62):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==5:
		for j in range(0,54):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==6:
		for j in range(0,63):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==7:
		for j in range(0,54):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==8:
		for j in range(0,60):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")
	if i==9:
		for j in range(0,59):
			nodule_images.append("subset"+str(i)+"_"+str(j)+".jpg")

print(len(nodule_images))
offset=50 #depencing on cropping
radius = (df['Diam']/2)
radius = radius.values.tolist()
print("radius")
print(len(radius))

width = df['Diam']
width = width.values.tolist()
print("width:")
print(len(width))

Xmin = df['Voxel-X'] - radius
Xmin = Xmin.values.tolist()
Ymin = df['Voxel-Y'] - radius
Ymin = Ymin.values.tolist()

Xmax = df['Voxel-X'] + radius
Xmax = Xmax.values.tolist()
Ymax = df['Voxel-Y'] + radius
Ymax = Ymax.values.tolist()

print("Xmin:")
print(len(Xmin))
classname = ["nodule"]*len(nodule_images)
print("Xmax:")
print(len(classname))

frames = [nodule_images,width,width,classname,Xmin,Ymin,Xmax,Ymax]

df_final = pd.DataFrame({'filename' : nodule_images, 'width' : width , 'height' : width, 'class':classname,'xmin': Xmin, 'ymin' : Ymin, 'xmax' : Xmax, 'ymax' : Ymax})

print(df_final)

# df_final.to_csv(output_dir,index=False)