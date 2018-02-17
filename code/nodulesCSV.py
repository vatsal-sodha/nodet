#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

import numpy as np
import csv
from glob import glob
import pandas as pd

file_list = sorted(glob("../data/Nodules/" + "*.jpg"))
df = pd.read_csv('../data/WorldToVoxel.csv')

output_dir = "../data/nodules.csv"

nodule_images = []
classname = []
labels = ['FileName','Width','Height','Class','Xmin','Ymin','Xmax','Ymax']

for img_file in file_list:
	#print(img_file)
	#nodule_images.append(img_file)
	classname.append("nodule")


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

radius = (df['Diam']/2)
radius = radius.values.tolist()
print("radius")
print(radius)

width = df['Diam']
width = width.values.tolist()
print("width:")
print(width)

Xmin = df['Voxel-X'] - radius
Xmin = Xmin.values.tolist()
Ymin = df['Voxel-Y'] - radius
Ymin = Ymin.values.tolist()

Xmax = df['Voxel-X'] + radius
Xmax = Xmax.values.tolist()
Ymax = df['Voxel-Y'] + radius
Ymax = Ymax.values.tolist()

print("Xmin:")
print(Xmin)

print("Xmax:")
print(Xmax)

frames = [nodule_images,width,width,classname,Xmin,Ymin,Xmax,Ymax]

df_final = pd.DataFrame({'FileName' : nodule_images, 'Width' : width , 'Height' : width, 'Class':classname,'Xmin': Xmin, 'Ymin' : Ymin, 'Xmax' : Xmax, 'Ymax' : Ymax})

print(df_final)

df_final.to_csv(output_dir, sep='\t')