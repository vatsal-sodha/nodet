#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#CSV FILE GENERATION FROM THE OVER SAMPLED DATASET

#Imports

import pandas as pd #Pandas for manipulating data frames
import numpy as np #Numpy for operations on n-dimensional arrays
from shutil import copyfile #For copying the image files from one location to other
import os
from glob import glob

#Columns in the final CSV file consisting of positive and negative examples
columns=['file_name','Class']

#Path of the FileName
featuresFileName="../data/sampled_data.csv"

file_names = []
labels = []

for i in range(0,10):
	file_list=glob("../../../Data_Nodet/data/data/subset"+str(i)+"_candidates/"+"*jpg")
	#print(file_list)

	for f in file_list:
		name = os.path.basename(f)
		print(name)
		print(name[8])
		if name[8]=='n':
			labels.append(0)
			file_names.append(name)
		if name[8]=='p':
			labels.append(1)
			file_names.append(name)

df = pd.DataFrame({'file_name':file_names,'Class':labels})

df.to_csv(featuresFileName)

	