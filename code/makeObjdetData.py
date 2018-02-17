#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#DATA PREPARATION FOR OBJECT DETECTOR

#Imports
import pandas as pd #Pandas for manipulating data frames
import numpy as np #Numpy for operations on n-dimensional arrays
from shutil import copyfile #For copying the image files from one location to other
from PIL import Image
import os

df = pd.read_csv('../data/nodules.csv')

total = len(df)
print("Length")
print(total)
print(df['FileName'])
print("After Shuffling")
print(df)

split = 0.8

df_train = df[0:int(total*split)]
print(df_train)
df_test = df[int(total*split):total]
print(df_test)

print("Lenght of Train and Test")
print(len(df_train))
print(len(df_test))

#Randomly chosen data is transfered from the original dataset to a new folder which is then used in training the CNN model
directory_train='../data/ObjdetDataSet/train/'
if not os.path.exists(directory_train):
	os.makedirs(directory_train)

for i in df_train['FileName']:
	im = Image.open('../data/Nodules/'+i)
	copyfile('../data/Nodules/'+i, directory_train + i)

directory_test='../data/ObjdetDataSet/test/'
if not os.path.exists(directory_test):
	os.makedirs(directory_test)

for i in df_test['FileName']:
	im = Image.open('../data/Nodules/'+i)
	copyfile('../data/Nodules/'+i, directory_test + i)

df_train.to_csv('../data/ObjdetDataSet/train.csv', sep='\t')
df_test.to_csv('../data/ObjdetDataSet/test.csv', sep='\t')