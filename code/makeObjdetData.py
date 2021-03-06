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
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
df = pd.read_csv('../data/nodules_preProcessed_Lungs_1.csv')
#shuffle
df=shuffle(df)
total = len(df)
print("Length")
print(total)
print(df['filename'])
print("After Shuffling")
print(df)

split = 0.8

# df_train = df[0:int(total*split)]
# print(df_train)
# df_test = df[int(total*split):total]
# print(df_test)
df_train, df_test = train_test_split(df, train_size = 0.8)
print("Lenght of Train and Test")
print(len(df_train))
print(len(df_test))

#Randomly chosen data is transfered from the original dataset to a new folder which is then used in training the CNN model
directory_train='../data/ObjdetDataSet/train_preProcessed_3/'
if not os.path.exists(directory_train):
	os.makedirs(directory_train)

for i in df_train['filename']:
	im = Image.open('../data/Lungs_with_resizing/'+i)
	copyfile('../data/Lungs_with_resizing/'+i, directory_train + i)

directory_test='../data/ObjdetDataSet/test_preProcessed_3/'
if not os.path.exists(directory_test):
	os.makedirs(directory_test)

for i in df_test['filename']:
	im = Image.open('../data/Lungs_with_resizing/'+i)
	copyfile('../data/Lungs_with_resizing/'+i, directory_test + i)

df_train.to_csv('../data/ObjdetDataSet/train_preProcessed_3.csv',index=False)
df_test.to_csv('../data/ObjdetDataSet/test_preProcessed_3.csv'	,index=False)