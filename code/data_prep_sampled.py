#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#DATA PREPARATION AFTER OVER SAMPLING OF DATASET

#Imports

import pandas as pd #Pandas for manipulating data frames
import numpy as np #Numpy for operations on n-dimensional arrays
from shutil import copyfile #For copying the image files from one location to other
from PIL import Image
#Data frame is loaded from the features csv which is generated previously
df = pd.read_csv('../data/allSubsetsFeatures.csv')

print df['Class']
ratio = 3

total_neg=1360*ratio
total_pos=1360

df_neg = df[df['Class']==0]
df_neg = df_neg.sample(n=total_neg)
df_pos = df[df['Class']==1]

print(len(df_pos))
print(len(df_neg))
#Randomly the data is shuffled
df_neg = df_neg.sample(frac=1)
df_pos = df_pos.sample(frac=1)

#Train and Test Set - Negatives
df_neg_train = df_neg[0:int(total_neg*0.8)]
df_neg_test = df_neg[int(total_neg*0.8):total_neg]

#Train and Test Set - Positives
df_pos_train = df_pos[0:int(total_pos*0.8)]
df_pos_test = df_pos[int(total_pos*0.8):total_pos]

#Both are combined
df_fin_train = pd.concat([df_pos_train,df_neg_train])
df_fin_test = pd.concat([df_pos_test,df_neg_test])

#Final Train and Test sets are randomly sampled i.e. shuffled
df_fin_train = df_fin_train.sample(frac=1)
df_fin_test = df_fin_test.sample(frac=1)



#Randomly chosen data is transfered from the original dataset to a new folder which is then used in training the CNN model
for i in df_fin_train['file_name']:
	n = i[6]
	im = Image.open('../../../Data_Nodet/data/data/subset'+str(n)+'_candidates/'+i)
	width, height = im.size
	if width==64 and height==64:
		copyfile('../../../Data_Nodet/data/data/subset'+str(n)+'_candidates/'+i,'../data/Simple_CNN_Data/Train_1to'+str(ratio)+'_Data/' + i)
		print i
	else:
		df_fin_train = df_fin_train[df_fin_train.file_name != i]

for i in df_fin_test['file_name']:
	n = i[6]
	im = Image.open('../../../Data_Nodet/data/data/subset'+str(n)+'_candidates/'+i)
	width, height = im.size
	if width==64 and height==64:
		copyfile('../../../Data_Nodet/data/data/subset'+str(n)+'_candidates/'+i,'../data/Simple_CNN_Data/Test_1to'+str(ratio)+'_Data/' + i)
		print i
	else:
		df_fin_test = df_fin_test[df_fin_test.file_name != i]

#Names and Labels are stored in a new CSV files
df_fin_train.to_csv('../data/Simple_CNN_Data/Train_1to'+str(ratio)+'_Lables.csv')
df_fin_test.to_csv('../data/Simple_CNN_Data/Test_1to'+str(ratio)+'_Lables.csv')