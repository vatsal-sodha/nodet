#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#DATA PREPARATION FOR CONV NETS

'''The below code prepares data of 1360 postive and 1360 negative examples which are trained in 
the other module named cnn_model1.py.  The data is prepared because we had uneven number of positive 
and negative training examples'''

#Imports

import pandas as pd #Pandas for manipulating data frames
import numpy as np #Numpy for operations on n-dimensional arrays
from shutil import copyfile #For copying the image files from one location to other

#Data frame is loaded from the features csv which is generated previously
df = pd.read_csv('../data/allSubsetsFeatures.csv')

#Only file name and class are chosen from the features and 1360 negative examples are taken randomly
df2 = df[['file_name','Class']]
df3 = df2[df2['Class']==0]
df_neg = df3.sample(n=1360)
df_pos = df2[df2['Class']==1]

#Randomly the data is shuffled
df_neg = df_neg.sample(frac=1)
df_pos = df_pos.sample(frac=1)

#Train and Test Set - Negatives
df_neg_train = df_neg[0:1100]
df_neg_test = df_neg[1100:1360]

#Train and Test Set - Positives
df_pos_train = df_pos[0:1100]
df_pos_test = df_pos[1100:1360]

#Both are combined
df_fin_train = pd.concat([df_pos_train,df_neg_train])
df_fin_test = pd.concat([df_pos_test,df_neg_test])

#Final Train and Test sets are randomly sampled i.e. shuffled
df_fin_train = df_fin_train.sample(frac=1)
df_fin_test = df_fin_test.sample(frac=1)

#Names and Labels are stored in a new CSV files
df_fin_train.to_csv('../data/Simple_CNN_Data/Train_Lables.csv')
df_fin_test.to_csv('../data/Simple_CNN_Data/Test_Lables.csv')

#Randomly chosen data is transfered from the original dataset to a new folder which is then used in training the CNN model
for i in df_fin_train['file_name']:
	n = i[6]
	copyfile('../../../Data_Nodet/mohith/subset'+n+'_candidates/'+i,'../data/Simple_CNN_Data/Train_Data/' + i)

for i in df_fin_test['file_name']:
	n = i[6]
	copyfile('../../../Data_Nodet/mohith/subset'+n+'_candidates/'+i,'../data/Simple_CNN_Data/Test_Data/' + i)