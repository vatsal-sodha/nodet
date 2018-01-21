#Project: Pulmonary Nodule Detection and Lung Cancer Prediction
#Author: Mohith Damarapati
#Guide: Dr. Mukesh A Zaveri, SVNIT

#DEEP NET BASED ON FEATURES

'''The below is the code for Deep Neural Network model which takes eight features as inputs 
that we obtained from the dataset of nodules and non-nodules - prepared in the previous semester '''

#Imports

from __future__ import division #I imported it to perform float divisions
from sklearn.neural_network import MLPClassifier #Multi Layer Perceptron Classifier functions are available in Scikit Learn's MLPClassifier
import pandas as pd #Pandas for performing data frame operations
import random #Random Functions 
import numpy as np #Numpy array for operations on N-Dimensional Arrays


#Features are read into a dataframe from the previously generated features csv
df=pd.read_csv('../data/allSubsetsFeatures.csv', sep=',',low_memory=False)

Xp = []
Xn = []

#Positive Rows - Nodules
df_pos = df[df['Class']==1]
#Negative Rows - Non-nodules
df_neg = df[df['Class']==0]

#Just printing their shapes to verify the correctness
print df_pos.shape
print df_neg.shape

#Values are assigned
Positives = df_pos.values
Negatives = df_neg.values

#Values are shuffled inorder to obtain data randomly
np.random.shuffle(Positives)
np.random.shuffle(Naiegatives)

#Train and Test Positive examples
Train_P = Positives[0:1224]
Test_P = Positives[1224:1360]

#Train and Test Negative examples
Train_N = Negatives[0:1224]
Test_N = Negatives[1224:1360]

#Add and Shuffle the Train and Test Data
Train_Data = np.concatenate((Train_P,Train_N),axis=0)
Test_Data = np.concatenate((Test_P,Test_N),axis=0)
np.random.shuffle(Train_Data)
np.random.shuffle(Test_Data)

#Final Train Data and Labels
X_Train = Train_Data[0:2448,1:11]
Y_Train = Train_Data[0:2448:,11]

#Final Test Data and Labels
X_Test =  Test_Data[0:272,1:11]
Y_Test =  Test_Data[0:272,11]

#Deep Net
#Architecture: 
#2 Hidden Layers with 5 nodes in each
#Adam Optimizer - Advanced Version of Gradient Descent - I used it because it worked better than stochastic gradient descent
#Learing Rate - 0.001
#Iterations - 500
clf = MLPClassifier(solver="adam",max_iter=500,alpha = 0.001, hidden_layer_sizes=(5,5))

#Conversion into float32 because the classifier's predict function wont allow float64
X_Train = X_Train.astype(np.float32)
Y_Train = Y_Train.astype(np.float32)
X_Test = X_Test.astype(np.float32)
Y_Test = Y_Test.astype(np.float32)

#Training
clf.fit(X_Train,Y_Train)

#Results on Test Data
print "*******Results on Test Data*******"
A = clf.predict(X_Test)
A = A.tolist()

#All the classes are initialized to 0
tp=0
fp=0
fn=0
tn=0
for i in range(len(Y_Test)):
	if A[i]==1 and Y_Test[i]==1:
		tp=tp+1
	if A[i]==1 and Y_Test[i]==0:
		fp=fp+1
	if A[i]==0 and Y_Test[i]==0:
		tn=tn+1
	if A[i]==0 and Y_Test[i]==1:
		fn=fn+1


print "True Positives"
print tp

print "False Positives"
print fp

print "True Negatives" 
print tn

print "False Negatives"
print fn

#Printing True and False Positive Rates
print "True Positive Rate (Accuracy) in %"
print (tp/(tp+fn))*100
print "False Positive Rate (Accuracy) in %"
print (fp/(fp+tn))*100


#Results on Train Data

print "*******Results on Train Data*******"
A = clf.predict(X_Train)
A = A.tolist()

#All the classes are initialized to 0
tp=0
fp=0
fn=0
tn=0
for i in range(len(Y_Train)):
	if A[i]==1 and Y_Train[i]==1:
		tp=tp+1
	if A[i]==1 and Y_Train[i]==0:
		fp=fp+1
	if A[i]==0 and Y_Train[i]==0:
		tn=tn+1
	if A[i]==0 and Y_Train[i]==1:
		fn=fn+1


print "True Positives"
print tp

print "False Positives"
print fp

print "True Negatives" 
print tn

print "False Negatives"
print fn

#Printing True and False Positive Rates
print "True Positive Rate (Accuracy) in %"
print (tp/(tp+fn))*100
print "False Positive Rate (Accuracy) in %"
print (fp/(fp+tn))*100