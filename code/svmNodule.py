import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict
from random import randint
featureFilePath="../data/subset0_features.csv"

def getAppropriateFeatures(filePath,lengthNegative):
	df=pd.read_csv(filePath)
	df_positive=df[df['Class']==1]
	df_negative=df[df['Class']==0]
	# df_negative[0:4000]
	flag=0
	while flag !=1:
		startIndex=randint(0,lengthNegative)
		if startIndex+lengthNegative < len(df_negative):
			flag=1
	frames = [df_positive,df_negative[startIndex:startIndex+lengthNegative]]
	df=pd.concat(frames)
	# print(df.describe())
	return df
def applySVM():
	# df=pd.read_csv(featureFilePath)
	df=getAppropriateFeatures(featureFilePath,300)

	X=df[['totalArea','Ecc','EquivlentDiameter','weightedX','weightedY','Rectangularity','Circularity'
	,'Elongation','EulerNumber']]
	Y=df[['Class']]

	# xTrain,xTest,yTrain,yTest=train_test_split(X,Y,test_size=0.10)
	print(X.shape)
	print(Y[Y['Class']==1].shape)

	model=SVC()
	yPred=cross_val_predict(model,X,Y.values.ravel(),cv=10)
	# print(yPred[0])
	# print(Y.iloc[0,0])
	# print(df.iloc[0,0])
	# tempFp=[]
	# for i in range(len(yPred)):
		# if yPred[i]==1 and Y.iloc[i,0]==0:

	print(cross_val_score(model,X,Y.values.ravel(),cv=10).mean())
	# model.fit(xTrain,yTrain.values.ravel())
	# yTestTrain=model.predict(xTest)
	# tn,fp,fn,tp=confusion_matrix(Y,yPred)
	print(confusion_matrix(Y,yPred))
	print(confusion_matrix(Y,yPred).ravel())
	# print(model.score(xTest,yTest.values.ravel()))
# getAppropriateFeatures(featureFilePath,1000)
applySVM()	