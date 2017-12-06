import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score,cross_val_predict
from random import randint
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


featureFilePath="../../data0/allSubsetsFeatures.csv"

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
def applySVM(lengthNegative):
	# df=pd.read_csv(featureFilePath)
	df=getAppropriateFeatures(featureFilePath,lengthNegative)
	print("Total data ",df.shape)
	X=df[['totalArea','Ecc','EquivlentDiameter','weightedX','weightedY','Rectangularity','Circularity'
	,'Elongation','EulerNumber']]
	Y=df[['Class']]

	#sampling on training data we need else test data will have high accuracy
	# xTrain,xTest,yTrain,yTest=train_test_split(X,Y,test_size=0.20)
	# print((yTrain==0).sum())
	# print((yTrain==1).sum())
	# print("Total Train ",len(xTrain))
	# print("Total test ", len(xTest))
	print("Total ", len(Y))


	sm=SMOTE(ratio="all")
	# xTrainSampled,yTrainSampled=sm.fit_sample(xTrain,yTrain.values.ravel())
	xTrainSampled,yTrainSampled=sm.fit_sample(X,Y.values.ravel())

	negativeTraining=(yTrainSampled==0).sum()
	positiveTraining=(yTrainSampled==1).sum()
	print("Total train after sampling ",len(xTrainSampled))

	# print(len(yTrainSampled))

	# print(X.shape)
	# print(Y[Y['Class']==1].shape)
	model=SVC()
	yPred=cross_val_predict(model,xTrainSampled,yTrainSampled,cv=10)
	# print(yPred[0])
	# print(Y.iloc[0,0])
	# print(df.iloc[0,0])
	# tempFp=[]
	# for i in range(len(yPred)):
		# if yPred[i]==1 and Y.iloc[i,0]==0:

	print(cross_val_score(model,xTrainSampled,yTrainSampled,cv=10).mean())
	# model=model.fit(xTrainSampled,yTrainSampled)
	# yTestTrain=model.predict(xTest)
	# tn,fp,fn,tp=confusion_matrix(yTest,yTestTrain).ravel()
	tn,fp,fn,tp=confusion_matrix(yTrainSampled,yPred).ravel()

	return (tn,fp,fn,tp,positiveTraining,negativeTraining)
	# print(confusion_matrix(yTest,yTestTrain).ravel())
	# print(confusion_matrix(Y,yPred).ravel())
	# print(model.score(xTest,yTest.values.ravel()))
# getAppropriateFeatures(featureFilePath,1000)
def plotCurve(lengthNegative=1500):
	tpRate=[]
	fpRate=[]
	fnRate=[]
	positiveTraining=[]
	negativeTraining=[]
	negative=[]
	for i in range(1000,lengthNegative,500):
		tn,fp,fn,tp,posTraining,negTraining=applySVM(i)
		tpTempRate=tp/(tp+fp)
		fpTempRate=fp/(fp+tn)
		fnTempRate=fn/(fn+tp)

		tpRate.append(tpTempRate)
		fpRate.append(fpTempRate)
		fnRate.append(fnTempRate)
		
		positiveTraining.append(posTraining)
		negativeTraining.append(negTraining)
		negative.append(i)
	plt.title("SVM with different negative size")
	plt.xlabel("No of negative Examples in training")
	plt.ylabel("Rates")
	plt.plot(negative,tpRate,'o-', color="b",label="True Positive Rate")
	plt.plot(negative,fpRate,'o-', color="g",label="False Positive Rate")
	plt.plot(negative,fnRate,'o-', color="c",label="False Negative Rate")
	plt.legend(loc="best")
	plt.show()


# applySVM()	
plotCurve(5000)