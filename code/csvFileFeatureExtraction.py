from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy import ndimage
from skimage import filters
# from scipy.signal import medfilt
# import np_image
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from featureExtraction import getRegionMetricRow
import os	

columns=['file_name','totalArea', 'Perimeter', 'Ecc','EquivlentDiameter', 'weightedX', 'weightedY', 'Rectangularity', 
		'MeanIntensity', 'Circularity', 'Elongation', 'EulerNumber', 'StandardDeviation', 
		'TranslationalInvariance', 'RotationalInvariance', 'ScaleInvariance', 'Class']
featuresFileName="../data/DataSets/"
# def makeCsvHeader():
	# df=pd.DataFrame([])
	# df.to_csv("../data/features.csv",columns=columns)
def writeToCSV(folder_path, featuresFileName):
	print(folder_path," ",featuresFileName)
	file_list=glob(folder_path+"*.jpg")
	# print(folder_path)
	# features=np.empty()
	
	features=[]
	for f in file_list:
		print(f)
		feature=getRegionMetricRow(fname=f)
		print(feature)
		features.append(feature)
		# print(pd.Series(feature,index=columns))
		# df.append(pd.Series(feature,index=columns),ignore_index=True)
		# np.concatenate(features,feature)
	df=pd.DataFrame(features,columns=columns)
	# print(df)
	# header will be written only when there is no features.csv file
	df.to_csv(featuresFileName,index=False,mode='a',header=(not os.path.exists(featuresFileName)))
#writeToCSV()
# makeCsvHeader()
for x in os.listdir("../data/DataSets"):
	print("x: ",x)
	writeToCSV(featuresFileName+x+"/", "../data/FeaturesDataSets/"+x+".csv")

#for appending 2 csv files command
#cat Train_0.8_Neg_0.5_Data.csv <(tail +2 Test_0.8_Neg_0.5_Data.csv) > Pos_0.5_Neg_0.5_Data.csv	