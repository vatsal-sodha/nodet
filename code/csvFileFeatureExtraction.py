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

columns=['file_name','totalArea','Ecc','EquivlentDiameter', 'weightedX', 'weightedY', 'Rectangularity', 
		'MeanIntensity', 'Circularity', 'Elongation', 'EulerNumber', 'Class']
featuresFileName="../data/allSubsetsFeatures.csv"
# def makeCsvHeader():
	# df=pd.DataFrame([])
	# df.to_csv("../data/features.csv",columns=columns)
def writeToCSV(folder_path="../data/subset9_candidates/"):
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
writeToCSV()
# makeCsvHeader()