from skimage import morphology
from skimage import measure
from skimage import io
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib
from PIL import Image


def normalising0to1(img):
    range1=np.max(img)-np.min(img)
    img=img-np.min(img)
    img=img/range1
    return img

subset_number = 6
subset_number = "subset"+str(subset_number)

file_list = glob("../data/"+subset_number+"/" + "*.mhd")

df_node = pd.read_csv("../data/WorldToVoxel.csv")

output_dir = "../data/Nodules/"
img_id = 0
for img_file in file_list:
	print(df_node)
	mini_df = df_node[df_node["FileName"]==img_file] #get all nodules associate with file
	if len(mini_df)>0:       # some files may not have a nodule--skipping those 


		itk_img=sitk.ReadImage(img_file)
		img_array=sitk.GetArrayFromImage(itk_img)
		img_slices=img_array.astype(np.float64)

		for index, row in mini_df.iterrows():
			cx = int(row['Voxel-X'])
			cy = int(row['Voxel-Y'])
			cz = int(row['Voxel-Z'])

			nodule_slice = img_slices[cz]
			nodule_slice=normalising0to1(nodule_slice)

			print(img_file)
			print(cx)
			print(cy)
			print(cz)

			io.imsave(output_dir+subset_number+"_"+str(img_id)+".jpg", nodule_slice)
			img_id+=1