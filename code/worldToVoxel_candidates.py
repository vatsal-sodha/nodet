from skimage import morphology
from skimage import measure
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

subset_number = 3
subset_number = "subset"+str(subset_number)

file_list = glob("../data/"+subset_number+"/" + "*.mhd")
def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

# The locations of the nodes
df_node = pd.read_csv("../data/candidates.csv")
print(df_node.shape)

df_node["file"] = df_node["seriesuid"].apply(get_filename)
print(df_node.shape)
print(df_node.seriesuid.unique().shape)
df_node = df_node.dropna()

print(df_node.shape)
print(df_node.seriesuid.unique().shape)
labels = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'class', 'file' , 'Voxel-X', 'Voxel-Y' , 'Voxel-Z']
nodules = pd.DataFrame(columns=labels)

for img_file in file_list:
    print(img_file)

    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 

        itk_img=sitk.ReadImage(img_file)
        img_array=sitk.GetArrayFromImage(itk_img)
        imgs_to_process=img_array.astype(np.float64)
        #center = np.array([node_x,node_y,node_z])   # nodule center
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

        mini_df["Voxel-X"] = np.rint((mini_df["coordX"] - origin[0])/spacing[0])
        mini_df["Voxel-Y"] = np.rint((mini_df["coordY"] - origin[1])/spacing[1])
        mini_df["Voxel-Z"] = np.rint((mini_df["coordZ"] - origin[2])/spacing[2])
        
        nodules = nodules.append(mini_df)
	    
nodules.to_csv("../data/WorldToVoxel_"+subset_number+"_candidates.csv", sep=',', encoding='utf-8')

print(subset_number+" done!")