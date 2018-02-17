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

subset_number = 0
subset_number = "subset"+str(subset_number)

file_list = glob("../data/"+subset_number+"/" + "*.mhd")

def normalising0to1(img):
    range1=np.max(img)-np.min(img)
    img=img-np.min(img)
    img=img/range1
    return img

def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

def awayFromNodules(nodules, cx, cy, cz):
    for index, row in nodules.iterrows():
        nx = int(row['Voxel-X'])
        ny = int(row['Voxel-Y'])

        if not (abs(nx-cx)>64 or abs(ny-cy)>64):
            return False
    return True        

output_dir = "../data/"+subset_number+"_candidates/"
# The locations of the nodes
df_node = pd.read_csv("../data/WorldToVoxel_"+subset_number+"_candidates.csv")
print(df_node.shape)

img_id = 100000

for img_file in file_list:
    print(" ")
    print("file name: ")
    print(img_file)

    initial_count = img_id

    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 

        nodules = mini_df.loc[mini_df['class']==1]
        non_nodules = mini_df.loc[mini_df['class']==0]
        print("nodules shape, ",nodules.shape)
        print("non_nodules shape, ",non_nodules.shape)

        itk_img=sitk.ReadImage(img_file)
        img_array=sitk.GetArrayFromImage(itk_img)
        img_slices=img_array.astype(np.float64)

        #generate nodules
        for index, row in nodules.iterrows():
            cx = int(row['Voxel-X'])
            cy = int(row['Voxel-Y'])
            cz = int(row['Voxel-Z'])

            nodule_slice = img_slices[cz]
            nodule_slice=normalising0to1(nodule_slice)
            crop_nodule = nodule_slice[cy-32:cy+32,cx-32:cx+32]
            filename = output_dir+subset_number+"_positive_"+img_file[-9:-4]+"_"+str(img_id)+".jpg"
            io.imsave(filename, crop_nodule)
            img_id+=1
            print(cx,cy,cz)
            print(filename)

        for index, row in non_nodules.iterrows():
            cx = int(row['Voxel-X'])
            cy = int(row['Voxel-Y'])
            cz = int(row['Voxel-Z'])

            if (awayFromNodules(nodules, cx, cy, cz) and (cx>32 and cy>32 and cx<480 and cy<480)):
                nodule_slice = img_slices[cz]
                nodule_slice=normalising0to1(nodule_slice)
                crop_nodule = nodule_slice[cy-32:cy+32,cx-32:cx+32]
                filename = output_dir+subset_number+"_negative_"+img_file[-9:-4]+"_"+str(img_id)+".jpg"
                print(cx,cy,cz)
                io.imsave(filename, crop_nodule)
                img_id+=1
                print(filename)

        final_count = img_id
        print("total blobs: ",final_count-initial_count)
        
print(subset_number+" done!")

	    
