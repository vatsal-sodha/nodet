from skimage import morphology
from skimage import measure
from skimage.transform import resize
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd

<<<<<<< HEAD
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
file_list = glob("../../subset0/" + "*.mhd")
=======
file_list = glob("../data/subset0/" + "*.mhd")
>>>>>>> word to voxel for annotations
=======

file_list = glob("../data/subset0/" + "*.mhd")

>>>>>>> 9acc25ca4a94d24312f761fd79fadef0da17fadb

def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)

# The locations of the nodes
df_node = pd.read_csv("../data/annotations.csv")
df_node["file"] = df_node["seriesuid"].apply(get_filename)
df_node = df_node.dropna()

nodules=[]
labels = ['FileName', 'World-X', 'World-Y', 'World-Z', 'Diam', 'Voxel-X', 'Voxel-Y' , 'Voxel-Z']

for img_file in file_list:
<<<<<<< HEAD
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
    print img_file
=======
    print(img_file)
>>>>>>> word to voxel for annotations


    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
=======
        print(biggest_node)
>>>>>>> word to voxel for annotations
=======
    print(img_file)


mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
if len(mini_df)>0:
	biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node


>>>>>>> 9acc25ca4a94d24312f761fd79fadef0da17fadb
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]

<<<<<<< HEAD
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
        print node_x
        print node_y
        print node_z
        print diam
=======
=======
>>>>>>> 9acc25ca4a94d24312f761fd79fadef0da17fadb
        print(node_x)
        print(node_y)
        print(node_z)
        print(diam)
<<<<<<< HEAD
>>>>>>> word to voxel for annotations
=======

>>>>>>> 9acc25ca4a94d24312f761fd79fadef0da17fadb

        itk_img=sitk.ReadImage(img_file)
        img_array=sitk.GetArrayFromImage(itk_img)
        imgs_to_process=img_array.astype(np.float64)
        center = np.array([node_x,node_y,node_z])   # nodule center
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)

<<<<<<< HEAD
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
        print "Vcenter Co-ordinates"

        print v_center[0]
        print v_center[1]
        print v_center[2]
=======
        print("Vcenter Co-ordinates")

        print(v_center[0])
        print(v_center[1])
        print(v_center[2])
>>>>>>> word to voxel for annotations

        nodules.append((img_file,node_x,node_y,node_z,diam,v_center[0],v_center[1],v_center[2]))

df = pd.DataFrame.from_records(nodules, columns=labels)
	    
<<<<<<< 4776eee312b8a97b2e99e09c818eef553287471f
df.to_csv("../data/WorldToVoxel.csv", sep='\t', encoding='utf-8')

=======
>>>>>>> 9acc25ca4a94d24312f761fd79fadef0da17fadb

        print("Vcenter Co-ordinates")

        print(v_center[0])
        print(v_center[1])
        print(v_center[2])

=======
df.to_csv("../data/WorldToVoxel_annotations.csv", sep=',', encoding='utf-8')
>>>>>>> word to voxel for annotations

        nodules.append((img_file,node_x,node_y,node_z,diam,v_center[0],v_center[1],v_center[2]))

df = pd.DataFrame.from_records(nodules, columns=labels)
	    