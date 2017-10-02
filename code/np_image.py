import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


file_list = glob("subset0/" + "*.mhd")

def sitk_show(img, title=None, margin=0.05, dpi=40 ):
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    ax.imshow(nda,extent=extent,interpolation=None)
    
    if title:
        plt.title(title)
    
    plt.show()



#print file_list
def get_filename(case):
	global file_list
	for f in file_list:
		if case in f:
			return(f)

df_node = pd.read_csv("annotations.csv")

#print df_node

df_node["file"] = df_node["seriesuid"].apply(get_filename)

#print df_node["file"]

df_node = df_node.dropna()

#print df_node

fcount = 0

for img_file in file_list:
    #print "Getting mask for image file %s" % img_file.replace(luna_subset_path,"")
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if len(mini_df)>0:       # some files may not have a nodule--skipping those 
        biggest_node = np.argsort(mini_df["diameter_mm"].values)[-1]   # just using the biggest node
        node_x = mini_df["coordX"].values[biggest_node]
        node_y = mini_df["coordY"].values[biggest_node]
        node_z = mini_df["coordZ"].values[biggest_node]
        diam = mini_df["diameter_mm"].values[biggest_node]

        itk_img = sitk.ReadImage(img_file) 
        img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering
        print img_array.shape
        center = np.array([node_x,node_y,node_z])   # nodule center
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        #print origin
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
        
        i = 0
        for i_z in range(int(v_center[2])-1,int(v_center[2])+2):
            print i_z
            print origin
            print center[0]
            print center[1]
            print v_center[2]
            print v_center[0]
            print v_center[1]
            sitk_show(itk_img[:,:,i_z])
        	#img = itk_img[:,:,i_z]
        	#nda = sitk.GetArrayFromImage(img)

        	#plt.imshow(nda)
        	#plt.axhline(y=v_center[1],color="g")
        	#plt.axvline(x=v_center[0],color="g")
        	#plt.show()
            c = raw_input("Press Any Key")
			