import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image


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


'''def readCSV(filename):
	lines = []
	with open(filename, "rb") as f:
		csvreader = csv.reader(f)
		for line in csvreader:
			lines.append(line)

	return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
	stretchedVoxelCoord = np.absolute(worldCoord - origin)
	voxelCoord = stretchedVoxelCoord / spacing
	
	return voxelCoord

def normalizePlanes(npzarray):
	maxHU = 400.
	minHU = -1000.
	
	npzarray = (npzarray - minHU) / (maxHU - minHU)
	npzarray[npzarray>1] = 1.
	npzarray[npzarray<0] = 0.
	
	return npzarray
'''
def load_itk(filename):

    itkimage = sitk.ReadImage(filename)
    sitk_show(itkimage[:,:,70])
    ct_scan = sitk.GetArrayFromImage(itkimage)
    print ct_scan.shape
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

ct_scan, origin, spacing = load_itk('1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd')
#cands = readCSV('candidates.csv')


#print cands

'''for cand in cands[1:]:
	worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
	voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)
	voxelWidth = 65
'''