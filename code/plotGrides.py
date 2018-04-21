import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from math import sqrt

file_list = glob("../data/Lungs_without_resizing/" + "*.jpg")

# print file_list[0]
scales=[4,8,16,32,64,100,128]
aspect_ratio=[0.5,1]
centerx=50
centery=256
for img_file in file_list[0:1]:
	im = np.array(Image.open(img_file), dtype=np.uint8)
	fig,ax = plt.subplots()
	ax.imshow(im)
	for i in scales:
		area=i*i
		for j in aspect_ratio:
			height=area/j
			height=sqrt(height)
			width=j*height
			ax.autoscale(False)
			ax.plot([centerx-(width/2),centerx+(width/2)],[centery-(height/2),centery-(height/2)],color='r')
			ax.plot([centerx-(width/2),centerx-(width/2)],[centery-(height/2),centery+(height/2)],color='r')
			ax.plot([centerx+(width/2),centerx+(width/2)],[centery-(height/2),centery+(height/2)],color='r')
			ax.plot([centerx-(width/2),centerx+(width/2)],[centery+(height/2),centery+(height/2)],color='r')

			# rect = patches.Rectangle((200,80),width,height,linewidth=1,edgecolor='r',facecolor='none')
			# ax.add_patch(rect)
plt.show()