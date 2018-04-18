import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/nodules_preProcessed2.csv')
diameter=df['width']
# print(diameter[0])
resize_factor=1.5
diameter *=3.14
diameter *= resize_factor
diameter /=4
print(diameter.mean())
plt.xlabel("Aread of nodules")
plt.ylabel("Frequency count")
# plt.hist(diameter, bins=np.arange(diameter.min(), diameter.max()+1))
plt.hist(diameter,bins=50)
plt.show()