import numpy as np
import pandas as pd
import csv

file_name = 'center_trim_3Dtracked'


x_data = pd.read_csv('./Out/Data/' + file_name + '_x.csv', header=None)
y_data = pd.read_csv('./Out/Data/' + file_name + '_y.csv', header=None)
z_data = pd.read_csv('./Out/Data/' + file_name + '_z.csv', header=None)


xyz_data = []
for i in range(31):
    xyz_data.append(x_data.iloc[:,i].values)
    xyz_data.append(y_data.iloc[:,i].values)
    xyz_data.append(z_data.iloc[:,i].values)

df = pd.DataFrame(np.transpose(xyz_data))
df.to_csv('./Out/Data/' + file_name + '_xyz.csv')