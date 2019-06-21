import numpy as np


file_path = '../samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243547836.pcd.bin'

lidar = np.fromfile(file_path, dtype=np.float32, count=-1)

print(lidar.shape)

a = lidar.reshape([-1, 4])
print(a.shape)
b = lidar.reshape([-1, 5])
print(b.shape)