import numpy as np
import tifffile
import os
from scipy import io
import pandas as pd



dir1 = '/SOC/2018'
dir2 = 'SOC/2018'

f_lai = os.listdir(dir1)

num1 = []
for i in range(len(f_lai)):
    f = f_lai[i]
    if os.path.splitext(f)[-1] == '.tif':
        num1.append(i)

for i in range(len(num1)):
    f = f_lai[num1[i]]
    nam = dir2 + '/' + f[:-4] + '.mat'
    in_put_lai = dir1 + os.sep + f
    lai_im = tifffile.imread(in_put_lai)
    s1 = (lai_im.shape[0]-28)//7+1
    s2 = (lai_im.shape[1]-28)//7+1

    im_t = np.empty(shape=(s1*s2, 28, 28, lai_im.shape[2]), dtype='float64')

    t = 0
    t1 = []
    for ii in range(28, lai_im.shape[0], 7):
        for jj in range(28, lai_im.shape[1], 7):
            px = ii - 28
            py = jj - 28
            im_t[t, :, :, :] = lai_im[px:ii, py:jj, :]
            t = t + 1

    im_t2 = np.empty(shape=(s1,s2,28,28,lai_im.shape[2]), dtype='float64')
    for ii in range(s1):
        for jj in range(s2):
            im_t2[ii,jj,:,:,:] = im_t[(ii*46+jj),:,:,:]

    io.savemat(nam, {'im_t':im_t, 'im_t2':im_t2})


Indexnum = np.array(range(8))
n = 10
for i in range(len(Indexnum)):
    LAI1 = pd.read_excel(r"LAI_spec.xls", sheet_name=Indexnum[i])
    LAI1 = np.array(LAI1)
    LAI2 = np.empty(shape=(LAI1.shape[0]*n, LAI1.shape[1]), dtype="float64")
    for ii in range(LAI2.shape[0]):
        n1 = ii//n
        LAI2[ii, :-1] = LAI1[n1, 1:]
        LAI2[ii, -1] = LAI1[n1, 0]
    if i == 0:
        LAI_X = LAI2
    else:
        LAI_X = np.concatenate((LAI_X, LAI2), axis=0)

np.save('spc_LAI.npy', LAI_X)
