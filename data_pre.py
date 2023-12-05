import numpy as np
import tifffile
import os
from scipy import io
import pandas as pd

dir1 = 'data/im'

f1 = os.listdir(dir1)

num1 = []
for i in range(len(f1)):
    f = f1[i]
    if os.path.splitext(f)[-1] == '.tif':
        num1.append(i)

for i in range(len(num1)):
    f = f1[num1[i]]
    in_put_im = dir1 + os.sep + f
    im1 = tifffile.imread(in_put_im)
    im2 = np.sum(im1, axis=2)
    p1 = np.where(im2 == 0)
    p11 = np.array(p1[0])
    p12 = np.array(p1[1])
    s1 = (im1.shape[0]-28)//14+1
    s2 = (im1.shape[1]-28)//14+1

    im_t = np.empty(shape=(s1*s2, 28, 28, im1.shape[2]), dtype='float64')

    t = 0
    t1 = []
    for ii in range(28, im1.shape[0], 14):
        for jj in range(28, im1.shape[1], 14):
            px = ii - 28
            py = jj - 28
            im_t[t, :, :, :] = im1[px:ii, py:jj, :]
            px1 = np.array(range(px, ii))
            py1 = np.array(range(py, jj))
            logic_x = np.in1d(p11, px1)
            logic_y = np.in1d(p12, py1)
            num_x = np.where(logic_x)[0]
            num_y = np.where(logic_y)[0]
            if not np.any(np.in1d(num_x, num_y)):
                t1.append(t)
            t = t + 1

    t2 = np.random.choice(t1, 20, replace=False)
    im_x1 = im_t[t2, :, :, :]
    if i == 0:
        IM_X = im_x1
    else:
        IM_X = np.concatenate((IM_X, im_x1), axis=0)

np.save('im_x.npy', IM_X)


Indexnum = np.array(range(4))
n = 20
for i in range(len(Indexnum)):
    spc_LAI = pd.read_excel(r"data/LAI_spec.xls", sheet_name=Indexnum[i])
    spc_LAI = np.array(spc_LAI)
    spc_LAI1 = spc_LAI[2, :]
    spc_LAI2 = np.empty(shape=(spc_LAI1.shape[0]*n, spc_LAI1.shape[1]), dtype="float64")
    for ii in range(spc_LAI2.shape[0]):
        n1 = ii//n
        spc_LAI2[ii, :-1] = spc_LAI1[n1, 1:]
        spc_LAI2[ii, -1] = spc_LAI1[n1, 0]
    if i == 0:
        spc_LAI_X = spc_LAI2
    else:
        spc_LAI_X = np.concatenate((spc_LAI_X, spc_LAI2), axis=0)

np.save('spc_LAI.npy', spc_LAI_X)
