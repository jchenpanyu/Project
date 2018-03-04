#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Newron-SRAF PGM pattern selection
@time: 2018/3/3
author: vincent.chen
contact: vincentchan.sysu@gmail.com
"""

import numpy as np
from sklearn import preprocessing 
from PIL import Image
import matplotlib.pyplot as plt


DEFAULT_IMAGE = r"D:\Document\Python\Project\ML-SRAF PGM analysis\c0displaymi.pgm"
CELL_H = 200
CELL_W = 200
CELL_NUM_ROW = 25
CELL_NUM_COL = 25

def PGM5toARRAY(input_pgm):
    pgm_image = Image.open(input_pgm)
    pgm_array = np.array(pgm_image, np.float32)
    # pgm_array = (pgm_array - pgm_array.min()) / (pgm_array.max() - pgm_array.min())
    min_max_scaler = preprocessing.MinMaxScaler()
    scale_pgm_array = min_max_scaler.fit_transform(pgm_array)  
    return scale_pgm_array

def crop_array(input_array, crop_H, crop_W):
    input_array_H, input_array_W = input_array.shape
    if (input_array_H >= crop_H) and (input_array_W >= crop_W):
        boundary_Y = (input_array_H - crop_H) / 2
        boundary_X = (input_array_W - crop_W) / 2
        crop_array = input_array[boundary_Y : boundary_Y + crop_H,
                                 boundary_X : boundary_X + crop_W]
        return crop_array
    else:
        print "Faile to crop"

def cellLibrary(input_array, cell_H, cell_W, cell_num_row, cell_num_col):
    cell_image = []
    cell_index = []
    cell_row_index = np.arange(cell_num_row)
    cell_col_index = np.arange(cell_num_row)
    for row in cell_row_index:
        for col in cell_col_index:
            cell_image.append(input_array[row*cell_H : (row+1)*cell_H,
                                         col*cell_W : (col+1)*cell_W])
            cell_index.append([row, col])
    return cell_image, cell_index






test_array = PGM5toARRAY(DEFAULT_IMAGE)
test_crop_array = crop_array(input_array=test_array,
                             crop_H = CELL_H*CELL_NUM_ROW,
                             crop_W = CELL_W*CELL_NUM_COL)
cell_lib, cell_index = cellLibrary(test_crop_array, cell_H=CELL_H, cell_W=CELL_W,
                                   cell_num_row=CELL_NUM_ROW, cell_num_col=CELL_NUM_COL)




print test_array.shape
print test_crop_array.shape

plt.figure()
plt.imshow(test_array, cmap='gray')
plt.show()









