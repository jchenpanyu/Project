#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
Newron-SRAF PGM pattern selection
@time: 2018/3/3
author: vincent.chen
contact: vincentchan.sysu@gmail.com

2018-03-05:
    1 加入K-means代码cluster切割后的clip，K值设定的原理为：num_col + num_row, 保证每一列跟每一行都有一个cluster
    2 给PGM5toARRAY加一个参数控制要不要做归一化处理
    3 加入代码从每个cluster中去pick up AUTO_SELECTION_PERCENT的pattern
    4 加入卷积代码
    5 对每个cluster下的image做卷积desample(200*200 -> 20*20), 然后计算同一个cluster下不同image同一个空间位置的标准差，最后把所有像素点的std求和来衡量一个cluster下图像的variance，这个variance越大，代表一个cluster下不同image之间的差异越大，反之差异越少 --> 可以用来指导取样的数目

2018-03-06:
    6 把crop size从200pixel扩大到400pixel
    7 保存每个cluster跟auto-pickup后的image
    8 把每个cluster的std保存到相应的List里，n_sample x 1
    9 每一个cluster的随机取样比例跟该cluster的std相关，std越大，比例越大
    10 直接读取的PGM显示是上下颠倒的，用array=array[::-1]翻转回去
    11 在full-GDS上把把selected cell的轮廓画出来
    12 用200crop size pick出来的有更好的coverage，改回200
    13 基于std pickup的比例调低
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# GDS size: 80x80 um^2 --> ~5700x5700 pixel (加上外扩的margin，实际为6180x6180 pixel)
DEFAULT_IMAGE = r"C:\Users\vincchen\Documents\1_Assignment\184-Newron_UMC-L14\data\c0displaymi.pgm"
CELL_H = 200 # cropped cell height
CELL_W = 200 # cropped cell width
CELL_NUM_ROW = 28 # number of cropped clips row
CELL_NUM_COL = 28 # number of cropped clips colunm

CONVOLUTION_FILTER_SIZE = 10 #把一个cropped的image desample，传递给后面的function算std

AUTO_SELECTION_PERCENT = 0.1 # randomly pick up 10% of patterns from each clusters


def PGM5toARRAY(input_pgm, normalization):
    pgm_image = Image.open(input_pgm) # 读取PGM
    pgm_array = np.array(pgm_image, np.float32) # convert pgm to array
    pgm_array = pgm_array[::-1] # 读入的pgm上下是颠倒的
    if normalization:
        pgm_array = (pgm_array - pgm_array.min()) / (pgm_array.max() - pgm_array.min()) # 归一化到0-1
        #min_max_scaler = preprocessing.MinMaxScaler()
        #pgm_array = min_max_scaler.fit_transform(pgm_array)  # 归一化到0-1
    return pgm_array

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
    cell_image = None
    cell_index = []
    cell_row_index = np.arange(cell_num_row)
    cell_col_index = np.arange(cell_num_row)
    # 按序crop出image并flatten了image后append到cell_image这个list里面
    # 把每个clip对于的row, colunm保存到cell_index这个list里面
    for row in cell_row_index:
        for col in cell_col_index:
            append_array = input_array[row*cell_H : (row+1)*cell_H, 
                                      col*cell_W : (col+1)*cell_W].flatten()
            if cell_image is None:
                cell_image = append_array
            else:
                cell_image = np.row_stack((cell_image, append_array))
            cell_index.append([row, col]) 
    return cell_image, cell_index

def convolutionAverageImage(input_array, input_filter=np.ones((CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE))/np.power(CONVOLUTION_FILTER_SIZE, 2)):
    input_array_shape = input_array.shape
    input_filter_shape = input_filter.shape
    output_array_H = input_array_shape[0]/input_filter_shape[0]
    output_array_W = input_array_shape[1]/input_filter_shape[1]
    AverageImage = np.zeros((output_array_H, output_array_W))
    for i in np.arange(output_array_H):
        for j in np.arange(output_array_W):
            AverageImage[i][j] = sum(sum(input_array[i*input_filter_shape[0]:(i+1)*input_filter_shape[0], j*input_filter_shape[1]:(j+1)*input_filter_shape[1]] * input_filter))    
    return AverageImage

def stdImage(cluster_array):
    num_sample = cluster_array.shape[0]
    sub_image = np.zeros((num_sample, (CELL_H/CONVOLUTION_FILTER_SIZE)*(CELL_W/CONVOLUTION_FILTER_SIZE)))
    for n in np.arange(num_sample):
        sub_image[n, :] = convolutionAverageImage(cluster_array[n].reshape((CELL_H, CELL_W))).flatten()
    #sub_image_sumStd = np.sum(np.std(sub_image, axis=0))  
    sub_image_aveStd = np.sum(np.std(sub_image, axis=0)) / sub_image.shape[1]
    return sub_image_aveStd


####################################################################################################
####################################################################################################
## main flow
####################################################################################################
####################################################################################################

# test dataset
test_array = PGM5toARRAY(DEFAULT_IMAGE, normalization=True)
test_crop_array = crop_array(input_array=test_array,
                             crop_H = CELL_H*CELL_NUM_ROW,
                             crop_W = CELL_W*CELL_NUM_COL)
cell_lib, cell_index = cellLibrary(test_crop_array, cell_H=CELL_H, cell_W=CELL_W,
                                   cell_num_row=CELL_NUM_ROW, cell_num_col=CELL_NUM_COL)


# plot and save cropped image:
fig_1 = plt.figure()
ax_1 =fig_1.add_subplot(111)
ax_1.imshow(test_crop_array, cmap='gray')
fig_1.tight_layout()
ax_1.axis('off')
plt.title("Full-Chip Size: " + str(CELL_H*CELL_NUM_ROW*14) + "x" + str(CELL_W*CELL_NUM_COL*14) + "nm^2")
fig_1.savefig("cropped_image.png", dpi=300)


# clustering the pgm of different design via K-means
# number_clusters = int(np.ceil(np.sqrt(CELL_NUM_ROW * CELL_NUM_COL)))
number_clusters = 10
km = KMeans(n_clusters=number_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300000,
            tol=1e-04,
            random_state=0)
# predict the clustering results
y_km = km.fit_predict(cell_lib)


# 计算每个cluster的std
cell_std = np.zeros(CELL_NUM_ROW * CELL_NUM_COL)
cluster_std = []
for i in np.arange(number_clusters):
    cluster_position = np.where(y_km==i)[0]
    subcluster = cell_lib[cluster_position]
    cell_std[cluster_position] = stdImage(subcluster)
    cluster_std.append(stdImage(subcluster))
    print i, len(np.where(y_km==i)[0]), stdImage(subcluster)
cluster_std_array = np.array(cluster_std)
nor_cluster_std_array = ((cluster_std_array - cluster_std_array.min()) / (cluster_std_array.max() - cluster_std_array.min())+0.1)*0.15

# 自动挑选training dataset: randomly pick up AUTO_SELECTION_PERCENT pattern-sets from each clusters
auto_selection_set = []
auto_selection_set_label = []
for cluster_label in np.arange(number_clusters):
    sub_cluster_index = np.where(y_km==cluster_label)[0]
    cluster_percent = nor_cluster_std_array[cluster_label]
    sub_cluster_num = int(np.ceil(cluster_percent*len(sub_cluster_index)))
    randomSel_sub_cluster_index = np.random.choice(sub_cluster_index, sub_cluster_num, replace=False)
    auto_selection_set.extend(randomSel_sub_cluster_index)
    auto_selection_set_label.extend([cluster_label]*sub_cluster_num)


# plot the auto selection set
side = np.ceil(np.sqrt(len(auto_selection_set)))
fig_2=plt.figure(figsize=(2*side, 2*side))
for i in np.arange(len(auto_selection_set)):
    index = auto_selection_set[i]
    plt.subplot(side, side, i+1)
    plt.tight_layout() 
    plt.imshow(cell_lib[index].reshape((CELL_H, CELL_W)),cmap='gray')
    plt.title(str(i) + " Cluster-" + str(auto_selection_set_label[i]))
    plt.axis('off')
fig_2.savefig("auto-selectionset.png", dpi=300)


# plot the auto selection set in the full-GDS:
fig_3 = plt.figure()
ax_3 = fig_3.add_subplot(111)
ax_3.imshow(test_crop_array, cmap='gray') # 先画上full GDS image
fig_3.tight_layout()
ax_3.axis('off')
plt.title("Full-Chip Size: " + str(CELL_H*CELL_NUM_ROW*14) + "x" + str(CELL_W*CELL_NUM_COL*14) + "nm^2")
for i in np.arange(len(auto_selection_set)):
    index = cell_index[auto_selection_set[i]]    
    x = CELL_W*index[1] # rectangle左上角坐标
    y = CELL_H*index[0] # rectangle左上角坐标
    ax_3.add_patch(patches.Rectangle((x, y), CELL_W, CELL_H, linewidth=1, edgecolor='r', facecolor='none'))
    ax_3.text(x+20, y+CELL_H/2, str(i), fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
fig_3.savefig("fullGDS+selectedCell.png", dpi=300)








"""

# print one of the cluster:
label = 2
index_label = np.where(y_km==label)
len_cluster = len(index_label[0])
plt.figure(figsize=(15, 4*np.ceil(len_cluster/8.0)))
for i in np.arange(len_cluster):
    plt.subplot(np.ceil(len_cluster/8.0), 8, i+1)
    plt.tight_layout()   
    index = index_label[0][i]
    plt.imshow(cell_lib[index].reshape((CELL_H, CELL_W)),cmap='gray')
    plt.axis('off')


stdImage(cell_lib[sub_cluster_index])

    """
    
    
