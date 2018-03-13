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
2018-03-07:
    14 处理坐标问题，use指定input PGM的坐标，可以返回crop PGM的坐标
    15 做一个特别的例子， 先备注了，结果表明不太好
2018-03-08:
    16 输出每个cell的坐标
    17 写了计算每个图片之间的standard deviation的函数
    18 把分拆后的cell_lib里面每一张image去计算两两之间std
2018-03-09:
    19 test了一下image-image之间的std，不太work
2018-03-12:
    20 重写一下代码，clean一下
2018-03-13:
    21 重写一下代码，clean一下
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""
定义初始化变量
"""
# PGM size:
# 6180x6180 pixel (14nm pixel size), 包括Cell Window外约为3.2um的外扩boundary，此boundary需要剔除
# 剔除boundary后约为 80x80 um^2 --> ~5700x5700 pixel
DEFAULT_IMAGE = r"C:\Users\vincchen\Documents\1_Assignment\184-Newron_UMC-L14\data\c0displaymi.pgm"
PIXEL_SIZE = 0.014 # 14nm的pixel size
ELIMINATED_MARGIN = np.ceil(3.2/PIXEL_SIZE)
CELL_H = 200 # cropped cell height
CELL_W = 200 # cropped cell width

Y_0 = 1963  # 整张PGM左上角的Y坐标，单位um
X_0 = 1976  # 整张PGM左上角的X坐标，单位um
INITIAL_COORDINATE= (Y_0, X_0) # 整张GDS左上角的坐标，单位um

CONVOLUTION_FILTER_SIZE = 10 #把一个cropped的image desample，传递给后面的function算std
# AUTO_SELECTION_PERCENT = 0.1 # randomly pick up 10% of patterns from each clusters
CONVOLUTION_FILTER = np.ones((CONVOLUTION_FILTER_SIZE, CONVOLUTION_FILTER_SIZE))/np.power(CONVOLUTION_FILTER_SIZE, 2)


"""
定义函数
"""
# 把PGM转为array，并且默认做了归一化到[0, 1]范围的归一化处理
def PGM5toARRAY(input_pgm, normalization=True):
    pgm_image = Image.open(input_pgm) # 读取PGM
    pgm_array = np.array(pgm_image, np.float32) # convert pgm to array
    pgm_array = pgm_array[::-1] # 读入的pgm上下是颠倒的，所以反序一下
    if normalization:
        pgm_array = (pgm_array - pgm_array.min()) / (pgm_array.max() - pgm_array.min()) # 归一化到0-1
    return pgm_array

# 把一张完整的PGM，剔除四周的margin（ELIMINATED_MARGIN）
# 然后基于CELL_H， CELL_W的大小，基于剩余的PGM计算能crop出来的行列数目
# 最后输出每一个crop出的cell，对应的行列index，已经左上角坐标
def crop_array(input_array, input_array_coordinate, cell_h, cell_w):
    input_array_H, input_array_W = input_array.shape
    if (input_array_H >= cell_h + 2*ELIMINATED_MARGIN) and (input_array_W >= cell_w + 2*ELIMINATED_MARGIN):        
        crop_h = input_array_W - 2*ELIMINATED_MARGIN
        crop_w = input_array_H - 2*ELIMINATED_MARGIN
        num_cell_row = np.int8(np.floor(crop_h / cell_h))
        num_cell_col = np.int8(np.floor(crop_w / cell_w))
        crop_array = input_array[ELIMINATED_MARGIN : ELIMINATED_MARGIN + cell_h * num_cell_row,
                                 ELIMINATED_MARGIN : ELIMINATED_MARGIN + cell_w * num_cell_col]
        crop_y0 = input_array_coordinate[0] - ELIMINATED_MARGIN*PIXEL_SIZE
        crop_x0 = input_array_coordinate[1] - ELIMINATED_MARGIN*PIXEL_SIZE 
        crop_array_shape = (num_cell_row, num_cell_col)
        crop_array_coordinate = (crop_y0, crop_x0)
        return crop_array, crop_array_shape, crop_array_coordinate
    else:
        print "Faile to crop input_array as its size is too small"

# 把一张PGM按大小为（cell_H， cell_W）去切割成crop_array_shape的小clip
def cellLibrary(input_array, input_array_coordinate, cell_H, cell_W, cell_num_row, cell_num_col):
    cell_image = None
    cell_index = []
    cell_coordinate = []
    cell_row_index = np.arange(cell_num_row)
    cell_col_index = np.arange(cell_num_row)
    # 按序crop出image并flatten了image后append到cell_image这个list里面
    # 把每个clip对于的row, colunm保存到cell_index这个list里面
    # 把每个clip的左上角坐标保存到cell_coordinate这个list里面
    for row in cell_row_index:
        for col in cell_col_index:
            append_array = input_array[row*cell_H : (row+1)*cell_H, 
                                      col*cell_W : (col+1)*cell_W].flatten()
            if cell_image is None:
                cell_image = append_array
            else:
                cell_image = np.row_stack((cell_image, append_array))
            cell_index.append([row, col]) 
            cell_coordinate.append((input_array_coordinate[0] + row*cell_H*PIXEL_SIZE,
                                    input_array_coordinate[1] + col*cell_W*PIXEL_SIZE))
    return cell_image, cell_index, cell_coordinate

# 用一个平均数filter去卷积image
def convolveImgage(input_array, convolve_filter=CONVOLUTION_FILTER):
    convolve_array = signal.fftconvolve(input_array, CONVOLUTION_FILTER, mode='same')
    return convolve_array

# 输入的img是2D的，flatten后计算两个flatten images之间的std
def stdBetweenImage(array_1, array_2):
    if array_1.shape == array_2.shape:              
        array_1_flatten = array_1.flatten()
        array_2_flatten = array_2.flatten()
        num_pixel = len(array_1_flatten)
        std_perPixel = np.std(np.vstack((array_1_flatten, array_2_flatten)), axis=0)
        std_sumPixel = np.sum(std_perPixel)
        std_avePixel = std_sumPixel / num_pixel
        return std_perPixel.reshape(array_1.shape), std_sumPixel, std_avePixel
    else:
        print("Fail: img_1 shape is different from img_2 shape")

# 计算每个cluster下的所有图片的standard deviation
def stdImage(cluster_array):
    num_sample = cluster_array.shape[0]
    sub_image = np.zeros((num_sample, CELL_H*CELL_W))
    for n in np.arange(num_sample):
        sub_image[n, :] = convolveImgage(cluster_array[n].reshape((CELL_H, CELL_W))).flatten()  
    cluster_std = np.sum(np.std(sub_image, axis=0)) / (CELL_H*CELL_W)
    return cluster_std


"""
定义main flow
"""
# 读取并生成test dataset
test_array = PGM5toARRAY(DEFAULT_IMAGE, normalization=True)
test_crop_array, test_crop_array_shape, test_crop_coordinate = crop_array(input_array = test_array,
                                                                          input_array_coordinate = INITIAL_COORDINATE,
                                                                          cell_h = CELL_H, cell_w = CELL_W)
cell_lib, cell_index, cell_coordinate = cellLibrary(test_crop_array, input_array_coordinate = test_crop_coordinate,
                                                    cell_H = CELL_H, cell_W = CELL_W,
                                                    cell_num_row = test_crop_array_shape[0], cell_num_col = test_crop_array_shape[1])
# plot and save cropped image:
fig_1 = plt.figure()
ax_1 =fig_1.add_subplot(111)
ax_1.imshow(test_crop_array, cmap='gray')
ax_1.axis('off')
fig_1.tight_layout()
plt.title("Full-Chip Size: " + str(CELL_H*test_crop_array_shape[0]*14) + "x" + str(CELL_W*test_crop_array_shape[1]*14) + "nm^2")
fig_1.savefig("1 cropped_image.png", dpi=300)


# demo 演示不同cell image的std
demo_data_index_list = [0, 1, 5, 8, 10, 11]
demo_data = []
demo_data_convolve = []
for i in demo_data_index_list:
    temp_data = cell_lib[i].reshape((CELL_H, CELL_W))
    demo_data.append(temp_data)
    demo_data_convolve.append(convolveImgage(temp_data))
fig_2 = plt.figure(figsize=(20, 9))
for i in np.arange(len(demo_data)):
    fig_2.add_subplot(3, 6, i+1)
    plt.imshow(demo_data[i], cmap='gray')
    plt.axis('off')
    plt.title('cell ' + str(i))
    fig_2.add_subplot(3, 6, i+7)
    plt.imshow(demo_data_convolve[i], cmap='gray')
    plt.axis('off')
    plt.title('convolve cell ' + str(i))
std_pair_index = [(0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (2, 4)]
std_perPixel_pair = []
std_sumPixel_pair = []
std_avePixel_pair = []
for i, index in enumerate(std_pair_index):
    img_1 = demo_data_convolve[index[0]]
    img_2 = demo_data_convolve[index[1]]
    temp_std_perPixel, temp_std_sumPixel, temp_std_avePixel = stdBetweenImage(img_1, img_2)
    std_perPixel_pair.append(temp_std_perPixel)
    std_sumPixel_pair.append(temp_std_sumPixel)
    std_avePixel_pair.append(temp_std_avePixel)
    fig_2.add_subplot(3, 6, i+13)
    plt.imshow(std_perPixel_pair[i], cmap='gray')
    plt.axis('off')
    plt.title(str(index[0])+'-'+str(index[1]) + ' aveStd: ' + str(round(std_avePixel_pair[i], 4)))
fig_2.savefig("2 demo_stdComparison.png", dpi=300)   


# clustering the pgm of different design via K-means
cell_num_row = int(test_crop_array_shape[0])
cell_num_col = int(test_crop_array_shape[1])
number_clusters = int(np.ceil(np.sqrt(cell_num_row * cell_num_col)))
km = KMeans(n_clusters=number_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300000,
            tol=1e-04,
            random_state=0)
# predict the clustering results
cell_km = km.fit_predict(cell_lib)

# 计算每个cluster的std
cell_std = np.zeros(cell_num_row * cell_num_col) # 保存每个cell的std
cluster_std = [] # 保存每个cluster的std
for i in np.arange(number_clusters):
    cluster_position = np.where(cell_km==i)[0] # 提取cluster label=i的index position
    sub_cluster = cell_lib[cluster_position] # 提取上述index的cell
    cell_std[cluster_position] = stdImage(sub_cluster) # 把每个cluster的std放入每个cell
    cluster_std.append(stdImage(sub_cluster)) # 用一个list保存所有cluster的std
    print i, len(np.where(cell_km==i)[0]), stdImage(sub_cluster) # 打印每个cluster的样品数目及std
cluster_std_array = np.array(cluster_std) # convert to array
re_cluster_std_array = (cluster_std_array - cluster_std_array.min()) / (cluster_std_array.max() - cluster_std_array.min())
cluster_pickup_percent = (re_cluster_std_array + 0.05) * 0.1

# 自动挑选training dataset: randomly pick up AUTO_SELECTION_PERCENT pattern-sets from each clusters
auto_selection_set = []
auto_selection_set_label = []
for cluster_label in np.arange(number_clusters):
    sub_cluster_index = np.where(cell_km==cluster_label)[0]
    numSample = len(sub_cluster_index)
    cluster_percent = cluster_pickup_percent[cluster_label]
    sub_cluster_num = int(np.ceil(cluster_percent*numSample))
    randomSel_sub_cluster_index = np.random.choice(sub_cluster_index, sub_cluster_num, replace=False)
    auto_selection_set.extend(randomSel_sub_cluster_index)
    auto_selection_set_label.extend([cluster_label]*sub_cluster_num)

# plot the auto selection set
side = np.ceil(np.sqrt(len(auto_selection_set)))
fig_3=plt.figure(figsize=(2*side, 2*side))
for i in np.arange(len(auto_selection_set)):
    index = auto_selection_set[i]
    plt.subplot(side, side, i+1)
    plt.tight_layout() 
    plt.imshow(cell_lib[index].reshape((CELL_H, CELL_W)),cmap='gray')
    plt.title(str(i) + " Cluster-" + str(auto_selection_set_label[i]))
    plt.axis('off')
fig_3.savefig("3 auto-selectionset.png", dpi=300)

# plot the auto selection set in the full-GDS:
fig_4 = plt.figure()
ax_4 = fig_4.add_subplot(111)
ax_4.imshow(test_crop_array, cmap='gray') # 先画上full GDS image
# 画上切割线
plt.hlines(np.arange(0, cell_num_row*CELL_H, CELL_H), 0, cell_num_col*CELL_W, linestyles='--', colors='blue') 
plt.vlines(np.arange(0, cell_num_col*CELL_W, CELL_W), 0, cell_num_row*CELL_H, linestyles='--', colors='blue')
fig_4.tight_layout()
ax_4.axis('off')
plt.title("Full-Chip Size: " + str(CELL_H*test_crop_array_shape[0]*14) + "x" + str(CELL_W*test_crop_array_shape[1]*14) + "nm^2")
for i in np.arange(len(auto_selection_set)):
    index = cell_index[auto_selection_set[i]]    
    x = CELL_W*index[1] # rectangle左上角坐标
    y = CELL_H*index[0] # rectangle左上角坐标
    ax_4.add_patch(patches.Rectangle((x, y), CELL_W, CELL_H, linewidth=2, edgecolor='r', facecolor='none'))
    ax_4.text(x+20, y+CELL_H/2, str(i), fontsize=5, bbox=dict(facecolor='red', alpha=0.5))
fig_4.savefig("4 fullGDS+selectedCell.png", dpi=300)


