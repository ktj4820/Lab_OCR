#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 23:14:28 2018
@author: kimmeen
"""

import cv2
import func
import numpy as np
import tensorflow as tf

# 1、读取图像，并把图像转换为灰度图像并显示
img = cv2.imread("C:\\Users\\dby_freedom\\Desktop\\661\\Image\\6.jpg")  # 读取图片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)

# 2、将灰度图像二值化，设定阈值是100, 根据实际情况调节，然后腐蚀膨胀
img_thre = img_gray
cv2.threshold(img_gray, 165, 255, cv2.THRESH_BINARY_INV, img_thre)
# cv2.imshow('threshold', img_thre)
# cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)
img_thre = cv2.morphologyEx(img_thre, cv2.MORPH_OPEN, kernel)

# 3、保存处理后的图片
cv2.imwrite('thre_res.png', img_thre)

# 4、分割字符操作
white = []  # 记录每一列的白色像素总和
black = []  # ..........黑色.......
height = img_thre.shape[0]
width = img_thre.shape[1]
white_max = 0
black_max = 0
# 计算每一列的黑白色像素总和
for i in range(width):
    s = 0  # 这一列白色总数
    t = 0  # 这一列黑色总数
    for j in range(height):
        if img_thre[j][i] == 255:
            s += 1
        if img_thre[j][i] == 0:
            t += 1
    white_max = max(white_max, s)
    black_max = max(black_max, t)
    white.append(s)
    black.append(t)
    # print(s)
    # print(t)

arg = False  # False表示白底黑字；True表示黑底白字
if black_max > white_max:
    arg = True


# 分割图像
def find_end(start_):
    end_ = start_ + 1
    for m in range(start_ + 1, width - 1):
        if (black[m] if arg else white[m]) > (
        0.99 * black_max if arg else 0.99 * white_max):  # 0.95这个参数请多调整，对应下面的0.05
            end_ = m
            break
    return end_


#########################前向网络##########################################
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28, 28, 1])  # 28x28
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ##
W_conv1 = func.weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
b_conv1 = func.bias_variable([32])
h_conv1 = tf.nn.relu(func.conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
h_pool1 = func.max_pool_2x2(h_conv1)  # output size 14x14x32

## conv2 layer ##
W_conv2 = func.weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
b_conv2 = func.bias_variable([64])
h_conv2 = tf.nn.relu(func.conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = func.max_pool_2x2(h_conv2)  # output size 7x7x64

## fc1 layer ##
W_fc1 = func.weight_variable([7 * 7 * 64, 1024])
b_fc1 = func.bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

## fc2 layer ##
W_fc2 = func.weight_variable([1024, 10])
b_fc2 = func.bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

# image = cv2.imread(image_path, 0)
# image = np.reshape(image, [1, 28, 28, 1])

#########################进行预测##########################################

sess = tf.Session()
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./model/')
saver.restore(sess, ckpt.model_checkpoint_path)
print("\nModel has been restored.\n")

final_result = []
n = 1
start = 1
end = 2
while n < width - 2:
    n += 1
    if (white[n] if arg else black[n]) > (0.01 * white_max if arg else 0.01 * black_max):
        # 上面这些判断用来辨别是白底黑字还是黑底白字
        # 0.05这个参数请多调整，对应上面的0.95
        start = n
        end = find_end(start)
        n = end
        if end - start > 5:
            cj = img_thre[1:(height + 20), (start - 70):(end + 70)]  # 调节裁剪区域
            cj = cv2.resize(cj, (28, 28), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow('caijian', cj)
            # cv2.waitKey(0)
            cj = np.reshape(cj, [1, 28, 28, 1])
            result = sess.run(prediction, feed_dict={xs: cj})
            max_index = np.argmax(result)
            final_result.append(max_index)

string = ','.join(str(i) for i in final_result)
print("The result is:", string)