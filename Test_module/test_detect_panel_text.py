# coding:utf8

import sys
from PIL import Image
import cv2
import numpy as np
# import pytesseract
import argparse
import os
from PIL import ImageEnhance


# from image_recognition_mode import *
# from img_preprocessing import *


def gen_thresh_img(cut_image, threshold, mode='binary'):
    """
    依据图像亮度，进行图像二值化（保留原图像）
    注：为区分红色与灰色，进行列切分时候将不再采用RGB进行切分，而是转换到LAB空间
    :param cut_image:
    :param threshold:
    :param mode:
    :return:
    """
    img = Image.fromarray(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
    print("the format of image has changed from OpenCv to Image")
    # img.show(title='The raw real image to be thresh')
    length_img = img.size[0]
    height_img = img.size[1]
    img2video = img.convert("YCbCr")
    info = []
    for x_raw in range(length_img):
        for y_raw in range(height_img):
            light_video, blue_video, red_video = img2video.getpixel((x_raw, y_raw))

            if light_video < threshold:
                info.append('0, 0, 0')
            else:
                info.append('255, 255, 255')

    gen_img = Image.new("RGB", (length_img, height_img))
    gen_binary_img = Image.new("1", (length_img, height_img))

    for x_gen in range(length_img):
        for y_gen in range(height_img):
            rgb = info[x_gen * height_img + y_gen].split(",")
            gen_img.putpixel([x_gen, y_gen], (int(rgb[0]), int(rgb[1]), int(rgb[2])))
            gen_binary_img.putpixel([x_gen, y_gen], (int(rgb[0])))

    if mode == 'binary':
        # gen_binary_img.show()
        print('return: gen_binary_img')
        return gen_binary_img
    else:
        print('return: gen_img')
        return gen_img


def preprocess(gray):
    # # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
    # input_image = "../Image/6.png"
    # raw_img = cv2.imread(input_image)
    # gen_raw_img = gen_thresh_img(raw_img, threshold=65, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
    # gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    # img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    # _, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图
    #
    # cv2.imshow('The binary row image for detection', thresh_img)
    # cv2.waitKey(0)

    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow('The binary row image for detection', binary)
    cv2.waitKey(0)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element3, iterations=1)
    dilation = cv2.dilate(dilation, element3, iterations=1)
    dilation = cv2.dilate(dilation, element3, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element3, iterations=1)
    erosion = cv2.erode(erosion, element3, iterations=1)
    erosion = cv2.erode(erosion, element3, iterations=1)
    erosion = cv2.erode(erosion, element3, iterations=1)
    erosion = cv2.erode(erosion, element3, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    return erosion


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 500):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print "rect is:"
        print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 10):
            continue
        # if(width >height  * 15):
        #     continue
        if (height <= 0 or width <= 0):
            continue

        region.append(box)

    return region


def image_detection(imagePath, image_output_path):
    img = cv2.imread(imagePath)

    image_usefor_split = img
    image_usefor_draw = img

    # get the original image name
    image_name = os.path.basename(imagePath)
    print("------------------------------------")
    print(image_name)
    print("------------------------------------")

    # 1.  转化成灰度图
    gray = cv2.cvtColor(image_usefor_draw, cv2.COLOR_BGR2GRAY)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    cv2.imshow("dilation", dilation)
    cv2.waitKey(0)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)
    #
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~region~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(region)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~region[0]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(region[0])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~region[0][1]~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(region[0][1])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~region[0][1][0]~~~~~~~~~~~~~~~~~~~~~~~~")
    print(region[0][1][0])
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~region[0][1][0]~~~~~~~~~~~~~~~~~~~~~~~~")

    m = 0
    image_detect = []
    cv2.imshow("cropped", image_usefor_split)
    cv2.waitKey(0)
    for i in region:

        # 计算高和宽
        height = abs(region[m][3][1] - region[m][1][1])
        width = abs(region[m][3][0] - region[m][1][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 10):
            continue

        if (height <= 0 or width <= 0):
            continue

        print(region[m][1][0])

        image_usefor_split = img

        if region[m][1][1] < region[m][3][1] and region[m][1][0] < region[m][3][0]:
            crop_img = image_usefor_split[region[m][1][1]:region[m][3][1], region[m][1][0]:region[m][3][0]]

        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        elif region[m][1][1] < region[m][3][1] and region[m][1][0] > region[m][3][0]:
            crop_img = image_usefor_split[region[m][1][1]:region[m][3][1], region[m][3][0]:region[m][1][0]]
        elif region[m][1][1] > region[m][3][1] and region[m][1][0] < region[m][3][0]:
            crop_img = image_usefor_split[region[m][3][1]:region[m][1][1], region[m][1][0]:region[m][3][0]]
        elif region[m][1][1] > region[m][3][1] and region[m][1][0] > region[m][3][0]:
            crop_img = image_usefor_split[region[m][3][1]:region[m][1][1], region[m][3][0]:region[m][1][0]]

        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~region[m][1][1]:region[m][3][1],region[m][1][0]:region[m][3][0]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # cv2.imshow("cropped", image_usefor_split)
        # cv2.waitKey(0)
        print(region[m][1][1], region[m][3][1], region[m][1][0], region[m][3][0])
        print(
            "~~~~~~~~~~~~~~~~~~~~~~~~~~region[m][1][1]:region[m][3][1],region[m][1][0]:region[m][3][0]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~crop_img~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        m = m + 1
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0)
        image_detect_name = "crop_img" + str(m) + ".png"

        ##对于png图片，第三个参数表示的是压缩级别。

        # ##cv2.IMWRITE_PNG_COMPRESSION，从0到9,压缩级别越高，图像尺寸越小。默认级别为3
        # cv2.imwrite("C:\\Users\\abner\\Desktop\\OCR\\Step_three\\output\\crop_img" + str(m)+".png",
        #             crop_img,[int(cv2.IMWRITE_JPEG_QUALITY), 0])
        # cv2.waitKey(0)
        #
        #
        #
        #
        # image_detect_name="C:\\Users\\abner\\Desktop\\OCR\\Step_three\\output\\crop_img" + str(m)+".png"
        # image_detect.append(image_detect_name)

        # save the rotated iamge
        selected_image_detection_name = '../Img_processed/image_recognition_' + str(m) + '_' + image_name
        print('------------------------edge_detection_image_name----------------------------')
        print(selected_image_detection_name)
        print('------------------------edge_detection_image_name----------------------------')
        cv2.imwrite(selected_image_detection_name, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 0])
        cv2.waitKey(0)

        image_detect.append(selected_image_detection_name)
        # 将切割后的图片写入本地，同时将文件名称保存进image_detect
        if cv2.waitKey(30) & 0xFF == ord('q'):
            continue

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(image_usefor_draw, [box], 0, (0, 255, 0), 2)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", image_usefor_draw)
    cv2.waitKey(0)

    # 带轮廓的图片
    outline_image_path = "../Img_processed/eimage_recognition_" + "image_recognize_zone.png"
    cv2.imwrite(outline_image_path, image_usefor_draw)
    cv2.waitKey(0)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (image_detect, outline_image_path)


image_detection('C:\\Users\\dby_freedom\\Desktop\\Lab_OCR\\Image\\59.png', '..\Image')
