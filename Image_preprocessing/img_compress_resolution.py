# _*_coding:utf-8
from PIL import Image
import numpy as np
import cv2
import os


def img_compression(input_image, img_width_compressed_pixel):
    """
    完成图像压缩，
    :param input_image:
    :param img_width_compressed_pixel: 输出压缩图像对应width pixel
    :return: 压缩后图像
    """

    # 判定是否是Image格式，如果是OpenCV的'BGR'格式则需要转换成Image的'RGB'格式
    if isinstance(input_image, np.ndarray):
        img_to_compress = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        img_to_compress = input_image
    # img_to_compress = Image.open(input_image_address)
    img_width, img_height = img_to_compress.size

    change_ratio = img_width / img_width_compressed_pixel
    compressed_height = int(img_height / change_ratio)
    img_to_change = img_to_compress.resize((img_width_compressed_pixel, compressed_height), Image.ANTIALIAS)
    img_compressed = cv2.cvtColor(np.asarray(img_to_change), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    cv2.imshow('img_compressed', img_compressed)
    cv2.waitKey(0)
    # cv2.imwrite('../Image/img_compressed_{}.jpg'.format(1), img_compressed, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return img_compressed
