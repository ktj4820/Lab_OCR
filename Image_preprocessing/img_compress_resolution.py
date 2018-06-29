# _*_coding:utf-8
from PIL import Image
import numpy as np
import cv2
import os


def img_compression(input_image_address):
    img_to_compress = Image.open(input_image_address)
    img_width, img_height = img_to_compress.size
    change_ratio = img_width / 3000
    compressed_height = int(img_height / change_ratio)
    img_to_change = img_to_compress.resize((3000, compressed_height), Image.ANTIALIAS)
    img_compressed = cv2.cvtColor(np.asarray(img_to_change), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    cv2.imshow('img_compressed', img_compressed)
    cv2.waitKey(0)
    cv2.imwrite('../Image/img_compressed_{}.jpg'.format(1), img_compressed, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return img_to_change