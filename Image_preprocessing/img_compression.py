# _*_coding:utf-8
from PIL import Image
import numpy as np
import cv2


def img_compression(to_compress_img, compressed_height_pixel):
    """
    完成图像压缩
    :param to_compress_img: 待压缩的图像
    :param compressed_height_pixel: 压缩图像对应width pixel
    :return: 压缩后图像
    """

    # 判定是否是Image格式，如果是OpenCV的'BGR'格式则需要转换成Image的'RGB'格式
    if isinstance(to_compress_img, np.ndarray):
        img_to_compress = Image.fromarray(cv2.cvtColor(to_compress_img, cv2.COLOR_BGR2RGB))
    else:
        img_to_compress = to_compress_img
    # img_to_compress = Image.open(input_image_address)
    img_width, img_height = img_to_compress.size

    change_ratio = img_height / compressed_height_pixel
    compressed_width = int(img_width / change_ratio)
    img_to_change = img_to_compress.resize((compressed_width, compressed_height_pixel), Image.ANTIALIAS)
    img_compressed = cv2.cvtColor(np.asarray(img_to_change), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    cv2.imshow('img_compressed', img_compressed)
    cv2.waitKey(0)
    # cv2.imwrite('../Image/img_compressed_{}.jpg'.format(1), img_compressed, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return img_compressed