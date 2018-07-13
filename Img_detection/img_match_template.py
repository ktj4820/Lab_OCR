# _*_coding:utf-8
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt


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


def template_match_many(raw_img, template_img, threshold=0.7):
    img_rgb = raw_img
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = template_img
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # threshold = 0.67
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.namedWindow("img_template", cv2.WINDOW_NORMAL)
    cv2.imshow('img_template', img_rgb)
    cv2.waitKey(0)


def template_match_one(raw_img, template_img):
    img_rgb = raw_img
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # img2 = img.copy()
    template = template_img
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
               'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        # img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img_gray, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img_rgb, top_left, bottom_right, 255, 2)
        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_rgb, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def template_match_shape(raw_img):
    # img_rgb = raw_img
    # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img = cv2.imread("../Image/114.png", 0)
    cv2.namedWindow("raw_img", cv2.WINDOW_NORMAL)
    cv2.imshow('raw_img', img)
    cv2.waitKey(0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    cv2.namedWindow("img_template", cv2.WINDOW_NORMAL)
    cv2.imshow('img_template', cl1)
    cv2.waitKey(0)
    # cv2.imwrite('clahe_2.jpg', cl1)


def hough_circles(img_compressed_input):
    img_gray = cv2.cvtColor(img_compressed_input, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img_gray, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # template = cv2.imread("../Image/121.png", 0)
    #
    # for input_image_mark in range(2605, 2782):
    #     input_image = "D:/Win10/102_FUJI/DSCF" + str(input_image_mark) + ".JPG"
    #     # input_image = "../Image/" + str(input_image_mark) + ".png"
    #     raw_img = cv2.imread(input_image)
    #     img_compressed = img_compression(raw_img, 956)
    #     template_match_many(img_compressed, template, threshold=0.7)

    # input_image = "D:/Win10/102_FUJI/DSCF" + str(2065) + ".JPG"
    # # input_image = "../Image/" + str(input_image_mark) + ".png"
    # raw_img = cv2.imread(input_image)
    # template_match_shape(raw_img)
    input_image = "D:/Win10/102_FUJI/DSCF" + str(2680) + ".JPG"
    # input_image = "../Image/" + str(input_image_mark) + ".png"
    raw_img = cv2.imread(input_image)
    img_compressed = img_compression(raw_img, 956)

    # raw_img = cv2.imread("../Image/114.png", 0)
    # img_compressed = img_compression(raw_img, 956)
    hough_circles(img_compressed)