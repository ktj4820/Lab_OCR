# coding:utf-8
from PIL import Image
import numpy as np
import cv2


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


def detect_row_img_color(image):
    """
    鉴定切割行图片的颜色，返回对应最佳二值图
    通过判定颜色，进行不同亮度阈值设定，最终得到最佳二值图
    :param image:
    :return:
    """

    img = image
    # python
    bgr_red = [40, 40, 200]  # 红色判定基准
    bgr_light_blue = [176, 181, 57]  # 淡蓝判定基准
    bgr_yellow = [98, 158, 195]  # 黄色判定基准
    bgr_white = [225, 225, 225]  # 白色判定基准

    def detect_img_submodule(submodule_img, color_to_check):

        bright = submodule_img
        img_detect_bgr = color_to_check
        img_detect_thresh = 40  # 上下范围扩展40
        min_bgr = np.array([img_detect_bgr[0] - img_detect_thresh, img_detect_bgr[1] - img_detect_thresh,
                            img_detect_bgr[2] - img_detect_thresh])
        max_bgr = np.array([img_detect_bgr[0] + img_detect_thresh, img_detect_bgr[1] + img_detect_thresh,
                            img_detect_bgr[2] + img_detect_thresh])

        mask_bgr = cv2.inRange(bright, min_bgr, max_bgr)
        result_bgr = cv2.bitwise_and(bright, bright, mask=mask_bgr)

        # cv2.imshow("filter color image", result_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result_bgr

    def check_color(raw_img_input):
        """依据图片内白色（字体）像素点占总像素点的比率，判定所截图片是否对应所判定颜色
        :param raw_img_input:
        :return: final_binary_img
        """
        # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
        gen_raw_img = gen_thresh_img(raw_img_input, threshold=65, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
        gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
        img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

        row_white_num = []  # 记录每一行的白色像素总和
        row_black_num = []  # ..........黑色.......
        image_height = thresh_img.shape[0]
        width = thresh_img.shape[1]
        row_white_num_max = 0
        row_black_num_max = 0
        # 计算每一行的黑白色像素总和
        for i in range(image_height):
            white_num = 0  # 这一行白色总数
            black_num = 0  # 这一行黑色总数
            for j in range(width):
                if thresh_img[i][j] == 255:
                    white_num += 1
                if thresh_img[i][j] == 0:
                    black_num += 1
            row_white_num_max = max(row_white_num_max, white_num)
            row_black_num_max = max(row_black_num_max, black_num)
            row_white_num.append(white_num)  # 记录该行的白色像素总数
            row_black_num.append(black_num)  # 记录该行的黑色像素总数
        total_white_num = sum(row_white_num)
        total_black_num = sum(row_black_num)

        arg = True
        '''arg = True表示黑底白字, arg = False表示白底黑字'''
        if total_black_num < total_white_num:
            arg = False
        # if (row_white_num[row_mark] if arg else row_black_num[row_mark])
        # 因为前面使用的是二值化方法是cv2.THRESH_BINARY，二值图表现为黑底白字
        if (total_white_num if arg else total_black_num) > (total_black_num * 0.01 if arg else total_white_num):
            print("The check color is right")
            return True
        else:
            print("The check color is wrong")
            return False

    # 判定红色字体
    if check_color(detect_img_submodule(img, bgr_red)):
        print('This row image is red')
        # 依照红色字体对应亮度进行二值化处理
        # 因为传入的是切割过后的行图像
        # 为了区分红色与灰色，不能再直接传原图，而是传转换到LAB空间处理过的图
        cv2.imshow("the raw red img input", img)
        cv2.waitKey(0)

        red_img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # convert 1D array to 3D, then convert it to LAB and take the first element

        # python
        bgr_red = [40, 40, 200]
        bgr = bgr_red
        thresh = 40

        lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

        min_lab = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
        max_lab = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

        mask_lab = cv2.inRange(red_img_lab, min_lab, max_lab)
        filter_for_red_img = cv2.bitwise_and(red_img_lab, red_img_lab, mask=mask_lab)

        cv2.imshow("the changed red img input", filter_for_red_img)
        cv2.waitKey(0)
        final_binary_img = gen_thresh_img(filter_for_red_img, threshold=65, mode='dynamic')

    elif check_color(detect_img_submodule(img, bgr_light_blue)):
        print('This row image is light blue')
        # 依照淡蓝字体对应亮度进行二值化处理
        final_binary_img = gen_thresh_img(img, threshold=100, mode='dynamic')

    elif check_color(detect_img_submodule(img, bgr_yellow)):
        print('This row image is light yellow')
        # 依照黄色字体对应亮度进行二值化处理
        final_binary_img = gen_thresh_img(img, threshold=130, mode='dynamic')

    elif check_color(detect_img_submodule(img, bgr_white)):
        print('This row image is light white')
        # 依照白色字体对应亮度进行二值化处理
        final_binary_img = gen_thresh_img(img, threshold=160, mode='dynamic')

    else:
        print('The color of this row image out of index')
        # 对其他情况，对应亮度进行二值化处理，取一个尽量低的值保证所有字体显现
        final_binary_img = gen_thresh_img(img, threshold=130, mode='dynamic')

    return final_binary_img
