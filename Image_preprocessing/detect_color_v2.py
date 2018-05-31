# coding:utf-8
from PIL import Image
import colorsys
import cv2
import numpy as np


def get_dominant_color(image):
    # 颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

    # 生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))
    # 原来的代码此处为None
    max_score = 0
    dominant_color = 0  # 原来的代码此处为None，但运行出错，改为0以后运行成功，原因在于在下面的 < span
    # style = "font-family:Arial, Helvetica, sans-serif;" > score > max_score的比较中，max_score的初始格式不定 < / span >

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]

        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.9:
            continue

            # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color


def detect_color():
    bright = cv2.imread("../Image/18.JPG")

    bright_lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    bright_ycb = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
    bright_hsv = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)

    # python
    bgr_red = [40, 40, 200]
    bgr_light_blue = [176, 181, 57]
    bgr_yellow = [98, 158, 195]
    bgr_white = [230, 230, 230]
    bgr = bgr_red
    thresh = 40

    min_bgr = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    max_bgr = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    mask_bgr = cv2.inRange(bright, min_bgr, max_bgr)
    result_bgr = cv2.bitwise_and(bright, bright, mask=mask_bgr)

    # convert 1D array to 3D, then convert it to HSV and take the first element
    # this will be same as shown in the above figure [65, 229, 158]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    min_hsv = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
    max_hsv = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

    mask_hsv = cv2.inRange(bright_hsv, min_hsv, max_hsv)
    result_hsv = cv2.bitwise_and(bright_hsv, bright_hsv, mask=mask_hsv)

    # convert 1D array to 3D, then convert it to YCrCb and take the first element
    ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

    min_ycb = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
    max_ycb = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])

    mask_ycb = cv2.inRange(bright_ycb, min_ycb, max_ycb)
    result_ydb = cv2.bitwise_and(bright_ycb, bright_ycb, mask=mask_ycb)

    # convert 1D array to 3D, then convert it to LAB and take the first element
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

    min_lab = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
    max_lab = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

    mask_lab = cv2.inRange(bright_lab, min_lab, max_lab)
    result_lab = cv2.bitwise_and(bright_lab, bright_lab, mask=mask_lab)

    cv2.imshow("Result BGR", result_bgr)
    cv2.waitKey(0)
    cv2.imshow("Result HSV", result_hsv)
    cv2.waitKey(0)
    cv2.imshow("Result YCB", result_ydb)
    cv2.waitKey(0)
    cv2.imshow("Output LAB", result_lab)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detect_img_gray = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
    _, detect_thresh_img = cv2.threshold(detect_img_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Result BGR", detect_img_gray)
    cv2.waitKey(0)
    cv2.imshow("Result HSV", detect_thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detect_thresh_img



def get_gery(img):
    img = Image.open(img)
    length_img = img.size[0]
    height_img = img.size[1]
    img2video = img.convert("YCbCr")
    # img2img = img.convert('RGB')
    info = []
    for x_raw in range(length_img):
        for y_raw in range(height_img):
            light_video, blue_video, red_video = img2video.getpixel((x_raw, y_raw))
            # red_img, green_img, blue_image = img2img.getpixel((x_raw, y_raw))

            if light_video < 65:
                # red_img, green_img, blue_img = img.getpixel((x, y))
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
    gen_img.show()
    # gen_img.save('../Img_processed/%s.png' % int(time.time()))
    gen_img_save_name = 'gen_gray'
    gen_img.save('../Img_processed/%s.png' % gen_img_save_name)

    gen_binary_img.show()
    # gen_img.save('../Img_processed/%s.png' % int(time.time()))
    gen_binary_img_save_name = 'gen_binary'
    gen_img.save('../Img_processed/%s.png' % gen_binary_img_save_name)
    return gen_img_save_name


def gen_thresh_img(cut_image, threshold, mode='binary'):
    """
    依据图像亮度，进行图像二值化（保留原图像）
    :param cut_image:
    :param threshold:
    :param mode:
    :return:
    """
    img = Image.fromarray(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
    # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
        return gen_binary_img
    else:
        return gen_img


if __name__ == '__main__':
    input_image_address = "../Image/18.JPG"
    input_img = cv2.imread(input_image_address)
    # dominant_color = get_dominant_color(img)
    # print(dominant_color)
    # print('The dominant color is: r{},g{},b{}'.format(dominant_color[0], dominant_color[1], dominant_color[2]))

    final_img = gen_thresh_img(input_img, threshold=100)
    final_img.show()
