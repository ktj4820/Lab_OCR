# _*_coding:utf-8
from PIL import Image
import numpy as np
import cv2


def identify_panel_text(panel_text_input):
    """
    依据字符列平均字符数量，完成panel字符行鉴定
    :param panel_text_input:
    :return:
    """

    gray = cv2.cvtColor(panel_text_input, cv2.COLOR_BGR2GRAY)    # 转化成灰度图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=1)   # Sobel算子，x方向求梯度
    ret, panel_text_binary_img = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)    # 二值化
    cv2.imshow('The binary row image for detection', panel_text_binary_img)
    cv2.waitKey(0)

    column_white_num = []  # 记录每一列的白色像素总和
    column_black_num = []  # ..........黑色.......

    char_image_height = panel_text_binary_img.shape[0]
    char_width = panel_text_binary_img.shape[1]
    column_white_num_max = 0
    column_black_num_max = 0

    # 计算每一列的黑白色像素总和
    for i in range(char_width):
        white_num = 0  # 这一列白色总数
        black_num = 0  # 这一列黑色总数
        for j in range(char_image_height):
            if panel_text_binary_img[j][i] == 255:
                white_num += 1
            if panel_text_binary_img[j][i] == 0:
                black_num += 1

        column_white_num_max = max(column_white_num_max, white_num)
        column_black_num_max = max(column_black_num_max, black_num)

        column_white_num.append(white_num)  # 记录该列的白色像素总数
        column_black_num.append(black_num)  # 记录该列的黑色像素总数

    # 依据每列平均字符判定是否出现数字行以及出现了几行数字
    avg_column_white = np.average(column_white_num)
    print('avg_column_white:{}'.format(avg_column_white))

    if avg_column_white < 2.45:
        print("This's no text in panel")
    elif avg_column_white < 4.5:
        print("This's one line text in panel")
    else:
        print("This's two lines text in panel")


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


def detect_textbox(raw_img_input):
    """
    完成面板部分文本框切割，面板部分文字切割；返回对应面板部分文本框切割图及文字切割图
    :param raw_img_input:
    :return: textbox, panel_text
    """
    # 分割图像，提取字符行

    def find_real_start_row(real_image_height):
        find_real_start_row_mark = 5
        while find_real_start_row_mark < real_image_height - 2:
            find_real_start_row_mark += 1
            '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
            即将该行看做出现字符的起始行。
            对于白底黑字情况，则反之'''
            if (row_white_num[real_image_height - find_real_start_row_mark] if arg else
                row_black_num[real_image_height - find_real_start_row_mark]) >= \
                    (0.65 * row_white_num_max if arg else 0.85 * row_black_num_max) and \
                    (row_white_num[real_image_height - find_real_start_row_mark - 1] <= 0.5 * row_white_num_max):

                real_start_row = real_image_height - find_real_start_row_mark

                return real_start_row

    def find_real_height(initial_real_image_height):
        find_real_end_row_mark = 0
        initial_image_height = initial_real_image_height
        while find_real_end_row_mark < initial_image_height - 2:
            find_real_end_row_mark += 1
            '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
            即将该行看做出现字符的起始行。
            对于白底黑字情况，则反之'''
            if (row_white_num[initial_image_height-find_real_end_row_mark] if arg else
                row_black_num[find_real_end_row_mark]) > (0.5 * row_white_num_max if arg else 0.8 * row_black_num_max) \
                    and row_white_num[initial_image_height - find_real_end_row_mark - 1] <= 0.5 * row_white_num_max:

                real_row_end = initial_image_height-find_real_end_row_mark

                return real_row_end

    def find_real_start_column(initial_real_image_width):
        find_real_start_column_mark = 0
        while find_real_start_column_mark < initial_real_image_width - 2:
            find_real_start_column_mark += 1
            '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
            即将该行看做出现字符的起始行。
            对于白底黑字情况，则反之'''
            if (column_white_num[find_real_start_column_mark] if arg else row_black_num[find_real_start_column_mark]) \
                    >= (0.7 * column_white_num_max if arg else 0.7 * column_black_num_max) and \
                    (column_white_num[find_real_start_column_mark + 1] <= 0.3 * column_white_num_max):

                final_real_start_column = find_real_start_column_mark
                return final_real_start_column

    def find_real_end_column(initial_real_image_width):
        find_real_end_row_mark = 0
        # initial_image_height = initial_real_image_width
        while find_real_end_row_mark < initial_real_image_width - 2:
            find_real_end_row_mark += 1
            '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
            即将该行看做出现字符的起始行。
            对于白底黑字情况，则反之'''
            if (column_white_num[initial_real_image_width - find_real_end_row_mark]) >= 0.55 * column_white_num_max \
                    and column_white_num[initial_real_image_width - find_real_end_row_mark - 1] <= \
                    0.45 * column_white_num_max:

                real_column_end = initial_real_image_width - find_real_end_row_mark

                return real_column_end

    def find_instrument_panel_text(initial_row_split_start, initial_real_start_column):

        panel_text_start_row = initial_row_split_start - 35
        panel_text_end_row = initial_row_split_start - 5
        panel_text_start_column = initial_real_start_column - 20
        panel_text_end_column = initial_real_start_column + 15

        return panel_text_start_row, panel_text_end_row, panel_text_start_column, panel_text_end_column

    # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
    gen_raw_img = gen_thresh_img(raw_img_input, threshold=50, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
    gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

    cv2.imshow('The binary row image for detection', thresh_img)
    cv2.waitKey(0)

    # thresh_img = cv2.cvtColor(np.asarray(thresh_img), cv2.COLOR_RGB2BGR)
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
        row_white_num.append(white_num)    # 记录该行的白色像素总数
        row_black_num.append(black_num)    # 记录该行的黑色像素总数

    arg = True
    '''arg = True表示黑底白字, arg = False表示白底黑字'''
    if row_black_num_max < row_white_num_max:
        arg = False

    # 寻找文本框起始行与结束行
    real_image_height = find_real_height(image_height)
    img_textbox_start = find_real_start_row(real_image_height)

    while img_textbox_start < image_height - 2:
        img_textbox_start += 1
        row_find_end = real_image_height
        # 因为行间隔比较大，故而在此取检测出现字符行的前面的第3行作为行切割开始行（保证将框切割进去）
        row_split_start = img_textbox_start - 3 if img_textbox_start > 3 else img_textbox_start
        row_split_end = row_find_end + 3 if row_find_end + 3 < image_height else row_find_end

        # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
        raw_image_split_row = raw_img_input[row_split_start:row_split_end, 0:width]
        cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(img_textbox_start - 1),
                    raw_image_split_row, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        '''
        得到行切割图像，直接对原图像进行颜色判定，进而完成选定亮度阈值，最后得到二值图，显示并保存
        '''
        binary_img = detect_row_img_color(raw_image_split_row)
        binary_img.save('../Img_processed/binary_row_{}.jpg'.format(img_textbox_start - 1))
        '''
        此处，不在将分行处理结果转变为二值图，而是转变为只有0与255的RGB图，以便于进行字符切割中的Image到OpenCV转换
        下面正式开始字符切割：
        '''

        # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
        char_gen_img = cv2.cvtColor(np.asarray(binary_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
        cv2.imshow('char_gen_img', char_gen_img)
        cv2.waitKey(0)
        # print('Complete the format of char_split method from Image to OpenCV')
        char_img_gray = cv2.cvtColor(char_gen_img, cv2.COLOR_BGR2GRAY)
        _, char_thresh_img = cv2.threshold(char_img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

        cv2.imshow('char_thresh_img', char_thresh_img)
        cv2.waitKey(0)

        column_white_num = []  # 记录每一列的白色像素总和
        column_black_num = []  # ..........黑色.......

        char_image_height = char_thresh_img.shape[0]
        char_width = char_thresh_img.shape[1]
        column_white_num_max = 0
        column_black_num_max = 0

        # 计算每一列的黑白色像素总和
        for i in range(char_width):
            white_num = 0  # 这一列白色总数
            black_num = 0  # 这一列黑色总数
            for j in range(char_image_height):
                if char_thresh_img[j][i] == 255:
                    white_num += 1
                if char_thresh_img[j][i] == 0:
                    black_num += 1

            column_white_num_max = max(column_white_num_max, white_num)
            column_black_num_max = max(column_black_num_max, black_num)

            column_white_num.append(white_num)  # 记录该列的白色像素总数
            column_black_num.append(black_num)  # 记录该列的黑色像素总数

        # 寻找文本框起始行与结束行
        real_start_column = find_real_start_column(char_width)
        real_end_column = find_real_end_column(char_width)

        real_start_column -= 3 if real_start_column - 3 >= 0 else real_start_column
        real_end_column += 3 if real_end_column + 3 <= char_width else real_end_column

        textbox = char_thresh_img[0:char_image_height, real_start_column:real_end_column]

        cv2.imshow('raw_image_split_char', textbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('../Img_processed/textbox_{}.jpg'.format(real_start_column - 1),
                    textbox, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        print("textbox coordinates: ")
        print("upper left coordinate {}, lower left coordinate {},".format((
            row_split_start, real_start_column), (row_split_end, real_start_column)))
        print(" top right coordinate {}, lower right coordinate {}".format((
            row_split_start, real_end_column), (row_split_end, real_end_column)))

        final_panel_text_start_row, final_panel_text_end_row, final_panel_text_start_column, \
            final_panel_text_end_column = find_instrument_panel_text(row_split_start, real_start_column)

        panel_text = raw_img_input[
                     final_panel_text_start_row:final_panel_text_end_row,
                     final_panel_text_start_column:final_panel_text_end_column
                     ]

        cv2.imshow('panel_text', panel_text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('../Img_processed/panel_text_{}.jpg'.format(final_panel_text_start_row - 1),
                    panel_text, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return textbox, panel_text


def detect_row_img_color(image):
    """
    鉴定切割行图片的颜色，返回对应最佳二值图
    通过判定颜色，进行不同亮度阈值设定，最终得到最佳二值图
    :param image:
    :return:
    """

    img = image
    # python
    bgr_red = [40, 40, 180]  # 红色判定基准
    bgr_light_blue = [176, 181, 57]  # 淡蓝判定基准
    bgr_yellow = [98, 158, 195]  # 黄色判定基准
    bgr_white = [195, 195, 195]  # 白色判定基准

    def detect_img_submodule(submodule_img, color_to_check, thresh_red=40):

        bright = submodule_img
        detect_img_bgr = color_to_check
        detect_img_thresh = 40  # 上下范围扩展40
        min_bgr = np.array([detect_img_bgr[0] - detect_img_thresh, detect_img_bgr[1] - detect_img_thresh,
                            detect_img_bgr[2] - thresh_red])
        max_bgr = np.array([detect_img_bgr[0] + detect_img_thresh, detect_img_bgr[1] + detect_img_thresh,
                            detect_img_bgr[2] + thresh_red])

        mask_bgr = cv2.inRange(bright, min_bgr, max_bgr)
        result_bgr = cv2.bitwise_and(bright, bright, mask=mask_bgr)

        # cv2.imshow("filter color image", result_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result_bgr

    def check_color(raw_img_input, decision_threshold=0.01):
        """
        依据图片内白色（字体）像素点占总像素点的比率，判定所截图片是否对应所判定颜色
        :param raw_img_input:
        :param decision_threshold:
        :return:
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
        if (total_white_num if arg else total_black_num) > \
                (total_black_num * decision_threshold if arg else total_white_num):
            print("The check color is right")
            return True
        else:
            print("The check color is wrong")
            return False

    # 判定红色字体
    if check_color(detect_img_submodule(img, bgr_red, thresh_red=60)):
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
        final_binary_img = gen_thresh_img(img, threshold=125, mode='dynamic')

    elif check_color(detect_img_submodule(img, bgr_white), decision_threshold=0.0065):
        print('This row image is light white')
        # 依照白色字体对应亮度进行二值化处理
        final_binary_img = gen_thresh_img(img, threshold=135, mode='dynamic')

    else:
        print('The color of this row image out of index')
        # 对其他情况，对应亮度进行二值化处理，取一个尽量低的值保证所有字体显现
        final_binary_img = gen_thresh_img(img, threshold=130, mode='dynamic')

    return final_binary_img


if __name__ == '__main__':

    input_image = "../Image/73.png"
    raw_img = cv2.imread(input_image)
    textbox, panel_text = detect_textbox(raw_img)
    identify_panel_text(panel_text)
