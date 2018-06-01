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
    # img = Image.fromarray(cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB))
    # print("the format of image has changed from OpenCv to Image")
    # img.show(title='The raw real image to be thresh')
    # length_img = img.size[0]
    # height_img = img.size[1]
    # img2video = img.convert("YCbCr")
    # info = []
    # for x_raw in range(length_img):
    #     for y_raw in range(height_img):
    #         light_video, blue_video, red_video = img2video.getpixel((x_raw, y_raw))
    #
    #         if light_video < threshold:
    #             info.append('0, 0, 0')
    #         else:
    #             info.append('255, 255, 255')
    #
    # gen_img = Image.new("RGB", (length_img, height_img))
    # gen_binary_img = Image.new("1", (length_img, height_img))
    #
    # for x_gen in range(length_img):
    #     for y_gen in range(height_img):
    #         rgb = info[x_gen * height_img + y_gen].split(",")
    #         gen_img.putpixel([x_gen, y_gen], (int(rgb[0]), int(rgb[1]), int(rgb[2])))
    #         gen_binary_img.putpixel([x_gen, y_gen], (int(rgb[0])))
    #
    # gen_binary_img.show()
    # # gen_binary_img_save_name = 'gen_binary'
    # # gen_binary_img.save('../Img_processed/%s.png' % gen_binary_img_save_name)
    # # final_gen_binary_img = cv2.cvtColor(np.asarray(gen_binary_img), cv2.COLOR_RGB2BGR)
    # return gen_binary_img


def row_split(raw_img_input):
    # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
    gen_raw_img = gen_thresh_img(raw_img_input, threshold=65, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
    gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图
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

    # 分割图像，提取字符行
    def find_end(start_row):
        end_ = start_row + 1
        '''arg = True，黑底白字情况下：对于第m行，如果该行黑色像素大于
        0.95*（所有行中黑色像素数目和的最大值），则证明该列包含的白色字符太少，判定次列即为字符切割结束列；
        对于白底黑字情况，则反之'''
        for m in range(start_row + 1, image_height + 1):
            if (row_black_num[m] if arg else row_white_num[m]) > \
                    (0.95 * row_black_num_max if arg else 0.95 * row_white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    arg = True
    '''arg = True表示黑底白字, arg = False表示白底黑字'''
    if row_black_num_max < row_white_num_max:
        arg = False

    row_mark = 1

    while row_mark < image_height - 2:
        row_mark += 1
        '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
        即将该行看做出现字符的起始行。
        对于白底黑字情况，则反之'''
        if (row_white_num[row_mark] if arg else row_black_num[row_mark]) > \
                (0.05 * row_white_num_max if arg else 0.05 * row_black_num_max):
            row_split_start = row_mark - 5 if row_mark > 5 else row_mark
            row_find_end = find_end(row_split_start)
            row_split_end = row_find_end + 5 if row_find_end + 5 < image_height else row_find_end
            row_mark = row_split_end
            if row_find_end - row_split_start > 5:  # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰
                # 对上述二值图完成行切割、显示、保存
                # image_split_row = thresh_img[row_split_start:row_split_end, 0:width]
                # cv2.imshow('image_split_row', image_split_row)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(row_mark-1),
                # image_split_row, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                raw_image_split_row = raw_img_input[row_split_start:row_split_end, 0:width]
                # cv2.imshow('raw_image_split_row', raw_image_split_row)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(row_mark - 1),
                            raw_image_split_row, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                '''
                得到行切割图像，直接对原图像进行颜色判定，进而完成选定亮度阈值，最后得到二值图，显示并保存
                '''
                binary_img = detect_row_img_color(raw_image_split_row)

                # final_binary_img_save_name = 'final_row_binary'
                binary_img.save('../Img_processed/binary_row_{}.jpg'.format(row_mark - 1))

#
# def char_split_submodule_find_end(start_row, height_or_width_of_image, black_num, white_num,
#                                   black_num_max, white_num_max, arg):
#     """
#     该函数旨在找到对应切割图片行或列结束的位置
#     :param start_row:
#     :param height_or_width_of_image:
#     :param black_num:
#     :param white_num:
#     :param black_num_max:
#     :param white_num_max:
#     :param arg:
#     :return:
#     """
#     end_ = start_row + 1
#     '''arg = True，黑底白字情况下：对于第m行，如果该行黑色像素大于
#     0.95*（所有行中黑色像素数目和的最大值），则证明该列包含的白色字符太少，判定次列即为字符切割结束列；
#     对于白底黑字情况，则反之'''
#     for m in range(start_row + 1, height_or_width_of_image + 1):
#         if (black_num[m] if arg else white_num[m]) > \
#                 (0.95 * black_num_max if arg else 0.95 * white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
#             end_ = m
#             break
#     return end_


def char_split(raw_img_input):
    """
    分割图像，完成字符级切割
    :param raw_img_input:
    :return:
    """
    # 分割图像，提取字符行
    def find_end_row(start_row):
        end_row = start_row + 1
        '''arg = True，黑底白字情况下：对于第m行，如果该行黑色像素大于
        0.95*（所有行中黑色像素数目和的最大值），则证明该列包含的白色字符太少，判定次列即为字符切割结束列；
        对于白底黑字情况，则反之'''
        for m in range(start_row + 1, image_height + 1):
            if (row_black_num[m] if arg else row_white_num[m]) > \
                    (0.9 * row_black_num_max if arg else 0.9 * row_white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_row = m
                break
        return end_row

    # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
    gen_raw_img = gen_thresh_img(raw_img_input, threshold=65, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
    gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
    img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图
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

    row_mark = 1

    while row_mark < image_height - 2:
        row_mark += 1
        '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
        即将该行看做出现字符的起始行。
        对于白底黑字情况，则反之'''
        if (row_white_num[row_mark] if arg else row_black_num[row_mark]) > \
                (0.1 * row_white_num_max if arg else 0.1 * row_black_num_max):

            row_find_end = find_end_row(row_mark)

            if row_find_end - row_mark > 5:  # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰

                # 因为行间隔比较大，故而在此取检测出现字符行的前面的第3行作为行切割开始行（行长为12）
                row_split_start = row_mark - 3 if row_mark > 3 else row_mark
                row_split_end = row_find_end + 3 if row_find_end + 3 < image_height else row_find_end

                # 对上述二值图完成行切割、显示、保存
                # image_split_row = thresh_img[row_split_start:row_split_end, 0:width]
                # cv2.imshow('image_split_row', image_split_row)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(row_mark-1), image_split_row)

                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                raw_image_split_row = raw_img_input[row_split_start:row_split_end, 0:width]
                # cv2.imshow('raw_image_split_row', raw_image_split_row)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(row_mark - 1),
                            raw_image_split_row, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                '''
                得到行切割图像，直接对原图像进行颜色判定，进而完成选定亮度阈值，最后得到二值图，显示并保存
                '''
                binary_img = detect_row_img_color(raw_image_split_row)

                # final_binary_img_save_name = 'final_row_binary'
                binary_img.save('../Img_processed/binary_row_{}.jpg'.format(row_mark - 1))
                print('Completing the row image split')
                '''
                此处，不在将分行处理结果转变为二值图，而是转变为只有0与255的RGB图，以便于进行字符切割中的Image到OpenCV转换
                下面正式开始字符切割：
                '''

                def find_end_column(start_column):
                    # 这里要尤其注意，因为start_column 输入时候减去了2，如果这里直接+1，
                    # 会造成下一列立马结束（因为出现背景列），导致end_column < start_column
                    # 现在直接输入原始开始列，不在纠结这个问题
                    start_column = start_column + 1
                    '''arg = True，黑底白字情况下：对于第m行，如果该行黑色像素大于
                    0.95*（所有行中黑色像素数目和的最大值），则证明该列包含的白色字符太少，判定次列即为字符切割结束列；
                    对于白底黑字情况，则反之'''
                    for m in range(start_column, char_width + 1):
                        # 0.95这个参数请多调整，对应下面的0.05
                        if (column_black_num[m] if char_arg else column_white_num[m]) > \
                                (0.96 * column_black_num_max if arg else 0.96 * column_white_num_max):

                            end_column = m

                            break

                    return end_column

                # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
                # char_gen_raw_img = gen_thresh_img(binary_img, threshold=65, mode='fixed')  # 生成gen_raw_img图片格式为PIL格式
                char_gen_img = cv2.cvtColor(np.asarray(binary_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
                cv2.imshow('char_gen_img', char_gen_img)
                cv2.waitKey(0)
                print('Complete the format of char_split method from Image to OpenCV')
                char_img_gray = cv2.cvtColor(char_gen_img, cv2.COLOR_BGR2GRAY)
                _, char_thresh_img = cv2.threshold(char_img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

                cv2.imshow('char_thresh_img', char_thresh_img)
                cv2.waitKey(0)

                # thresh_img = cv2.cvtColor(np.asarray(thresh_img), cv2.COLOR_RGB2BGR)
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

                    temp_column_white_num_max = max(column_white_num_max, white_num)
                    temp_column_black_num_max = max(column_black_num_max, black_num)

                    # 排除列中出现的框线引起column_white_num_max=char_image_height情况
                    if char_image_height - temp_column_white_num_max > 1:
                        column_white_num_max = temp_column_white_num_max
                        column_black_num_max = temp_column_black_num_max

                    column_white_num.append(white_num)  # 记录该列的白色像素总数
                    column_black_num.append(black_num)  # 记录该列的黑色像素总数

                char_arg = True
                '''char_arg = True表示黑底白字, char_arg = False表示白底黑字'''
                if column_black_num_max < column_white_num_max:
                    char_arg = False

                column_mark = 1

                while column_mark < char_width - 2:
                    column_mark += 1
                    '''char_arg = True，即黑底白字情况下，第n列的白色像素数目和 > 0.05 * （所有列中白色像素数目和的最大值），
                    即将该列看做出现字符的起始列。
                    对于白底黑字情况，则反之；
                    注：该种情况会将小数点过滤掉，因此此处需添加对小数点的支持，
                    设定规则为当检测到该列字符像素点小于0.05*最大白色像素列，则若其列宽也<=2，则将其判定为小数点，
                    若不是这种情况，则直接将一列长结束位置重新复制为检测列起始位置，开始下一轮字符检测；
                    '''
                    # 首先检测列方框线，判定规则为列框线长度与行高之差在1之内，若检测到，则直接过滤进入下轮检测。
                    if (column_white_num_max if char_arg else column_black_num_max) - \
                            (column_white_num[column_mark] if char_arg else column_black_num[column_mark]) <= 1:

                        continue

                    elif (column_white_num[column_mark] if char_arg else column_black_num[column_mark]) > \
                            (0.05 * column_white_num_max if char_arg else 0.05 * column_black_num_max):

                        column_find_end = find_end_column(column_mark)

                        # 对于一般字符判定：最短字符长度为3，因此要求字符列长必须在3以上
                        # 对于数字1判定：当列包含字符长度高于一定阈值，则即便列宽为1依旧视为字符处理
                        if column_find_end - column_mark <= 2:
                            # 首先进行数字1判定，拿开始列字符所含字符像素数目大于所有列中最大字符像素数的0.8作为判定条件
                            if column_find_end - column_mark == 1:  # 列宽为1
                                if column_white_num[column_mark] / column_white_num_max >= 0.5:

                                    column_split_start = column_mark - 1 \
                                        if column_mark > 1 else column_mark

                                    column_split_end = column_find_end + 1 \
                                        if column_find_end + 1 < char_width else column_find_end

                                    # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                    raw_image_split_char = raw_image_split_row[
                                                           0:char_image_height,
                                                           column_split_start:column_split_end
                                                           ]

                                    cv2.imshow('raw_image_split_char', raw_image_split_char)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                                raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                # 进行小数点判定，阈值设到0.34保证能承受大约3个像素点
                                elif column_white_num[column_mark] / column_white_num_max < 0.34:

                                    column_split_start = column_mark - 1 \
                                        if column_mark > 1 else column_mark

                                    column_split_end = column_find_end + 1 \
                                        if column_find_end + 1 < char_width else column_find_end

                                    # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                    raw_image_split_char = raw_image_split_row[
                                                           0:char_image_height,
                                                           column_split_start:column_split_end
                                                           ]

                                    cv2.imshow('raw_image_split_char', raw_image_split_char)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                                raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                            # 首先进行数字1判定，拿开始列字符所含字符像素数目大于所有列中最大字符像素数的0.8作为判定条件
                            elif column_find_end - column_mark == 2:  # 列宽为1
                                if column_white_num[column_mark] / column_white_num_max >= 0.5 or \
                                        column_white_num[column_mark+1] / column_white_num_max >= 0.5:

                                    column_split_start = column_mark - 1 \
                                        if column_mark > 1 else column_mark

                                    column_split_end = column_find_end + 1 \
                                        if column_find_end + 1 < char_width else column_find_end

                                    # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                    raw_image_split_char = raw_image_split_row[
                                                           0:char_image_height,
                                                           column_split_start:column_split_end
                                                           ]

                                    cv2.imshow('raw_image_split_char', raw_image_split_char)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                                raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                # 进行小数点判定，同理，阈值设到0.34，保证能承受大约3个像素值
                                elif column_white_num[column_mark] / column_white_num_max and \
                                        column_white_num[column_mark+1] / column_white_num_max < 0.34:

                                    column_split_start = column_mark - 1 \
                                        if column_mark > 1 else column_mark

                                    column_split_end = column_find_end + 1 \
                                        if column_find_end + 1 < char_width else column_find_end

                                    # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                    raw_image_split_char = raw_image_split_row[
                                                           0:char_image_height,
                                                           column_split_start:column_split_end
                                                           ]

                                    cv2.imshow('raw_image_split_char', raw_image_split_char)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                                    cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                                raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        elif column_find_end - column_mark > 2:  # 要求切割字符行需要有2列以上的距离也是为了保证排除直线的干扰

                            '''此处开始判定是否出现黏连字符情况，此处只考虑双字符黏连情况'''
                            char_split_width = column_find_end - column_mark
                            char_split_height = char_image_height

                            # 通过高宽比进行判定字符是否出现黏连，如果出现黏连则取中间线进行切分
                            # (两字符黏连阈值设为1.8，三字符黏连阈值设到0.9
                            if 0.85 < char_split_height / char_split_width < 1.8:
                                first_char_split_start = column_mark
                                first_char_split_end = second_char_split_start = int((column_find_end + column_mark)/2)
                                second_char_split_end = column_find_end

                                first_char_split_start = first_char_split_start - 1 \
                                    if first_char_split_start > 1 else first_char_split_start
                                second_char_split_end = second_char_split_end + 1 \
                                    if second_char_split_end + 1 < char_width else second_char_split_end

                                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                first_raw_image_split_char = raw_image_split_row[
                                                             0:char_image_height,
                                                             first_char_split_start:first_char_split_end
                                                             ]

                                second_raw_image_split_char = raw_image_split_row[
                                                              0:char_image_height,
                                                              second_char_split_start:second_char_split_end
                                                              ]

                                cv2.imshow('first_raw_image_split_char', first_raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(first_char_split_start - 1),
                                            first_raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                cv2.imshow('second_raw_image_split_char', second_raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(second_char_split_start - 1),
                                            second_raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                            elif char_split_height / char_split_width <= 0.85:  # 出现三字符情况

                                first_char_split_start = column_mark
                                first_char_split_end = second_char_split_start = \
                                    int((column_find_end - column_mark) / 3 + column_mark)
                                second_char_split_end = third_char_split_start = \
                                    int((column_find_end - column_mark) * 2 / 3 + column_mark)
                                third_char_split_end = column_find_end

                                first_char_split_start = first_char_split_start - 1 \
                                    if first_char_split_start > 1 else first_char_split_start
                                third_char_split_end = third_char_split_end + 1 \
                                    if third_char_split_end + 1 < char_width else third_char_split_end

                                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                first_raw_image_split_char = raw_image_split_row[
                                                             0:char_image_height,
                                                             first_char_split_start:first_char_split_end
                                                             ]

                                second_raw_image_split_char = raw_image_split_row[
                                                              0:char_image_height,
                                                              second_char_split_start:second_char_split_end
                                                              ]

                                third_raw_image_split_char = raw_image_split_row[
                                                              0:char_image_height,
                                                              third_char_split_start:third_char_split_end
                                                              ]

                                cv2.imshow('first_raw_image_split_char', first_raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(first_char_split_start - 1),
                                            first_raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                cv2.imshow('second_raw_image_split_char', second_raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite(
                                    '../Img_processed/char_split_{}.jpg'.format(second_char_split_start - 1),
                                    second_raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                                cv2.imshow('third_raw_image_split_char', third_raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite(
                                    '../Img_processed/char_split_{}.jpg'.format(third_char_split_start - 1),
                                    second_raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            else:

                                # 由于字符之间间隔只有两个像素距离，因此取1个空白字符
                                column_split_start = column_mark - 1 \
                                    if column_mark > 1 else column_mark

                                column_split_end = column_find_end + 1 \
                                    if column_find_end + 1 < char_width else column_find_end

                                # 对上述二值图完成行切割、显示、保存
                                # image_split_row = thresh_img[column_split_start:column_split_end, 0:char_width]
                                # cv2.imshow('image_split_row', image_split_row)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()
                                # cv2.imwrite('../Img_processed/row_split_{}.jpg'.format(column_mark-1),
                                # image_split_row)

                                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                raw_image_split_char = raw_image_split_row[
                                                       0:char_image_height,
                                                       column_split_start:column_split_end
                                                       ]

                                cv2.imshow('raw_image_split_char', raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                            raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                        column_mark = column_find_end  # 不管检测到的列是不是字符，都将结束列重新赋值给开始列，进行下一轮循环

                    elif 0 < (column_white_num[column_mark] if char_arg else column_black_num[column_mark]) <= \
                            (0.05 * column_white_num_max if char_arg else 0.05 * column_black_num_max):

                        column_find_end = find_end_column(column_mark)
                        # 添加对小数点起始检测列检测
                        # 对于一般字符判定：最短字符长度为3，因此要求字符列长必须在3以上
                        # 对于数字1判定：当列包含字符长度高于一定阈值，则即便列宽为1依旧视为字符处理
                        # 进行小数点判定，拿开始列字符所含字符像素数目小于所有列中最大字符像素数的0.2作为判定条件
                        if column_find_end - column_mark <= 2:

                            if column_white_num[column_mark] / column_white_num_max <= 0.2:

                                column_split_start = column_mark - 1 \
                                    if column_mark > 1 else column_mark

                                column_split_end = column_find_end + 1 \
                                    if column_find_end + 1 < char_width else column_find_end

                                # 对应上述标准完成原图切割、显示、保存（注意是对原图的切割，而非二值图）
                                raw_image_split_char = raw_image_split_row[
                                                       0:char_image_height,
                                                       column_split_start:column_split_end
                                                       ]

                                cv2.imshow('raw_image_split_char', raw_image_split_char)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
                                cv2.imwrite('../Img_processed/char_split_{}.jpg'.format(column_mark - 1),
                                            raw_image_split_char, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # row_find_end
            row_mark = row_find_end  # 无论是否进行行切割，将本次切割结束位置复制给下一次行切割初始点


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
        bgr = color_to_check
        thresh = 40  # 上下范围扩展40
        min_bgr = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        max_bgr = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

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
        # 因为前面使用的是二值化方法是cv2.THRESH_BINARY，二值图表现为白底黑字
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


if __name__ == '__main__':
    # input_image = "../Img_processed/row_split_18.jpg"
    input_image = "../Image/3.jpg"
    # print(get_dominant_color(Image.open("../Image/2.JPG")))
    # get_dominant_color(input_image)

    # gen_img_address = get_gery(input_image)
    # img = cv2.imread('../Img_processed/%s.png' % gen_img_address)

    # cv2.imshow("binary_img.png", binary_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    raw_img = cv2.imread(input_image)
    # row_split(raw_img)
    char_split(raw_img)
    # get_gery(input_image)

    # detect_row_img_color(input_image)
