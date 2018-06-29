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
    # cv2.imwrite(target_folder + '/img_compressed.jpg', compressed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return img_to_change, img_compressed


def img_segmentation(initial_raw_img):
    """
    依据字符列平均字符数量，完成panel字符行鉴定
    :param initial_raw_img:
    :return:
    """

    # 设置压缩尺寸和选项，注意尺寸要用括号
    def img_row_or_col_compress(initial_img_input, width_compression_ratio=1, height_compression_ratio=1):
        initial_width, initial_height = initial_img_input.size
        used_compressed_img = initial_img_input.resize((int(initial_width / width_compression_ratio),
                                                        int(initial_height / height_compression_ratio)),
                                                       Image.ANTIALIAS)

        compressed_width, compressed_height = used_compressed_img.size
        print("compressed_width:{}, compressed_height:{}".format(compressed_width, compressed_height))
        return used_compressed_img

    def img_column_split(initial_img):
        # initial_img = Image.open(initial_img)
        initial_opencv_img = cv2.cvtColor(np.asarray(initial_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
        initial_width, initial_height = initial_img.size
        # print("initial_width:{}, initial_height:{}".format(initial_width, initial_height))

        final_compressed_img = img_row_or_col_compress(initial_img, height_compression_ratio=20)

        gen_img = cv2.cvtColor(np.asarray(final_compressed_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
        cv2.namedWindow("gen_img", cv2.WINDOW_NORMAL)
        cv2.imshow('gen_img', gen_img)
        cv2.waitKey(0)

        gray_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
        sobel = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=1)  # Sobel算子，x方向求梯度
        ret, panel_text_binary_img = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)  # 二值化
        cv2.namedWindow("panel_text_binary_img", cv2.WINDOW_NORMAL)
        cv2.imshow('panel_text_binary_img', panel_text_binary_img)
        cv2.waitKey(0)

        column_white_num = []  # 记录每一列的白色像素总和

        char_image_height = panel_text_binary_img.shape[0]
        char_width = panel_text_binary_img.shape[1]
        column_white_num_max = 0

        # 计算每一列的黑白色像素总和
        for i in range(char_width):
            white_num = 0  # 这一列白色总数
            # black_num = 0  # 这一列黑色总数
            for j in range(char_image_height):
                if panel_text_binary_img[j][i] == 255:
                    white_num += 1

            column_white_num_max = max(column_white_num_max, white_num)
            column_white_num.append(white_num)  # 记录该列的白色像素总数

        column_index_list = []

        for column_index in range(len(column_white_num)):
            if column_white_num[column_index] >= column_white_num_max * 0.7:
                column_index_list.append(column_index)

        print('column_index:{}'.format(column_index_list))

        # 完成图像块第一步列切割
        _initial_img_block_one = initial_opencv_img[0:initial_height, 0:min(column_index_list) - 10]
        _initial_img_block_two = initial_opencv_img[
                                 0:initial_height, min(column_index_list) + 20:max(column_index_list) - 5]
        _initial_img_block_three = initial_opencv_img[0:initial_height, max(column_index_list) + 20:initial_width]

        # cv2.imshow('img_block_one', initial_img_block_one)
        # cv2.waitKey(0)
        # cv2.imshow('img_block_two', initial_img_block_two)
        # cv2.waitKey(0)
        # cv2.imshow('img_block_three', initial_img_block_three)
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()

        # # 对第一列图像进行存档
        # cv2.imwrite('../Img_processed/img_block_one_{}.jpg'.format(min(column_index_list)),
        #             _initial_img_block_one, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        return _initial_img_block_one, _initial_img_block_two, _initial_img_block_three

    def img_block_two_split(img_block_two_input):
        """
        完成中间块分割，按照行块分成上下两块
        :param img_block_two_input:
        :return:
        """
        # 对两二列图形进一步切割，采用上述类似方式
        img_block_two = Image.fromarray(cv2.cvtColor(img_block_two_input, cv2.COLOR_BGR2RGB))  # 转换为PIL.Image图片格式
        img_block_two_width, img_block_two_height = img_block_two.size
        img_block_two_compressed_img = img_row_or_col_compress(img_block_two, width_compression_ratio=10)

        # 再次转换为OpenCV图片格式
        img_block_two_gen_img = cv2.cvtColor(np.asarray(img_block_two_compressed_img), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("img_block_two_gen_img", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_two_gen_img', img_block_two_gen_img)
        cv2.waitKey(0)

        img_block_two_gray_img = cv2.cvtColor(img_block_two_gen_img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
        img_block_two_sobel = cv2.Sobel(img_block_two_gray_img, cv2.CV_8U, 0, 1, ksize=5)  # Sobel算子，x方向求梯度

        # 二值化
        _, img_block_two_binary_img = cv2.threshold(img_block_two_sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        cv2.namedWindow("img_block_two_binary_img", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_two_binary_img', img_block_two_binary_img)
        cv2.waitKey(0)

        img_block_two_row_white_num = []  # 记录每一行的白色像素总和
        img_block_two_image_height = img_block_two_gen_img.shape[0]
        img_block_two_image_width = img_block_two_gen_img.shape[1]
        img_block_two_row_white_num_max = 0

        # 计算每一行的白色像素总和
        for img_block_two_i in range(img_block_two_image_height):
            img_block_two_white_num = 0  # 这一行白色总数
            for img_block_two_j in range(img_block_two_image_width):
                if img_block_two_binary_img[img_block_two_i][img_block_two_j] == 255:
                    img_block_two_white_num += 1
            img_block_two_row_white_num_max = max(img_block_two_row_white_num_max, img_block_two_white_num)
            img_block_two_row_white_num.append(img_block_two_white_num)  # 记录该行的白色像素总数

        img_block_two_row_index_list = []

        for img_block_two_row_index in range(len(img_block_two_row_white_num)):
            if img_block_two_row_white_num[img_block_two_row_index] >= img_block_two_row_white_num_max * 0.8:
                img_block_two_row_index_list.append(img_block_two_row_index)

        print('img_block_two_column_index_list:{}'.format(img_block_two_row_index_list))

        # 完成第二块图像块行切割
        img_block_two_block_one = img_block_two_input[0:min(img_block_two_row_index_list) - 5, 0:img_block_two_width]
        img_block_two_block_two = img_block_two_input[
                                  min(img_block_two_row_index_list) + 5:max(img_block_two_row_index_list) - 5,
                                  0:img_block_two_width]

        cv2.namedWindow("img_block_two_block_one", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_two_block_one', img_block_two_block_one)
        cv2.waitKey(0)
        cv2.imshow('img_block_two_block_one', img_block_two_block_two)
        cv2.waitKey(0)
        return img_block_two_block_one, img_block_two_block_two

    def img_block_three_split(img_block_three_input):
        """
        完成最右列图像块分割，按照行框，分成4份
        :param img_block_three_input:
        :return:
        """

        # 对两二列图形进一步切割，采用上述类似方式
        img_block_three = Image.fromarray(cv2.cvtColor(img_block_three_input, cv2.COLOR_BGR2RGB))  # 转换为PIL.Image图片格式
        img_block_three_width, img_block_three_height = img_block_three.size
        img_block_three_compressed_img = img_row_or_col_compress(img_block_three, width_compression_ratio=10)

        # 再次转换为OpenCV图片格式
        img_block_three_gen_img = cv2.cvtColor(np.asarray(img_block_three_compressed_img), cv2.COLOR_RGB2BGR)
        cv2.namedWindow("img_block_three_gen_img", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_three_gen_img', img_block_three_gen_img)
        cv2.waitKey(0)

        img_block_three_gray_img = cv2.cvtColor(img_block_three_gen_img, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
        img_block_three_sobel = cv2.Sobel(img_block_three_gray_img, cv2.CV_8U, 1, 0, ksize=7)  # Sobel算子，x方向求梯度

        # 二值化
        _, img_block_three_binary_img = cv2.threshold(img_block_three_sobel, 0, 255,
                                                      cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        cv2.namedWindow("img_block_three_binary_img", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_three_binary_img', img_block_three_binary_img)
        cv2.waitKey(0)

        img_block_three_row_white_num = []  # 记录每一行的白色像素总和
        img_block_three_image_height = img_block_three_gen_img.shape[0]
        img_block_three_image_width = img_block_three_gen_img.shape[1]
        img_block_three_row_white_num_max = 0

        print('img_block_three_input_height:{}, img_block_three_input_width:{}'.format(
            img_block_three_input.shape[0], img_block_three_input.shape[1]))
        print('img_block_three_image_height:{}, img_block_three_image_width:{}'.format(
            img_block_three_gen_img.shape[0], img_block_three_gen_img.shape[1]))

        # 计算每一行的白色像素总和
        for img_block_three_i in range(img_block_three_image_height):
            img_block_three_white_num = 0  # 这一行白色总数
            for img_block_three_j in range(img_block_three_image_width):
                if img_block_three_binary_img[img_block_three_i][img_block_three_j] == 255:
                    img_block_three_white_num += 1
            img_block_three_row_white_num_max = max(img_block_three_row_white_num_max, img_block_three_white_num)
            img_block_three_row_white_num.append(img_block_three_white_num)  # 记录该行的白色像素总数

        img_block_three_row_index_list = []

        for img_block_three_row_index in range(len(img_block_three_row_white_num)):
            if img_block_three_row_white_num[img_block_three_row_index] >= img_block_three_row_white_num_max * 0.7:
                img_block_three_row_index_list.append(img_block_three_row_index)

        print('img_block_three_row_index_list:{}'.format(img_block_three_row_index_list))

        final_img_block_three_row_index_list = []

        # print('img_block_three_row_index_list:{}'.format(img_block_three_row_index_list))
        # remove_list = []
        for img_block_three_row_split_count in range(2):
            remove_list = []
            for img_block_three_row_index in range(len(img_block_three_row_index_list)):
                img_block_three_row_split_start = np.abs(img_block_three_row_index_list[img_block_three_row_index] -
                                                         min(img_block_three_row_index_list))
                img_block_three_row_split_end = np.abs(img_block_three_row_index_list[img_block_three_row_index] -
                                                       max(img_block_three_row_index_list))

                if (0 < img_block_three_row_split_start <= 8) or (0 < img_block_three_row_split_end <= 8):
                    remove_list.append(img_block_three_row_index_list[img_block_three_row_index])
                    # img_block_three_row_index_list.remove(img_block_three_row_index_list[img_block_three_row_index])
            for remove_index in remove_list:
                img_block_three_row_index_list.remove(remove_index)
                # remove_list.remove(remove_index)
                # img_block_three_row_index_list.remove(img_block_three_row_index_list[remove_index])

            final_img_block_three_row_index_list.append(min(img_block_three_row_index_list))
            final_img_block_three_row_index_list.append(max(img_block_three_row_index_list))

            img_block_three_row_index_list.remove(min(img_block_three_row_index_list))
            img_block_three_row_index_list.remove(max(img_block_three_row_index_list))

        final_img_block_three_row_index_list.sort()  # 对于列表进行重新排序
        print('final_img_block_three_row_index_list:{}'.format(final_img_block_three_row_index_list))

        # 完成第二块图像块行切割
        _img_block_three_block_one = img_block_three_input[
                                     0:final_img_block_three_row_index_list[0] - 10,
                                     0:img_block_three_width]

        _img_block_three_block_two = img_block_three_input[
                                     final_img_block_three_row_index_list[0] + 10:
                                     final_img_block_three_row_index_list[1] - 10,
                                     0:img_block_three_width]

        _img_block_three_block_three = img_block_three_input[
                                       final_img_block_three_row_index_list[1] + 10:
                                       final_img_block_three_row_index_list[2] - 10,
                                       0:img_block_three_width]

        _img_block_three_block_four = img_block_three_input[
                                      final_img_block_three_row_index_list[2] + 10:
                                      final_img_block_three_row_index_list[3] - 10,
                                      0:img_block_three_width]

        # cv2.namedWindow("img_block_three_block_one", cv2.WINDOW_NORMAL)
        cv2.imshow('img_block_three_block_one', _img_block_three_block_one)
        cv2.waitKey(0)
        cv2.imshow('img_block_three_block_two', _img_block_three_block_two)
        cv2.waitKey(0)
        cv2.imshow('img_block_three_block_three', _img_block_three_block_three)
        cv2.waitKey(0)
        cv2.imshow('img_block_three_block_four', _img_block_three_block_four)
        cv2.waitKey(0)

        return _img_block_three_block_one, _img_block_three_block_two, _img_block_three_block_three, \
            _img_block_three_block_four

    def img_block_three_block_four_split(img_block_three_block_four_input):

        # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）
        # 生成gen_raw_img图片格式为PIL格式
        gen_raw_img = gen_thresh_img(img_block_three_block_four_input, threshold=50, mode='fixed')
        gen_img = cv2.cvtColor(np.asarray(gen_raw_img), cv2.COLOR_RGB2BGR)  # 转换为OpenCV图片格式
        img_gray = cv2.cvtColor(gen_img, cv2.COLOR_BGR2GRAY)
        _, img_block_three_block_four_binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

        cv2.imshow('img_block_three_block_four_binary_img', img_block_three_block_four_binary_img)
        cv2.waitKey(0)
        # img_block_three_block_four_binary_img = thresh_img

        img_three_col_four_row_height = img_block_three_block_four_binary_img.shape[0]
        img_three_col_four_row_width = img_block_three_block_four_binary_img.shape[1]
        img_three_col_four_row_column_white_num_max = 0
        img_three_col_four_row = []

        # 计算每一列的黑白色像素总和
        for img_three_col_four_row_i in range(img_three_col_four_row_width):
            img_three_col_four_row_white_num = 0  # 这一列白色总数
            for img_three_col_four_row_j in range(img_three_col_four_row_height):
                if img_block_three_block_four_binary_img[img_three_col_four_row_j][img_three_col_four_row_i] == 255:
                    img_three_col_four_row_white_num += 1

            img_three_col_four_row_column_white_num_max = \
                max(img_three_col_four_row_column_white_num_max, img_three_col_four_row_white_num)

            img_three_col_four_row.append(img_three_col_four_row_white_num)  # 记录该列的白色像素总数

        img_three_col_four_row_split_index = []

        print('img_three_col_four_row:{}'.format(img_three_col_four_row))

        # 寻找分列起始点
        for img_three_col_four_row_column_index in range(len(img_three_col_four_row)):
            start_point_count = 0

            for mid_point_threshold in range(10):
                if img_three_col_four_row[img_three_col_four_row_column_index + mid_point_threshold] >= \
                        img_three_col_four_row_column_white_num_max * 0.2:
                    start_point_count += 1
            if start_point_count == 10:
                # 在切割开始点减10保证切割留有一定冗余
                img_three_col_four_row_split_index.append(img_three_col_four_row_column_index)
                break

        # 寻找分列结束点
        for img_three_col_four_row_column_index in range(len(img_three_col_four_row), 0, -1):
            end_point_count = 0

            for mid_point_threshold in range(10):
                if img_three_col_four_row[img_three_col_four_row_column_index - mid_point_threshold - 1] >= \
                        img_three_col_four_row_column_white_num_max * 0.2:
                    end_point_count += 1
            if end_point_count == 10:
                # 在切割结束点加10保证切割有一定冗余
                img_three_col_four_row_split_index.append(img_three_col_four_row_column_index)
                break

        # 寻找分列中点
        for img_three_col_four_row_column_index in range(len(img_three_col_four_row)):
            min_point_count = 0
            if img_three_col_four_row_split_index[0] < img_three_col_four_row_column_index < \
                    img_three_col_four_row_split_index[1]:
                for mid_point_threshold in range(25):
                    if img_three_col_four_row[img_three_col_four_row_column_index] > 0 and \
                            img_three_col_four_row[img_three_col_four_row_column_index + mid_point_threshold + 1] == 0:
                        min_point_count += 1
                if min_point_count == 25:
                    img_three_col_four_row_split_index.append(img_three_col_four_row_column_index)
                    break

        img_three_col_four_row_split_index.sort()
        # 完成右下角图像块行切割
        _img_three_col_four_row_one = img_block_three_block_four_input[
                                     0:img_three_col_four_row_height,
                                     img_three_col_four_row_split_index[0] - 5:
                                     img_three_col_four_row_split_index[1] + 15]

        _img_three_col_four_row_two = img_block_three_block_four_input[
                                     0:img_three_col_four_row_height,
                                     img_three_col_four_row_split_index[1] + 15:
                                     img_three_col_four_row_split_index[2] + 5]

        # # cv2.namedWindow("img_block_three_block_one", cv2.WINDOW_NORMAL)
        cv2.imshow('img_three_col_four_row_one', _img_three_col_four_row_one)
        cv2.waitKey(0)
        cv2.imshow('img_three_col_four_row_two', _img_three_col_four_row_two)
        cv2.waitKey(0)

        return _img_three_col_four_row_one, _img_three_col_four_row_two

    def img_block_two_block_one_split(img_block_two_block_one_input):

        # 首先，依据亮度值得到黑白图像（虽然像素值只有0与255，但有BGR三个维度）,生成gen_raw_img图片格式为PIL格式
        img_two_one_binary_img_temp = gen_thresh_img(img_block_two_block_one_input, threshold=50, mode='fixed')
        # 转换为OpenCV图片格式
        initial_img_two_one_binary_img = cv2.cvtColor(np.asarray(img_two_one_binary_img_temp), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(initial_img_two_one_binary_img, cv2.COLOR_BGR2GRAY)
        _, img_two_one_binary_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 得到二值图

        cv2.namedWindow("img_two_one_binary_img", cv2.WINDOW_NORMAL)
        cv2.imshow('img_two_one_binary_img', img_two_one_binary_img)
        cv2.waitKey(0)

        img_two_one_row_white_num = []  # 记录每一行的白色像素总和
        img_two_block_one_image_height = img_two_one_binary_img.shape[0]
        img_two_block_one_image_width = img_two_one_binary_img.shape[1]
        img_two_block_one_row_white_num_max = 0

        # print('img_two_block_one_image_height:{},img_two_block_one_image_width:{}'.format(
        #     img_two_block_one_image_height, img_two_block_one_image_width))

        # 计算每一行的白色像素总和
        for img_two_one_i in range(img_two_block_one_image_height):
            img_two_one_white_num = 0  # 这一行白色总数
            for img_two_one_j in range(img_two_block_one_image_width):
                if img_two_one_binary_img[img_two_one_i][img_two_one_j] == 255:
                    img_two_one_white_num += 1
            img_two_block_one_row_white_num_max = max(img_two_block_one_row_white_num_max, img_two_one_white_num)
            img_two_one_row_white_num.append(img_two_one_white_num)  # 记录该行的白色像素总数

        img_two_one_row_index_list = []
        # 寻找行一起始点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num)):
            img_two_one_first_point_count = 0

            for mid_point_threshold in range(10):
                if img_two_one_row_white_num[img_three_col_four_row_column_index + mid_point_threshold] >= \
                        img_two_block_one_row_white_num_max * 0.07:
                    img_two_one_first_point_count += 1

            if img_two_one_first_point_count == 10:
                img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                break

        # 寻找行一结束点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num)):
            if img_three_col_four_row_column_index > img_two_one_row_index_list[0]:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(10):
                    if img_two_one_row_white_num[
                        img_three_col_four_row_column_index - mid_point_threshold] >= \
                            img_two_block_one_row_white_num_max * 0.03 and \
                            img_two_one_row_white_num[
                                img_three_col_four_row_column_index + mid_point_threshold + 1] < \
                            img_two_block_one_row_white_num_max * 0.030:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 10:
                    img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                    break

        # 寻找行二开始点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num)):
            if img_three_col_four_row_column_index > img_two_one_row_index_list[1]:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(10):
                    if img_two_one_row_white_num[
                        img_three_col_four_row_column_index + mid_point_threshold] >= \
                            img_two_block_one_row_white_num_max * 0.07:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 10:
                    img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                    break

        # 寻找行二结束点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num)):
            if img_three_col_four_row_column_index > img_two_one_row_index_list[2]:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(10):
                    if img_two_one_row_white_num[
                        img_three_col_four_row_column_index - mid_point_threshold] >= \
                            img_two_block_one_row_white_num_max * 0.03 and \
                            img_two_one_row_white_num[
                                img_three_col_four_row_column_index + mid_point_threshold + 1] < \
                            img_two_block_one_row_white_num_max * 0.030:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 10:
                    img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                    break

        # 寻找行三结束点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num), 0, -1):
            img_two_one_second_point_count = 0
            for mid_point_threshold in range(10):
                if img_two_one_row_white_num[
                    img_three_col_four_row_column_index - mid_point_threshold - 1] >= \
                        img_two_block_one_row_white_num_max * 0.03:
                    img_two_one_second_point_count += 1
            if img_two_one_second_point_count == 10:
                img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                break

        # 寻找行三开始点
        for img_three_col_four_row_column_index in range(len(img_two_one_row_white_num), 0, -1):
            if img_three_col_four_row_column_index < img_two_one_row_index_list[4] - 5:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(10):
                    if img_two_one_row_white_num[
                        img_three_col_four_row_column_index - mid_point_threshold] < \
                            img_two_block_one_row_white_num_max * 0.01 and \
                            img_two_one_row_white_num[
                                img_three_col_four_row_column_index + mid_point_threshold + 1] >= \
                            img_two_block_one_row_white_num_max * 0.010:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 10:
                    img_two_one_row_index_list.append(img_three_col_four_row_column_index)
                    break

        img_two_one_row_index_list.sort()
        # print('img_two_one_row_index_list:{}'.format(img_two_one_row_index_list))

        # 完成第二块图像块行切割
        _img_two_one_row_one = img_block_two_block_one_input[
                               img_two_one_row_index_list[0] - 10:
                               img_two_one_row_index_list[1] + 10,
                               0:img_two_block_one_image_width]

        # _img_two_one_row_two = img_block_two_block_one_input[
        #                        img_two_one_row_index_list[2] - 15:
        #                        img_two_one_row_index_list[3] + 5,
        #                        0:img_two_block_one_image_width]
        #
        # _img_two_one_row_three = img_block_two_block_one_input[
        #                          img_two_one_row_index_list[4] - 5:
        #                          img_two_one_row_index_list[5] + 15,
        #                          0:img_two_block_one_image_width]
        #
        # cv2.imshow('_img_two_one_row_one', _img_two_one_row_one)
        # cv2.waitKey(0)
        # cv2.imshow('_img_two_one_row_two', _img_two_one_row_two)
        # cv2.waitKey(0)
        # cv2.imshow('_img_two_one_row_three', _img_two_one_row_three)
        # cv2.waitKey(0)

        '''
        至此，完成行分割，结合进行列分割，最终分割为一行文本，四个仪表盘;
        直接依据二值图img_two_one_binary_img找到列分割点，完成对两行仪表盘分割
        '''

        img_two_one_column_white_num = []  # 记录每一列的白色像素总和
        img_two_one_column_white_num_max = 0

        # 计算每一列的黑白色像素总和
        for img_two_one_i in range(img_two_block_one_image_width):
            img_two_one_white_num = 0  # 这一列白色总数
            for img_two_one_j in range(img_two_block_one_image_height):
                if img_two_one_binary_img[img_two_one_j][img_two_one_i] == 255:
                    img_two_one_white_num += 1

            img_two_one_column_white_num_max = max(img_two_one_column_white_num_max, img_two_one_white_num)
            img_two_one_column_white_num.append(img_two_one_white_num)  # 记录该列的白色像素总数

        print('img_two_one_column_white_num:{}'.format(img_two_one_column_white_num))

        img_two_one_column_index_list = []
        # 寻找列一起始点
        for img_two_one_column_index in range(len(img_two_one_column_white_num)):
            img_two_one_first_point_count = 0

            for mid_point_threshold in range(10):
                if img_two_one_column_white_num[img_two_one_column_index + mid_point_threshold] >= \
                        img_two_one_column_white_num_max * 0.01:
                    img_two_one_first_point_count += 1

            if img_two_one_first_point_count == 10:
                img_two_one_column_index_list.append(img_two_one_column_index)
                break
        print('First: img_two_one_column_index_list:{}'.format(img_two_one_column_index_list))

        # 寻找列一结束点
        for img_two_one_column_index in range(len(img_two_one_column_white_num)):
            if img_two_one_column_index_list[0] + 50 < img_two_one_column_index < \
                    len(img_two_one_column_white_num) - 11:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(10):
                    if img_two_one_column_white_num[
                        img_two_one_column_index - mid_point_threshold] >= img_two_one_column_white_num_max * 0.01 and \
                            img_two_one_column_white_num[img_two_one_column_index + mid_point_threshold + 1] < \
                            img_two_one_column_white_num_max * 0.010:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 10:
                    img_two_one_column_index_list.append(img_two_one_column_index)
                    break
        print('Second: img_two_one_column_index_list:{}'.format(img_two_one_column_index_list))

        # 寻找列二结束点
        for img_two_one_column_index in range(len(img_two_one_column_white_num), 0, -1):
            img_two_one_second_point_count = 0
            for mid_point_threshold in range(10):
                if img_two_one_column_white_num[
                    img_two_one_column_index - mid_point_threshold - 1] >= \
                        img_two_one_column_white_num_max * 0.01:
                    img_two_one_second_point_count += 1
            if img_two_one_second_point_count == 10:
                img_two_one_column_index_list.append(img_two_one_column_index)
                break
        print('Third: img_two_one_column_index_list:{}'.format(img_two_one_column_index_list))

        # 寻找列二开始点
        for img_two_one_column_index in range(len(img_two_one_column_white_num), 0, -1):
            if img_two_one_column_index < img_two_one_column_index_list[2] - 5:
                img_two_one_second_point_count = 0
                for mid_point_threshold in range(5):
                    if img_two_one_column_white_num[
                        img_two_one_column_index - mid_point_threshold] < img_two_one_column_white_num_max * 0.01 and \
                            img_two_one_column_white_num[img_two_one_column_index + mid_point_threshold + 1] >= \
                            img_two_one_column_white_num_max * 0.010:
                        img_two_one_second_point_count += 1
                if img_two_one_second_point_count == 5:
                    img_two_one_column_index_list.append(img_two_one_column_index)
                    break
        print('fourth: img_two_one_column_index_list:{}'.format(img_two_one_column_index_list))

        img_two_one_column_index_list.sort()
        print('img_two_one_column_index_list:{}'.format(img_two_one_column_index_list))
        print('img_block_two_block_one_input:{}, {}'.format(img_block_two_block_one_input.shape[0],
                                                            img_block_two_block_one_input.shape[1]))

        # 完成第二行仪表盘图像列切割
        _img_top_left_panel = img_block_two_block_one_input[
                              img_two_one_row_index_list[2] - 15:
                              img_two_one_row_index_list[3] + 5,
                              img_two_one_column_index_list[0]:img_two_one_column_index_list[1] + 10]

        _img_top_right_panel = img_block_two_block_one_input[
                               img_two_one_row_index_list[2] - 15:
                               img_two_one_row_index_list[3] + 5,
                               img_two_one_column_index_list[2]:img_two_one_column_index_list[3]]

        # 完成第三行仪表盘图像列切割
        _img_lower_left_panel = img_block_two_block_one_input[
                                img_two_one_row_index_list[4] - 5:
                                img_two_one_row_index_list[5] + 15,
                                img_two_one_column_index_list[0]:img_two_one_column_index_list[1] + 10]

        _img_lower_right_panel = img_block_two_block_one_input[
                                 img_two_one_row_index_list[4]:
                                 img_two_one_row_index_list[5] + 15,
                                 img_two_one_column_index_list[2]:img_two_one_column_index_list[3]]

        cv2.imshow('_img_two_one_row_one', _img_two_one_row_one)
        cv2.waitKey(0)
        cv2.imshow('_img_top_left_panel', _img_top_left_panel)
        cv2.waitKey(0)
        cv2.imshow('_img_top_right_panel', _img_top_right_panel)
        cv2.waitKey(0)
        cv2.imshow('_img_lower_left_panel', _img_lower_left_panel)
        cv2.waitKey(0)
        cv2.imshow('_img_lower_right_panel', _img_lower_right_panel)
        cv2.waitKey(0)

        return _img_two_one_row_one, _img_top_left_panel, _img_top_right_panel, _img_lower_left_panel,\
            _img_lower_right_panel

    initial_img_block_one, initial_img_block_two, initial_img_block_three = img_column_split(initial_raw_img)

    initial_img_block_two_block_one, initial_img_block_two_block_two = img_block_two_split(initial_img_block_two)

    img_two_one_row_one, img_top_left_panel, img_top_right_panel, img_lower_left_panel, img_lower_right_panel = \
        img_block_two_block_one_split(initial_img_block_two_block_one)

    img_block_three_block_one, img_block_three_block_two, img_block_three_block_three, img_block_three_block_four = \
        img_block_three_split(initial_img_block_three)

    img_three_col_four_row_one, img_three_col_four_row_two = \
        img_block_three_block_four_split(img_block_three_block_four)

    return initial_img_block_one, img_two_one_row_one, img_top_left_panel, \
        img_top_right_panel, img_lower_left_panel, img_lower_right_panel, initial_img_block_two_block_two, \
        img_block_three_block_one, img_block_three_block_two, img_block_three_block_three, \
        img_three_col_four_row_one, img_three_col_four_row_two


def identify_panel_text(panel_text_input):
    """
    依据字符列平均字符数量，完成panel字符行鉴定
    :param panel_text_input:
    :return:
    """

    gray = cv2.cvtColor(panel_text_input, cv2.COLOR_BGR2GRAY)  # 转化成灰度图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=1)  # Sobel算子，x方向求梯度
    ret, panel_text_binary_img = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)  # 二值化
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
    # print("the format of image has changed from OpenCv to Image")
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
        # print('return: gen_binary_img')
        return gen_binary_img
    else:
        # print('return: gen_img')
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
    for name_index in range(97, 115, 1):
        input_image = "../Image/" + str(name_index) + ".png"
        # input_image = "../Image/panel_text_1.jpg"
        compressed_img, compressed_opencv_img = img_compression(input_image)
        # compressed_img = img_compression(initial_img_address=input_image)
        img_one, img_two_one_row_one, img_top_left_panel, img_top_right_panel, img_lower_left_panel, \
            img_lower_right_panel, img_two_two, img_three_one, img_three_two, img_three_three, img_three_four_one, \
            img_three_four_two = img_segmentation(compressed_img)

        target_folder_base = "C:/Users/dby_freedom/Desktop/Lab_OCR/Img_processed/"
        folder_name = (os.path.basename(input_image)).split(".")[0]
        target_folder = target_folder_base + str(folder_name)
        print('target_folder: {}'.format(target_folder))
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
            cv2.imwrite(target_folder + '/img_compressed.jpg', compressed_opencv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_two_one_row_one.jpg', img_two_one_row_one, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_top_left_panel.jpg', img_top_left_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_top_right_panel.jpg', img_top_right_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_lower_left_panel.jpg', img_lower_left_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_lower_right_panel.jpg', img_lower_right_panel, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_two_two.jpg', img_two_two, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_three_one.jpg', img_three_one, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_three_two.jpg', img_three_two, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_three_three.jpg', img_three_three, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_three_four_one.jpg', img_three_four_one, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(target_folder + '/img_three_four_two.jpg', img_three_four_two, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            print(target_folder + " already exists")
            pass

        cv2.destroyAllWindows()
