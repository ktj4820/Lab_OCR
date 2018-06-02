

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
