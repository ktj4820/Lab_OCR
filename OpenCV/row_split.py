###
# 需要在上面代码的基础上添加这两个方法
# 这些代码仅为演示效果，实际使用中并不需要绘制分割线
###

# 将图像中的文字行分割出来
def image_row_split(image):
    # 假设背景色是255（即白色）
    bg_color = 255
    row_length = len(image)
    column_length = len(image[0])

    is_last_blank_row = True    # 是最后的空白行（背景色）
    '''针对每行，检测该行中的各列元素，如果是白色（背景色），则通过，如果首次出现黑色，证明该行出现了字符，将
    其判定为字符行，在该字符行的上一行进行划线（通过将上一行像素转换为黑色0像素点），
    赋值is_last_blank_row = False，转到下一行is_row_blank 再次赋值为True，则当再次出现字符行的时候，
    is_row_blank = False， 满足 is_last_blank_row == is_row_blank，直接pass，直到最后
    is_row_blank = True，即该行为背景行的时候，才会出现is_last_blank_row ！= is_row_blank，
    然后再次对该行进行划线（转换该行颜色为背景色）'''
    for row in range(0, row_length):
        is_row_blank = True
        for column in range(0, column_length):
            if image[row][column] == bg_color:
                pass
            else:
                is_row_blank = False
                break

        if is_last_blank_row == is_row_blank:
            pass
        else:
            if is_row_blank == False:
                image[row - 1] = [0 for column in range(0, column_length)]    # 在首次出现字符行的前一行进行划线
            else:
                image[row] = [0 for column in range(0, column_length)]    # 在不再是字符行的时候进行划线

            is_last_blank_row = is_row_blank

    return image


# 对行分割的代码进行改造，即可得出分割字符的代码
def image_char_split(image, image_row_splited):
    image_save("output/image_test.png", image)

    bg_color = 255
    row_length = len(image)
    column_length = len(image[0])

    is_last_blank_column = True
    for column in range(0, column_length):
        is_column_blank = True
        for row in range(0, row_length):
            if image[row][column] == bg_color:
                pass
            else:
                is_column_blank = False
                break

        if is_last_blank_column == is_column_blank:
            pass
        else:
            if is_column_blank == False:
                for row in range(0, row_length):
                    image_row_splited[row][column - 1] = 0
            else:
                for row in range(0, row_length):
                    image_row_splited[row][column] = 0

            is_last_blank_column = is_column_blank

    return image_row_splited









    row_mark = 0
    while row_mark < image_height - 2:
        row_mark += 1
        '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
        即将该行看做出现字符的起始行。
        对于白底黑字情况，则反之'''
        if (row_white_num[row_mark] if arg else row_black_num[row_mark]) > (0.05 * row_white_num_max if arg else 0.05 * row_black_num_max):
            row_split_start = row_mark - 2 if row_mark > 2 else row_mark
            row_split_end = find_end(row_split_start)
            row_split_end += 2 if row_split_end + 2 < image_height else row_split_end

            if row_split_end - row_split_start > 5:    # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰
                image_split_row = img_thre[row_split_start:row_split_end,0:width]
                cv2.imshow('image_split_row', image_split_row)
                cv2.waitKey(0)
                cv2.destroyAllWindows()












def main():
    # 在上节的main代码上添加这几行

    # 分行
    image_row_splited = image_row_split(np.copy(image_binary))
    image_save("output/number_row_splited.png", image_row_splited)

    # 分字符
    image_char_splited = image_char_split(np.copy(image_binary), np.copy(image_row_splited))
    image_save("output/number_char_splited.png", image_char_splited)
