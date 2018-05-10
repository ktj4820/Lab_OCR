import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from sklearn.cluster import KMeans


def image_read( filename, mode = None ):

    if mode != None:
        return cv2.imread( filename, mode )
    else:
        return cv2.imread( filename )

def image_save( filename, image ):

    return cv2.imwrite( filename, image )

def img_read(image_name):
    img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name)
    img_grav = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name,0)
    img_alpha = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name,-1)
    img_color_one = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name,1)
    img_color_two = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name,2)
    img_color_three = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ image_name,3)
    # img_color_red = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ input_image,CV_LOAD_IMAGE_COLOR ='b')
    # plt.subplot(231),plt.imshow(img_raw,'gray'),plt.title('ORIGINAL')
    # plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
    # plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
    # plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
    # plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
    # plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

    # cv2.imshow('img_raw', img_raw)
    # cv2.waitKey(0)
    # cv2.imshow('img_grav', img_grav)
    # cv2.waitKey(0)
    # cv2.imshow('img_alpha', img_alpha)
    # cv2.waitKey(0)
    # cv2.imshow('img_color_one', img_color_one)
    # cv2.waitKey(0)
    # cv2.imshow('img_color_two', img_color_two)
    # cv2.waitKey(0)
    # cv2.imshow('img_color_three', img_color_three)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img_raw


def find_backgraund_color(img, cluster):
    clt = KMeans(n_clusters=cluster)
    clt.fit(img)

def ImageFilter_DETAIL(img_address):
    img = Image.open(img_address)
    img.filter(ImageFilter.DETAIL)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    gray = img.convert('L')  # 转化为灰度图
    plt.axis('off')  # 不显示坐标轴
    gray.show()
    Binary = img.convert('1')  # 转化为二值化图
    plt.axis('off')  # 不显示坐标轴
    Binary.show()

def ImageFilter_EDGE_ENHANCE(img_address):
    img = Image.open(img_address)
    img.filter(ImageFilter.EDGE_ENHANCE)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    gray = img.convert('L')  # 转化为灰度图
    plt.axis('off')  # 不显示坐标轴
    gray.show()
    Binary = img.convert('1')  # 转化为二值化图
    plt.axis('off')  # 不显示坐标轴
    Binary.show()


def ImageFilter_EDGE_ENHANCE_MORE(img_address):
    img = Image.open(img_address)
    img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    gray = img.convert('L')  # 转化为灰度图
    plt.axis('off')  # 不显示坐标轴
    gray.show()
    Binary = img.convert('1')  # 转化为二值化图
    plt.axis('off')  # 不显示坐标轴
    Binary.show()

def ImageFilter_FIND_EDGES(img_address):
    img = Image.open(img_address)
    img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    gray = img.convert('L')  # 转化为灰
     # 度图
    plt.axis('off')  # 不显示坐标轴
    gray.show()
    Binary = img.convert('1')  # 转化为二值化图
    plt.axis('off')  # 不显示坐标轴
    Binary.show()


def img2max_gray(img_raw):
    '''max gray method'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # # max_gray = cv2.CreateImage(cv2.GetSize(img_raw), img_raw.depth, 1)
    # print(img_raw.shape)
    # # max_gray[]
    # # # grayimg= cv2.CreateImage(cv.GetSize(image), image.depth, 1)
    
    img = img_raw
    max_gray = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
    b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    b[:, :] = img[:, :, 0]
    g[:, :] = img[:, :, 1]
    r[:, :] = img[:, :, 2]

    # max_gray[:, :] = (b[:, :], g[:, :], r[:, :])
    print(b[:,:])
    cv2.imshow("Blue", b)
    cv2.imshow("Red", r)
    cv2.imshow("Green", g)
    cv2.imshow("Max_Gray",max_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def img2avg_gray(img_raw):
    '''average gray method'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # # max_gray = cv2.CreateImage(cv2.GetSize(img_raw), img_raw.depth, 1)
    # print(img_raw.shape)
    # # max_gray[]
    # # # grayimg= cv2.CreateImage(cv.GetSize(image), image.depth, 1)
    img = img_raw
    avg_gray = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
    b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    b[:, :] = img[:, :, 0]
    g[:, :] = img[:, :, 1]
    r[:, :] = img[:, :, 2]

    avg_gray[:, :] = (b[:, :] +  g[:, :] + r[:, :] )/3

    cv2.imshow("Blue", b)
    cv2.imshow("Red", r)
    cv2.imshow("Green", g)
    cv2.imshow("Avg_Gray",avg_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def img2weight_gray(img_raw):
    '''average gray method'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # # max_gray = cv2.CreateImage(cv2.GetSize(img_raw), img_raw.depth, 1)
    # print(img_raw.shape)
    # # max_gray[]
    # # # grayimg= cv2.CreateImage(cv.GetSize(image), image.depth, 1)
    #
    img = img_raw
    weight_gray = np.zeros((img.shape[0],img.shape[1]), dtype=img.dtype)
    b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    b[:, :] = img[:, :, 0]
    g[:, :] = img[:, :, 1]
    r[:, :] = img[:, :, 2]

    weight_gray[:, :] = 0.11 * b[:, :] + 0.59 * g[:, :] +  0.3 * r[:, :]

    cv2.imwrite("Weight_Gray.png",weight_gray)
    cv2.imshow("Blue", b)
    cv2.imshow("Red", r)
    cv2.imshow("Green", g)
    cv2.imshow("Weight_Gray",weight_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return weight_gray

def img2thresh(img_gray, threshold = None):
    '''Simple Thresholding'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)

    if threshold == None:
        binary_range = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # binary_range = range(30,40)
        for binary_depth in binary_range:
            ret, thresh1 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_BINARY_INV)
            ret, thresh3 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TRUNC)
            ret, thresh4 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TOZERO)
            ret, thresh5 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TOZERO_INV)

            titles = ['Original Image'+ str(binary_depth), 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
            images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

            for i in range(6):
                plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])

            plt.show()
    else:
        ret, thresh1 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_TOZERO_INV)

        titles = ['Original Image' + str(threshold), 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
        images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

        for i in range(6):
            plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])

        plt.show()
        cv2.imwrite('img2thresh.png', thresh2)
        return thresh2
        # image_save('image_name' + '_' + 'threshold_binary.png', thresh1)


def img2adathresh(img_gray):
    '''Adaptive Thresholding'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # img_gray = cv2.cvtColor(img_raw,cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img_gray, 3)
    ret, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 5, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 5, 2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]

    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def img2otsuthresh(img_gray):
    '''Otsu’s Binarization'''
    # img_raw = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\' + image_name)
    # img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

    # global thresholding
    img = img_gray
    ret1, th1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()

def rowsplit(img_thre):

    row_white_num = []  # 记录每一行的白色像素总和
    row_black_num = []  # ..........黑色.......
    image_height = img_thre.shape[0]
    width = img_thre.shape[1]
    row_white_num_max = 0
    row_black_num_max = 0
    # 计算每一行的黑白色像素总和
    for i in range(image_height):
        white_num = 0  # 这一行白色总数
        black_num = 0  # 这一行黑色总数
        for j in range(width):
            if img_thre[i][j] == 255:
                white_num += 1
            if img_thre[i][j] == 0:
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
            if (row_black_num[m] if arg else row_white_num[m]) > (
                    0.95 * row_black_num_max if arg else 0.95 * row_white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    arg = False
    '''arg = True表示黑底白字, arg = False表示白底黑字'''
    if row_black_num_max > row_white_num_max:
        arg = True

    row_mark = 1
    while row_mark < image_height - 2:
        row_mark += 1
        '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
        即将该行看做出现字符的起始行。
        对于白底黑字情况，则反之'''
        if (row_white_num[row_mark] if arg else row_black_num[row_mark]) > (0.05 * row_white_num_max if arg else 0.05 * row_black_num_max):
            row_split_start = row_mark - 2 if row_mark > 2 else row_mark
            row_find_end = find_end(row_split_start)
            row_split_end = row_find_end + 2 if row_find_end + 2 < image_height else row_find_end
            row_mark = row_split_end
            if row_find_end - row_split_start > 5:  # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰
                image_split_row = img_thre[row_split_start:row_split_end, 0:width]
                cv2.imshow('image_split_row', image_split_row)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

def columnplit(img_thre):
    column_white_num = []  # 记录每一列的白色像素总和
    column_black_num = []  # ..........黑色.......
    image_height = img_thre.shape[0]
    image_width = img_thre.shape[1]
    column_white_num_max = 0
    column_black_num_max = 0
    # 计算每一列的黑白色像素总和
    for i in range(image_width):
        white_num = 0  # 这一列白色总数
        black_num = 0  # 这一列黑色总数
        for j in range(image_height):
            if img_thre[j][i] == 255:
                white_num += 1
            if img_thre[j][i] == 0:
                black_num += 1
        column_white_num_max = max(column_white_num_max, white_num)
        column_black_num_max = max(column_black_num_max, black_num)
        column_white_num.append(white_num)    # 记录该列的白色像素总数
        column_black_num.append(black_num)    # 记录该行的黑色像素总数

    # 分割图像，提取字符列
    def find_end(start_column):
        end_column = start_column + 1
        '''arg = True，黑底白字情况下：对于第m行，如果该行黑色像素大于
        0.95*（所有行中黑色像素数目和的最大值），则证明该列包含的白色字符太少，判定次列即为字符切割结束列；
        对于白底黑字情况，则反之'''
        for m in range(start_column + 1, image_width + 1):
            # if (column_black_num[m] if arg else column_white_num[m]) > \
            #         (0.95 * column_black_num_max if arg else 0.95 * column_white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
            #     end_column = m
            #     break
            if (column_white_num[m] if arg else column_black_num[m]) < \
                    0.15 * (column_white_num[m] + column_black_num[m]):  # 0.95这个参数请多调整，对应下面的0.05
                end_column = m
                break
        return end_column

    arg = False
    '''arg = True表示黑底白字, arg = False表示白底黑字'''
    if column_black_num_max > column_white_num_max:
        arg = True

    column_mark = 1
    while column_mark < image_width - 2:
        column_mark += 1
        '''arg = True，即黑底白字情况下，第n列的白色像素数目和 > 0.05 * （所有列中白色像素数目和的最大值），
        即将该行看做出现字符的起始列。
        对于白底黑字情况，则反之'''
        if (column_white_num[column_mark] if arg else column_black_num[column_mark]) > \
                (0.05 * column_white_num_max if arg else 0.05 * column_black_num_max):
            column_find_start = column_mark
            column_split_start = column_find_start - 2 if column_find_start > 2 else column_find_start
            column_find_end = find_end(column_mark)
            column_split_end = column_find_end + 2 if column_find_end + 2 < image_width else column_find_end
            column_mark = column_find_end
            if column_find_end - column_find_start > 5:  # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰
                image_split_row = img_thre[1:image_height, column_split_start:column_split_end]
                cv2.imshow('image_split_column', image_split_row)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def charsplit(img_thre):

    row_white_num = []  # 记录每一行的白色像素总和
    row_black_num = []  # ..........黑色.......
    image_height = img_thre.shape[0]
    width = img_thre.shape[1]
    row_white_num_max = 0
    row_black_num_max = 0
    # 计算每一行的黑白色像素总和
    for i in range(image_height):
        white_num = 0  # 这一行白色总数
        black_num = 0  # 这一行黑色总数
        for j in range(width):
            if img_thre[i][j] == 255:
                white_num += 1
            if img_thre[i][j] == 0:
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
            if (row_black_num[m] if arg else row_white_num[m]) > (
                    0.95 * row_black_num_max if arg else 0.95 * row_white_num_max):  # 0.95这个参数请多调整，对应下面的0.05
                end_ = m
                break
        return end_

    arg = False
    '''arg = True表示黑底白字, arg = False表示白底黑字'''
    if row_black_num_max > row_white_num_max:
        arg = True

    row_mark = 1
    while row_mark < image_height - 2:
        row_mark += 1
        '''arg = True，即黑底白字情况下，第n行的白色像素数目和 > 0.05 * （所有行中白色像素数目和的最大值），
        即将该行看做出现字符的起始行。
        对于白底黑字情况，则反之'''
        if (row_white_num[row_mark] if arg else row_black_num[row_mark]) > (0.05 * row_white_num_max if arg else 0.05 * row_black_num_max):
            row_split_start = row_mark - 2 if row_mark > 2 else row_mark
            row_find_end = find_end(row_split_start)
            row_split_end = row_find_end + 2 if row_find_end + 2 < image_height else row_find_end
            row_mark = row_split_end
            if row_find_end - row_split_start > 5:  # 要求切割字符行需要有5行以上的距离是为了保证排除直线的干扰
                image_split_row = img_thre[row_split_start:row_split_end, 0:width]
                cv2.imshow('image_split_row', image_split_row)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                rowsplit(image_split_row)

def img_processing():

    input_image = "1.jpg"
    img_raw = img_read(input_image)
    weight_gray = img2weight_gray(img_raw)
    thresh_image = img2thresh(weight_gray, threshold=30)
    ImageFilter_DETAIL('C:\\Users\\dby_freedom\\Desktop\\661\\OpenCV\\img2thresh.png')
    ImageFilter_EDGE_ENHANCE('C:\\Users\\dby_freedom\\Desktop\\661\\OpenCV\\img2thresh.png')
    ImageFilter_EDGE_ENHANCE_MORE('C:\\Users\\dby_freedom\\Desktop\\661\\OpenCV\\img2thresh.png')
    ImageFilter_FIND_EDGES('C:\\Users\\dby_freedom\\Desktop\\661\\OpenCV\\img2thresh.png')

if __name__ == '__main__':

    input_image = "11.jpg"
    img_raw = img_read(input_image)

    # find_backgraund_color()
    find_backgraund_color(img_raw)


    # rowsplit(thresh_image)
    # columnplit(thresh_image)

    # img2avg_gray(input_image)
    # img2max_gray(input_image)
    # img2weight_gray(input_image)
    # img2thresh(input_image,30)
    # rowsplit(input_image)