import cv2
import numpy as np
from matplotlib import pyplot as plt
# # 1、读取图像，并把图像转换为灰度图像并显示
# img = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\1.jpg')  # 读取图片
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)
#
# # 2、将灰度图像二值化，设定阈值是100
# img_thre = img_gray
# cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV, img_thre)
# cv2.imshow('threshold', img_thre)
# cv2.waitKey(0)

# def image2binarg(image_path):
#     # 1、读取图像，并把图像转换为灰度图像并显示
#     img = cv2.imread(image_path)  # 读取图片
#     # print(img.shape())
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
#     # img_gray = cv2.imread(self.image_path, 0)
#     cv2.imshow('gray', img_gray)  # 显示图片
#     cv2.waitKey(0)
#
#     # 2、将灰度图像二值化，设定阈值是50
#     # img_thre = img_gray
#     # ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
#     _, img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
#     # th3 = cv.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#     #                            cv2.THRESH_BINARY, 11, 2)
#     cv2.imshow('binary', img_binary)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     return img_binary

class image_precessing():
    def __init__(self, image_path = None):

        self.image_path = image_path
        if self.image_path is None:
            raise('Please input the image path')
        else:
            pass
    #
    # def image2binarg(self):
    #     # 1、读取图像，并把图像转换为灰度图像并显示
    #     img = cv2.imread(self.image_path)  # 读取图片
    #     # print(img.shape())
    #     # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
    #     img_gray = cv2.imread(self.image_path, 0)
    #     # cv2.imshow('gray', img_gray)  # 显示图片
    #     # cv2.waitKey(0)
    #
    #     # 2、将灰度图像二值化，设定阈值是50
    #     # img_thre = img_gray
    #     ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    #     _,img_binary = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    #     # th3 = cv.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #     #                            cv2.THRESH_BINARY, 11, 2)
    #     cv2.imshow('binary',img_binary)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    #     return img_binary

    def img_read(self):
        img_raw = cv2.imread(self.image_path)
        img_grav = cv2.imread(self.image_path, 0)
        img_alpha = cv2.imread(self.image_path, -1)
        img_color_one = cv2.imread(self.image_path, 1)
        img_color_two = cv2.imread(self.image_path, 2)
        img_color_three = cv2.imread(self.image_path, 3)
        # img_color_red = cv2.imread('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\'+ input_image,CV_LOAD_IMAGE_COLOR ='b')
        # plt.subplot(231),plt.imshow(img_raw,'gray'),plt.title('ORIGINAL')
        # plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
        # plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
        # plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
        # plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
        # plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

        cv2.imshow('img_raw', img_raw)
        cv2.waitKey(0)
        cv2.imshow('img_grav', img_grav)
        cv2.waitKey(0)
        cv2.imshow('img_alpha', img_alpha)
        cv2.waitKey(0)
        cv2.imshow('img_color_one', img_color_one)
        cv2.waitKey(0)
        cv2.imshow('img_color_two', img_color_two)
        cv2.waitKey(0)
        cv2.imshow('img_color_three', img_color_three)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def img2threshold(self):
        '''Simple Thresholding'''

        img_raw = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        img_gray_three = img_gray.copy()

        binary_range = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        # binary_range = range(30,40)
        for binary_depth in binary_range:
            ret, thresh1 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_BINARY)
            ret, thresh2 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_BINARY_INV)
            ret, thresh3 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TRUNC)
            ret, thresh4 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TOZERO)
            ret, thresh5 = cv2.threshold(img_gray, binary_depth, 255, cv2.THRESH_TOZERO_INV)

            titles = ['Original Image' + str(binary_depth), 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
            images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

            for i in range(6):
                plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
                plt.show()

    def img2ada_threshold(self):
        '''Adaptive Thresholding'''
        img_raw = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)

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

    def img2ostu_threshold(self):
        '''Otsu’s Binarization'''

        img_raw = cv2.imread(self.image_path)
        img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
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

    def erosion(self):
        img = self.image2binarg()
        # img = cv2.imread(self.image_path)
        kernel = np.ones((3,3),np.uint8)
        img_erosion = cv2.erode(img,kernel,iterations = 1)
        cv2.imshow('gray',img_erosion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img_erosion

    def dilation(self):
        img = self.image2binarg()
        # img = cv2.imread(image_path,0)
        kernel = np.ones((3,3),np.uint8)
        img_erosion = cv2.dilate(img,kernel,iterations = 1)
        cv2.imshow('gray',img_erosion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img_erosion


if __name__ == '__main__':

    img_path = 'C:\\Users\\dby_freedom\\Desktop\\661\\Image\\3.png'
    img_precessing = image_precessing(image_path = img_path)
    # shoew binary
    img_precessing.img2threshold()
    # img_binary = img_precessing.image2binarg()
    # # show dilation
    # img_dilation = img_precessing.dilation()
    # # show erosion
    # img_erosion = img_precessing.erosion()
