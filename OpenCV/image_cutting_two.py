# -*- coding: utf-8 -*-
import cv2
from PIL import Image

def cut_pic(filename):  # 图片处理（灰度化，二值化，切割图片）
    filepath = 'C:\\Users\\dby_freedom\\Desktop\\661\\Image\\1.jpg'
    im = Image.open(filepath)
    imgry = im.convert('L')  # 灰度化
    # imgry.show()
    # 二值化
    threshold = 130
    table = []
    cut = []
    realcut = []
    for i in range(256):
        if i < threshold:
            table.append(0)
        else:
            table.append(1)
    out = imgry.point(table, '1')
    # out.show()

    # 分割图片
    width = out.width
    height = out.height
    # 取有像素0的列
    for x in range(0, width):
        for y in range(0, height):
            if out.getpixel((x, y)) == 0:
                cut.append(x)
                break
            else:
                continue
    # 保存要切割的列
    realcut.append(cut[0] - 1)
    for i in range(0, len(cut) - 1):
        if cut[i + 1] - cut[i] > 1:
            realcut.append(cut[i] + 1)
            realcut.append(cut[i + 1] - 1)
        else:
            continue
    realcut.append(cut[-1] + 1)
    # 切割图片
    count = [0, 2, 4, 6]
    child_img_list = []
    for i in count:
        child_img = out.crop((realcut[i], 0, realcut[i + 1], height))
        child_img_list.append(child_img)
    # 保存切割的图片
    for i in range(0,4):
        child_img_list[i].save("C:\\Users\\dby_freedom\\Desktop\\661\\Image\\%d.jpg" % i)

    # 横向切割
    cut_second = []
    final_img_list = []
    for i in range(0, 4):
        width = child_img_list[i].width
        height = child_img_list[i].height
        # 取有像素0的列
        for y in range(0, height):
            for x in range(0, width):
                if child_img_list[i].getpixel((x, y)) == 0:
                    cut_second.append(y)
                    break
                else:
                    continue
        # 切割图片
        final_img = child_img_list[i].crop((0, cut_second[0] - 1, width, cut_second[-1] + 1))
        final_img_list.append(final_img)
    # 返回切割的图片
    return final_img_list


if __name__ == '__main__':
    m = cut_pic('C:\\Users\\dby_freedom\\Desktop\\661\\Image\\1.jpg')
