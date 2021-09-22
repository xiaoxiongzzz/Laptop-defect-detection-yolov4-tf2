'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import tensorflow as tf
from PIL import Image
import cv2 as cv
import os
import numpy as np
from yolo import YOLO
import sys

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

yolo = YOLO()
#将图片填充为正方形
def fill_image(image):
    width, height = image.size
    #选取长和宽中较大值作为新图片的
    new_image_length = width if width > height else height
    #生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    #将之前的图粘贴在新图上，居中
    if width > height:#原图宽大于高，则填充图片的竖直维度
        #(x,y)二元组表示粘贴上图相对下图的起始位置
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image,(int((new_image_length - width) / 2),0))

    return new_image
#切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0,3):#两重循环，生成9张图片基于原图的位置
        for j in range(0,3):
            #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list
#差分处理
def chafen_image(file_path):
    src1 = cv.imread(file_path2)
    src2 = cv.imread(file_path)
    # subtract = cv.subtract(src1, src2, dst=None, mask=None, dtype=None)
    subtracted = cv.subtract(src2, src1)#将图像image与M相减
    # cv.imshow("Subtracted", subtracted)
    return subtracted

while True:
    # file_path3 = 'img/ceshi.tiff'
    file_path = "img/"
    # 模板位置
    file_path2 = "img/WPS图片拼图.png"

    # img1 = input('Input image filename:')
    # img2 = input('Input image filename:')
    # img3 = input('Input image filename:')
    # img4 = input('Input image filename:')
    # img5 = input('Input image filename:')
    # img6 = input('Input image filename:')
    # img7 = input('Input image filename:')
    # img8 = input('Input image filename:')
    # img9 = input('Input image filename:')
    try:

            image = Image.open(file_path)
            image2 = Image.open(file_path2)
            # image3 = Image.open("photo2.jpg")
            # image4 = Image.open(img4)
            # image5 = Image.open(img5)
            # image6 = Image.open(img6)
            # image7 = Image.open(img7)
            # image8 = Image.open(img8)
            # image9 = Image.open(img9)
            image = fill_image(image)
            image_list = cut_image(image)
            # 深拷贝
            image_list2 = image_list





    except:
        print('Open Error! Try again!')
        # continue
    else:
        image_chafenhou = chafen_image(file_path)
        # cv.imwrite("photo2.jpg", image_chafenhou)
        # image3 = Image.open("photo2.jpg")
        # image_cha = fill_image(image3)
        # image_chafenlist = cut_image(image_cha)

        r_image1 = yolo.detect_image(image_list[0])
        r_image2 = yolo.detect_image(image_list[1])
        r_image3 = yolo.detect_image(image_list[2])
        r_image4 = yolo.detect_image(image_list[3])
        r_image5 = yolo.detect_image(image_list[4])
        r_image6 = yolo.detect_image(image_list[5])
        r_image7 = yolo.detect_image(image_list[6])
        r_image8 = yolo.detect_image(image_list[7])
        r_image9 = yolo.detect_image(image_list[8])
        r_image = np.hstack((np.vstack((r_image1, r_image4, r_image7)), np.vstack((r_image2, r_image5, r_image8)), np.vstack((r_image3, r_image6, r_image9))))
        cv.imwrite("photo1.jpg", r_image)

        # yolo.generate2()
        # c_image1 = yolo.detect_image2(image_chafenlist[0], image_list2[0])
        # c_image2 = yolo.detect_image2(image_chafenlist[1], image_list2[1])
        # c_image3 = yolo.detect_image2(image_chafenlist[2], image_list2[2])
        # c_image4 = yolo.detect_image2(image_chafenlist[3], image_list2[3])
        # c_image5 = yolo.detect_image2(image_chafenlist[4], image_list2[4])
        # c_image6 = yolo.detect_image2(image_chafenlist[5], image_list2[5])
        # c_image7 = yolo.detect_image2(image_chafenlist[6], image_list2[6])
        # c_image8 = yolo.detect_image2(image_chafenlist[7], image_list2[7])
        # c_image9 = yolo.detect_image2(image_chafenlist[8], image_list2[8])
        # c_image = np.hstack((np.vstack((c_image1, c_image4, c_image7)), np.vstack((c_image2, c_image5, c_image8)),
        #                      np.vstack((c_image3, c_image6, c_image9))))
        # cv.imwrite("photo3.jpg", c_image)
        # # os.remove("photo2.jpg")
        # img = Image.open('photo1.jpg')
        # img.show()
        # img2 = Image.open('photo3.jpg')
        # img2.show()
        # os.system("pause")

