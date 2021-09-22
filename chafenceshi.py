import cv2 as cv
from PIL import Image

def chafen_image():
    src1 = cv.imread('img/456.jpg')
    src2 = cv.imread('img/123.jpg')
    # subtract = cv.subtract(src1, src2, dst=None, mask=None, dtype=None)
    subtracted = cv.subtract(src2, src1)#将图像image与M相减
    # cv.imshow("Subtracted", subtracted)
    return subtracted
file_path = "img/123.jpg"
image = Image.open(file_path)
image_chafenhou = chafen_image()
cv.imwrite("photo2.jpg", image_chafenhou)