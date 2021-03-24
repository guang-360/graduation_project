# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # drawing= cv.resize(drawing, (640, 480))
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)


# Load source image
parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
parser.add_argument('--input', help='Path to input image.', default='drop.jpg')
args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(arg.input))
src = cv.imread('drop.jpg')
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src = cv.resize(src, (960, 720))
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3, 3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
# cv.resizeWindow(source_window, 640, 480)
max_thresh = 255
thresh = 100  # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()
cv.destroyAllWindows()

# import numpy as np
# import cv2
# import cv2 as cv
# from matplotlib import pyplot as plt

# 左键单机画圆
# def draw_circle(event, x, y, flags, param):
#     if event == cv.EVENT_LBUTTONDOWN:
#         cv.circle(img, (x, y), 100, (145, 45, 0), -1)
#
#
# 鼠标回调函数，自定义大小画矩形
# drawing = False  # 如果按下鼠标，则为真
# mode = True  # 如果为真，绘制矩形。按 m 键可以切换到曲线
# ix, iy = -1, -1
#
#
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, mode
#     if event == cv.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy = x, y
#     elif event == cv.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3)
#             else:
#                 cv.circle(img, (x, y), 5, (0, 0, 255), -1)
#     elif event == cv.EVENT_LBUTTONUP:
#         drawing = False
#         if mode == True:
#             cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 3)
#         else:
#             cv.circle(img, (x, y), 5, (0, 0, 255), -1)
#
#
# # # 创建一个黑色的图像，一个窗口，并绑定到窗口的功能
# img = np.zeros((1512, 2512, 3), np.uint8)
# cv.namedWindow('image')
# cv.setMouseCallback('image', draw_circle)
# while 1:
#     cv.imshow('image', img)
#     if cv.waitKey(20) & 0xFF == 27:
#         break
# cv.destroyAllWindows()

# 画自定义颜色点点 72-109

# def nothing(x):
#     pass
#
#
# def draw_circle(event, x, y, flags, param):
#     global radius, r, g, b
#     if event == cv.EVENT_LBUTTONDOWN:
#         cv.circle(img, (x, y), radius, (b, g, r), -1)
#
#
# # 创建一个黑色的图像，一个窗口
# img = np.zeros((1300, 2000, 3), np.uint8)
# cv.namedWindow('image')
# # 创建颜色变化的轨迹栏
# cv.createTrackbar('R', 'image', 0, 255, nothing)
# cv.createTrackbar('G', 'image', 0, 255, nothing)
# cv.createTrackbar('B', 'image', 0, 255, nothing)
# cv.createTrackbar('radius', 'image', 0, 100, nothing)
# # 为 ON/OFF 功能创建开关
# # switch = "0 : OFF \n1 : ON"
# # cv.createTrackbar(switch, 'image', 0, 1, nothing)
# cv.setMouseCallback('image', draw_circle)
# while 1:
#     cv.imshow('image', img)
#     k = cv.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     # 得到四条轨迹的当前位置
#     r = cv.getTrackbarPos('R', 'image')
#     g = cv.getTrackbarPos('G', 'image')
#     b = cv.getTrackbarPos('B', 'image')
#     radius = cv.getTrackbarPos('radius', 'image')
#     # s = cv.getTrackbarPos(switch, 'image')
#     # if s == 0:
#     #     img[:] = 0
#     # else:
#     #     img[:] = [b, g, r]
# cv.destroyAllWindows()
#

# 两张图片重叠
# img1 = cv.imread('cat.jpg')
# img2 = cv.imread('dog.jpg')
# # img1 = cv.imread('cat.jpg')
# # img2 = cv.imread('cat.jpg')
# # # eye = img[162:200,182:220]
# # # img[252:290,272:310] = eye
# # img0[:, :, 0] = 100
# # img1[:, :, 0] = 50
# # img2[:, :, 0] = 0
# # cv.imshow('no blue ',img0)
# # cv.imshow('no green ',img1)
# # cv.imshow('no red ',img2)
# # cv.waitKey(0)
#
# dst = cv.addWeighted(img1[100:300,100:400], 0.4, img2[100:300,0:400], 0.3, 0)
# cv.imshow('dst', dst)
# cv.waitKey(0)


# 摄像头实时取色
# cap = cv.VideoCapture(0)
# while 1:
#     # 读取帧
#     _, frame = cap.read()
#     # 转换颜色空间 BGR 到 HSV
#     hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     # 定义HSV中蓝色的范围
#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     # 设置HSV的阈值使得只取蓝色
#     mask = cv.inRange(hsv, lower_blue, upper_blue)
#     # 将掩膜和图像逐像素相加
#     res = cv.bitwise_and(frame, frame, mask=mask)
#     cv.imshow('frame', frame)
#     cv.imshow('mask', mask)
#     cv.imshow('res', res)
#     k = cv.waitKey(5) & 0xFF
#     if k == 27:
#         break
# cv.destroyAllWindows()

# (536,320) (1881,10) (551,3546) (1776,3882)


# 图像变形（文档矫正）
# img = cv.imread('kb.jpeg')
# rows, cols, ch = img.shape
# # 注意：numpy是（横坐标，纵坐标），cv2是（纵，横）
# pts1 = np.float32([[320, 536], [3520, 545], [5, 1810], [3882, 1776]])
# pts2 = np.float32([[0, 0], [3000, 0], [0, 1200], [3000, 1200]])
# M = cv.getPerspectiveTransform(pts1, pts2)
# dst = cv.warpPerspective(img, M, (3000, 1200))
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()


# Otsu的二值化
# from pip._vendor.msgpack.fallback import xrange
#
# img = cv.imread('drop.jpg', 0)
# ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
# ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
# ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
# ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
# ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# for i in xrange(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# 图像模糊
# 各种低通滤波器(LPF)，高通滤波器(HPF)等对图像进行滤波。
# LPF有助于消除噪声，使图像模糊等。HPF滤波器有助于在图像中找到边缘。
# img = cv.imread('drop.jpg')
# # 2D卷积（图像过滤）
# kernel = np.ones((15, 15), np.float32) / 225
# dst = cv.filter2D(img, -1, kernel)
# # 平均
# blur = cv.blur(img, (15, 15))
# # 高斯模糊
# blur_gaussian = cv.GaussianBlur(img, (15, 15), 15)
# # 中位模糊
# median = cv.medianBlur(img, 15)
# # 双边模糊
# blur_double = cv.bilateralFilter(img, 9, 75, 75)
#
# plt.subplot(231), plt.imshow(img), plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(232), plt.imshow(dst), plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.subplot(233), plt.imshow(blur), plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(234), plt.imshow(blur_gaussian), plt.title('GaussianBlurred')
# plt.xticks([]), plt.yticks([])
# plt.subplot(235), plt.imshow(median), plt.title('Median')
# plt.xticks([]), plt.yticks([])
# plt.subplot(236), plt.imshow(blur_double), plt.title('Double_edge')
# plt.xticks([]), plt.yticks([])
# plt.show()


# 腐蚀
# img = cv.imread('yummy.JPG')
# kernel = np.ones((5, 5), np.uint8)
# erosion = cv.erode(img, kernel, iterations=1)
# plt.subplot(121), plt.imshow(img), plt.title('Original')
# plt.subplot(122), plt.imshow(erosion), plt.title('Erosion')
# # plt.xticks([]), plt.yticks([])
# plt.show()


# 梯度
# img = cv.imread('drop.jpg', 0)
# laplacian = cv.Laplacian(img, cv.CV_64F)
# sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
# sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
# plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# plt.show()


# Canny边缘检测
# img = cv.imread('drop.jpg', 0)
# edges = cv.Canny(img, 30, 30)
# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()


# pic = 'drop3.jpg'
# original = cv.imread(pic)
# img = cv.imread(pic)
# imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ret, thresh = cv.threshold(imgray, 127, 255, 8)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# con = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
# # cnt = contours[3]
# # con2 = cv.drawContours(img, [cnt], -1, (0,255,0), 2)
# plt.subplot(121), plt.title('Original'), plt.imshow(original)
# plt.subplot(122), plt.title('THRESH_TRIANGLE'), plt.imshow(con)
# # plt.subplot(122), plt.title('Contour'), plt.imshow(con)
# for cnt in contours:
#     perimeter = cv.arcLength(cnt, True)
#     print(perimeter)
# plt.show()


# pic = 'drop.jpg'
# original = cv.imread(pic)
#
# img2 = cv.imread(pic, 2)
# # imread(pic, flag)
# # flag=-1时，8位深度，原通道
# # flag=0，8位深度，1通道 $
# # flag=1，8位深度，3通道
# # flag=2，原深度，1通道 $
# # flag=3, 原深度，3通道
# # flag=4，8位深度 ，3通道

# imgray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
#
# ret, thresh = cv.threshold(imgray, 127, 255, 8)
# # 当使用了THRESH_OTSU和THRESH_TRIANGLE两个标志时，输入图像必须为单通道。
# #  ret： 与参数thresh一致
# #  dst： 结果图像
# # 参数8表示使用THRESH_TRIANGLE
#
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# con = cv.drawContours(img2, contours, -1, (0, 255, 255), 3)
# # cnt = contours[3]
# # con2 = cv.drawContours(img, [cnt], -1, (0,255,0), 2)
#
# # canny边缘检测
#
# # edges2 = cv.Canny(img2, 30, 30)
# # plt.subplot(132), plt.title('canny0'), plt.imshow(edges2, cmap='gray')
#
# plt.subplot(121), plt.title('Original'), plt.imshow(original)
# plt.subplot(122), plt.title('THRESH_TRIANGLE'), plt.imshow(con)
#
#
# # for cnt in contours:
# #     perimeter = cv.arcLength(cnt, True)
# #     print(perimeter)
# plt.show()


# 抽象
# https://blog.csdn.net/niuxuerui11/article/details/108302993
# img = cv2.pyrDown(cv2.imread("drop.jpg", cv2.IMREAD_UNCHANGED))
#
# ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# black = cv2.cvtColor(np.zeros((img.shape[1], img.shape[0]), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
#
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for cnt in contours:
#     epsilon = 0.01 * cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     hull = cv2.convexHull(cnt)
#     cv2.drawContours(black, [cnt], -1, (0, 255, 0), 2)
#     cv2.drawContours(black, [approx], -1, (255, 255, 0), 2)
#     cv2.drawContours(black, [hull], -1, (0, 0, 255), 2)
# cv2.imshow("hull", black)
# cv2.waitKey()
# cv2.destroyAllWindows()


# findcontour
# mode
# 检索模式，可取值如下：
# CV_RETR_EXTERNAL：只检索最外面的轮廓；
# CV_RETR_LIST：检索所有的轮廓，并将其放入list中；
# CV_RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界；
# CV_RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次。
#
# method
# 边缘近似方法（除了CV_RETR_RUNS使用内置的近似，其他模式均使用此设定的近似算法）。可取值如下：
# CV_CHAIN_CODE：以Freeman链码的方式输出轮廓，所有其他方法输出多边形（顶点的序列）。
# CV_CHAIN_APPROX_NONE：将所有的连码点，转换成点。
# CV_CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
# CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS：使用the flavors of Teh-Chin chain近似算法
# 的一种。
# CV_LINK_RUNS：通过连接水平段的1，使用完全不同的边缘提取算法。使用CV_RETR_LIST检索模式能使用此方法。


# 轮廓特征
# img = cv.imread('drop2.jpg', 0)
# ret, thresh = cv.threshold(img, 127, 255, 0)
# contours, hierarchy = cv.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv.moments(cnt)
# print(M)


# img = cv2.imread('drop.jpg')
#
# ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, 0)
# _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 得到轮廓信息
# cnt = contours[0]  # 取第一条轮廓
# M = cv2.moments(cnt)  # 计算第一条轮廓的矩
#
# imgnew = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)  # 把所有轮廓画出来
# print(M)
# # 这两行是计算中心点坐标
# cx = int(M['m10'] / M['m00'])
# cy = int(M['m01'] / M['m00'])
#
# # 计算轮廓所包含的面积
# area = cv2.contourArea(cnt)
#
# # 计算轮廓的周长
# perimeter = cv2.arcLength(cnt, True)
#
# # 轮廓的近似
# epsilon = 0.02 * perimeter
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# imgnew1 = cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
#
# cv2.imshow('lunkuo', imgnew)
# cv2.imshow('approx_lunkuo', imgnew1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(0)
# cv2.waitKey(0)
# cv2.waitKey(0)
# cv2.waitKey(0)
#
# img = cv.imread('drop3.jpg', 0)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4)
# ret, thresh = cv.threshold(th3, 127, 255, 8)
#
# # contours, hierarchy = cv.findContours(th3, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# #
# # # newImage = cv2.imread('white.jpg')
# # # newImage = cv2.resize(newImage,(2560,1920))
# # con = cv.drawContours(img, contours, -1, (0, 255, 0), 3)
# # plt.imshow(th3, cmap='gray')
# # plt.show()
#
#
# titles = ['Original', 'Global Thresholding(v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()
