# # This is a sample Python script.
#
# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
# import cv2 as cv
# import numpy as np
# from matplotlib import pyplot as plt

import os
import cv2
import cv2 as cv
import numpy as np
import random as rng
import time
from matplotlib import pyplot as plt


# part1
# pic = 'drop0.jpg'
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

# part2
# img = cv.imread('drop0.jpg', 0)
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

#
# 引入图像
# pic = 'drop0.jpg'
# black = np.zeros((1920, 2560, 3), np.uint8)  #黑色背景
# img1 = cv.imread(pic, 0)    #灰度图像
#
#
# # # 2D卷积（图像过滤）
# # kernel = np.ones((15, 15), np.float32) / 225
# # dst = cv.filter2D(img1, -1, kernel)
# # # 平均
# # blur = cv.blur(img1, (15, 15))
# # # 高斯模糊
# # blur_gaussian = cv.GaussianBlur(img1, (15, 15), 15)
# # # 中位模糊
# # median = cv.medianBlur(img1, 15)
# # # 双边模糊
# # blur_double = cv.bilateralFilter(img1, 9, 75, 75)
#
#
# # 图像自适应阈值处理
# # blockSize越大，细节轮廓越少；C越大，阈值越小，应偏大
# th1 = cv.adaptiveThreshold(img1, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 89, 25)
# th1 = 255 - th1
# plt.subplot(1, 2, 1), plt.imshow(img1, 'gray')
# plt.subplot(1, 2, 2), plt.imshow(th1, 'gray')
# plt.show()

# canny边缘检测
# minval = [40, 50, 60, 70]
# maxval = [60, 65, 70, 75]
# for i in range(4):
#     canny = cv.Canny(th1, 50, maxval[i])
#     plt.subplot(2, 2, i+1), plt.imshow(canny, 'gray')
#
# plt.show()
#
#
# findContours寻找轮廓
# canny = cv.Canny(th1, 50, 75)
# ret, thresh = cv.threshold(canny, 127, 255, 8)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# findContours显示结果
# 利用面积筛选
# for cnt in contours:
#     area = cv.contourArea(cnt)
#     if area > 100:
#         con = cv.drawContours(black, cnt, -1, (0, 255, 0), 3)

# 最小包围矩形
# cnt = contours[5]
# for cnt in contours:
#     rect = cv.minAreaRect(cnt)
#     box = cv.boxPoints(rect)
#     box = np.int0(box)
#     area = cv.contourArea(box)
#     if area > 10000:
#         cv.drawContours(th1, [box], 0, (25, 155, 255), 2)
#         print(area)
#
# plt.imshow(th1)
# plt.show()
#
# # 测量长度
#
#
# # 模糊处理结果显示
# # titles = ['Origianl', '2d', 'mean', 'gaussian', 'median', 'blur_double']
# # images = [img1, dst, blur, blur_gaussian, median, blur_double]
# # for i in range(6):
# #     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]), plt.yticks([])
# # plt.show()


# coding=utf-8
# 导入一些后续需要使用到的python包
# 可能需要 pip install  imutils
# from scipy.spatial import distance as dist
# from imutils import perspective
# from imutils import contours
# import numpy as np
# import argparse
# import imutils
# import cv2
#
#
# # 定义一个中点函数，后面会用到
# def midpoint(ptA, ptB):
#     return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
#
#
# # 设置一些需要改变的参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to the input image")
# ap.add_argument("-w", "--width", type=float, required=True,
#                 help="width of the left-most object in the image (in inches)")
# args = vars(ap.parse_args())
#
# # 读取输入图片
# image = cv2.imread(args["image"])
# # 输入图片灰度化
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 对灰度图片执行高斯滤波
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
#
# # 对滤波结果做边缘检测获取目标
# edged = cv2.Canny(gray, 50, 100)
# # 使用膨胀和腐蚀操作进行闭合对象边缘之间的间隙
# edged = cv2.dilate(edged, None, iterations=1)
# edged = cv2.erode(edged, None, iterations=1)
#
# # 在边缘图像中寻找物体轮廓（即物体）
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
#
# # 对轮廓按照从左到右进行排序处理
# (cnts, _) = contours.sort_contours(cnts)
# # 初始化 'pixels per metric'
# pixelsPerMetric = None
#
# # 循环遍历每一个轮廓
# for c in cnts:
#     # 如果当前轮廓的面积太少，认为可能是噪声，直接忽略掉
#     if cv2.contourArea(c) < 100:
#         continue
#
#     # 根据物体轮廓计算出外切矩形框
#     orig = image.copy()
#     box = cv2.minAreaRect(c)
#     box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
#     box = np.array(box, dtype="int")
#
#     # 按照top-left, top-right, bottom-right, bottom-left的顺序对轮廓点进行排序，并绘制外切的BB，用绿色的线来表示
#     box = perspective.order_points(box)
#     cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
#
#     # 绘制BB的4个顶点，用红色的小圆圈来表示
#     for (x, y) in box:
#         cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
#
#     # 分别计算top-left 和top-right的中心点和bottom-left 和bottom-right的中心点坐标
#     (tl, tr, br, bl) = box
#     (tltrX, tltrY) = midpoint(tl, tr)
#     (blbrX, blbrY) = midpoint(bl, br)
#
#     # 分别计算top-left和top-right的中心点和top-righ和bottom-right的中心点坐标
#     (tlblX, tlblY) = midpoint(tl, bl)
#     (trbrX, trbrY) = midpoint(tr, br)
#
#     # 绘制BB的4条边的中心点，用蓝色的小圆圈来表示
#     cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
#     cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
#
#     # 在中心点之间绘制直线，用紫红色的线来表示
#     cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
#              (255, 0, 255), 2)
#     cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
#              (255, 0, 255), 2)
#
#     # 计算两个中心点之间的欧氏距离，即图片距离
#     dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
#     dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
#
#     # 初始化测量指标值，参考物体在图片中的宽度已经通过欧氏距离计算得到，参考物体的实际大小已知
#     if pixelsPerMetric is None:
#         pixelsPerMetric = dB / args["width"]
#
#     # 计算目标的实际大小（宽和高），用英尺来表示
#     dimA = dA / pixelsPerMetric
#     dimB = dB / pixelsPerMetric
#
#     # 在图片中绘制结果
#     cv2.putText(orig, "{:.1f}in".format(dimA),
#                 (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65, (255, 255, 255), 2)
#     cv2.putText(orig, "{:.1f}in".format(dimB),
#                 (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65, (255, 255, 255), 2)
#
#     # 显示结果
#     cv2.imshow("Image", orig)
#     cv2.waitKey(0)


#
# # Load source image
# parser = argparse.ArgumentParser(description='Code for Finding contours in your image tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='drop.jpg')
# args = parser.parse_args()
# # src = cv.imread(cv.samples.findFile(arg.input))
# src = img[120:1800, 160:2400]  # 去掉上下文字信息，避免干扰（src比例为4：3）
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# # src = cv.resize(src, (960, 720))
# src = cv.resize(src, (720, 540))
# # Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.GaussianBlur(src_gray, (3, 3), 1)
# # Create Window
# source_window = 'Source'
# cv.namedWindow(source_window)
# cv.imshow(source_window, src)
# # cv.resizeWindow(source_window, 640, 480)
# max_thresh = 255
# thresh = 100  # initial threshold
# cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)
# cv.waitKey()
# cv.destroyAllWindows()


def find_thresh(picture, contours_number, show):
    """picture为待处理图像，contours_number为预期获得轮廓的数量，show如果为1将即时显示图像结果"""
    rng.seed(1895)
    # 读取图像
    img = cv.imread(picture)

    # 早期照片裁剪
    # img = img[120:1800, 160:2400]  # 去掉上下文字信息，避免干扰（src比例为4：3）
    # src = cv.resize(img, (720, 540))

    # 自拍照片裁剪
    img = img[120:920, 140:1140]  # 去掉上下文字信息，避免干扰（src比例为纵：横 = 8：10）
    src = cv.resize(img, (800, 640))

    # 处理
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.GaussianBlur(src_gray, (3, 3), 1)

    # 寻找合适的阈值来获取适当的轮廓数量
    for threshold in range(50, 255, 1):
        # Detect edges using Canny
        canny_output = cv.Canny(src_gray, threshold, threshold * 2)
        # Find contours
        contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) == contours_number:
            break
    # print(threshold)      #显示当前阈值的取值

    # 取同样尺寸黑色背景
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    # 给轮廓随机取色
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

    # 计算每条轮廓的周长并在图像中显示
    for cnt in contours:
        perimeter = round(cv.arcLength(cnt, True), 1)
        if show==1:
            print(perimeter)
        cv.putText(drawing, str(perimeter), tuple(cnt[0][0] + [-10, -10]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # 显示原图与处理结果图
    # Show in a window
    if show == 1:
        Contour_window = 'Contours'
        cv.namedWindow(Contour_window)
        cv.imshow(Contour_window, drawing)
        cv.imshow('original', src)
        cv.waitKey()
    else:
        return drawing


def batch_contour(folder):
    """批量处理文件夹中的图像，并将结果保存在当前目录下的contours文件夹中。"""
    try:
        os.mkdir(folder + '/contours')
        for file in os.listdir(folder):
            if file.endswith('.jpg'):
                result = find_thresh(folder + '/' + file, 3, 0)
                cv.imwrite(folder + '/contours/con_' + file, result)
    except:
        pass


if __name__ == '__main__':

    start = time.time()
    #### 批处理
    #### if running in Windows OS, replace '/' with '\\' in path
    path = '/Users/duoguangxu/Documents/drop_pic/13_D_L/13_D_1110_H_L/13_D_1110_H_L_3'
    batch_contour(path)

    # 单张图像测试
    # pic = 'regular_3.jpg'
    # find_thresh(pic, 2, 1)

    end = time.time()
    print(str(end-start))