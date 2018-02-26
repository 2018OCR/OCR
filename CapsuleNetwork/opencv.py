# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_color = cv2.imread("pk107.jpg")#灰度图读入
image=cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
cv2.imshow("dd",image)
cv2.waitKey(0)
adaptive_threshold=cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#ret,adaptive_threshold = cv2.threshold(image,50,255,cv2.THRESH_BINARY_INV)#二值化
cv2.imshow('binary image', adaptive_threshold)
cv2.waitKey(0)

horinzontal_sum=np.sum(adaptive_threshold,axis=1)#行相加
plt.plot(horinzontal_sum,range(horinzontal_sum.shape[0]))
plt.gca().invert_yaxis()#y轴反转，使【0，0】坐标在左上角
plt.show()

#提取数组里面的峰值，然后找出文本行，定义提取峰值函数
def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges


peek_ranges=extract_peek_ranges_from_array(horinzontal_sum)

line_seg_adaptive_threshold = np.copy(adaptive_threshold)
for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    w = line_seg_adaptive_threshold.shape[1]
    h = peek_range[1] - y
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
cv2.imshow('line image', line_seg_adaptive_threshold)
cv2.waitKey(0)


def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges


vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = np.sum(line_img, axis=0)
    vertical_peek_ranges = extract_peek_ranges_from_array(
        vertical_sum,
        minimun_val=40,
        minimun_range=1)
    vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
    vertical_peek_ranges2d.append(vertical_peek_ranges)


color = (0, 0, 255)
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(image_color, pt1, pt2, color)
cv2.imshow('splited char image', image_color)
cv2.waitKey(0)


def cut(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    img =np.array(img)

    top = 0
    bottom = img.shape[0]
    left = img.shape[1]
    right = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] != 0:
                top = max(top, i)
                bottom = min(bottom, i)
                left = min(left, j)
                right = max(right, j)
    img = Image.fromarray(img)
    img=img.crop((left,bottom,right,top))
    img.show()
'''
    height = top - bottom
    width = right - left
    if height > width:
        region = img.resize((int(100.0 / height * width), 100), Image.ANTIALIAS)
    else:
        region = img.resize((100, int(100.0 / width * height)), Image.ANTIALIAS)
    
    region.show()
    ret_img = Image.new('RGB', (140, 140), 'white')
    ret_img.paste(region, (20, 20))
    ret_img.show()
'''
## cut

for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        pic=image_color[y:y+h,x:x+w]
        cut(pic)


