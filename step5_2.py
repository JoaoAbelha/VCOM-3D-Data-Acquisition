import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

SHOW_IMAGES = True

def getShadowPoints(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 7
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(kernel_size,kernel_size))
    cl1 = clahe.apply(frame_gray)

    kernel = 9
    blur = cv2.GaussianBlur(cl1,(kernel,kernel),0)

    lookUpTable = np.empty((1,256), np.uint8)
    gamma = 2
    for i in range(256):
        lookUpTable[0,i] = np.clip( 3*i - 60 , 0 , 255)
    lut_img = cv2.LUT(blur, lookUpTable)

    ret,thres = cv2.threshold(lut_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    imgWithSobelX = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=5)
    imgWithSobelY = cv2.Sobel(blur, cv2.CV_8U, 0, 1, ksize=5)

    abs_grad_x = cv2.convertScaleAbs(imgWithSobelX)
    abs_grad_y = cv2.convertScaleAbs(imgWithSobelY)
    grad = cv2.addWeighted(abs_grad_x, 0.1, abs_grad_y,0.9, 0)

    kernel = 7
    blur_new = cv2.GaussianBlur(grad,(kernel,kernel),0)

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    ellipse = cv2.morphologyEx(blur_new, cv2.MORPH_OPEN, kernel)

    thres = ~thres
    thres2 = cv2.dilate(thres,(3,1),iterations = 10)

    process = cv2.bitwise_and(ellipse,ellipse,mask=thres2)
    _,process = cv2.threshold(process,127,255,cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(process,cv2.MORPH_OPEN,(25,25))

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_contours = []

    current_cnt = None
    dist = 100000
    height,width = frame.shape[:2]
    pos = (0,height/2)

    for cnt in contours :
        for point in cnt:
            d = math.dist(point[0], pos)
            if d < dist:
                dist = d
                current_cnt = cnt
    new_contours.append(current_cnt)

    while current_cnt is not None:
        pos = tuple(current_cnt[current_cnt[:,:,0].argmax()][0])
        new_cnt = None
        dist = 100000
        for cnt in contours :
            for point in cnt:
                d = math.dist(point[0], pos)
                if point[0][0] <= pos[0]:
                    continue
                if d < dist:
                    dist = d
                    new_cnt = cnt
        if new_cnt is not None:
            new_contours.append(new_cnt)
        current_cnt = new_cnt

    selected_contours = np.zeros((height,width,3), np.uint8)
    cv2.drawContours(selected_contours, new_contours, -1, (0,255,0), 1)

    if SHOW_IMAGES:
        cv2.imshow('img', selected_contours)
        cv2.waitKey(1500)
    if SHOW_IMAGES:
        cv2.destroyAllWindows()

    points = []
    for x in range(width):
        for y in range(height):
            if (selected_contours[y][x] == [0,255,0]).all():
                points.append((x,y))
                break
    return points
    