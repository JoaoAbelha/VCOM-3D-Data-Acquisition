import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

'''
Shadow detection step : Try to detect the shadow line on the image and get one line segment representing the shadow projection into the object
'''


'''
* param {frame_0} : initial frame without shadow
* param {frame} : frame with shadow 
* return mask of the difference between images representing the shadow
'''


def processImage(frame_0, frame):

    """absdiff = cv2.absdiff(frame_0, frame)
    ret, absdiff_thresh = cv2.threshold(absdiff,40,255,cv2.THRESH_BINARY)
    result = cv2.morphologyEx(absdiff_thresh, cv2.MORPH_OPEN, (11, 11))
    result = cv2.morphologyEx(absdiff_thresh, cv2.MORPH_CLOSE, (11, 11))
    """
    absdiff = cv2.absdiff(frame_0,frame)
    ret, absdiff_thresh = cv2.threshold(absdiff,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret,thresh_0 = cv2.threshold(frame_0,150,255,cv2.THRESH_BINARY_INV)
    ret,thresh_n = cv2.threshold(frame,150,255,cv2.THRESH_BINARY_INV)   

    thresh_absdiff = cv2.absdiff(thresh_0,thresh_n)

    result = cv2.bitwise_and(absdiff_thresh , thresh_absdiff)
    result = cv2.morphologyEx(result, cv2.MORPH_ERODE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_ERODE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, (5, 5))

    return result


'''
* param {img} : binary shadow image
* param {direction} : get upper ou lower part of shadow
* return points selected from shadow
'''


def getPointsVertical(img, direction):
    height, width = img.shape[:2]

    y_top = 5
    y_bot = height - 5

    points = []
    x_top = -1
    x_bot = -1

    if x_top == 0 or x_bot == 0:
        return points

    for y in range(y_top, y_bot):
        interval = range(
            width-1, -1, -1) if direction == "left" else range(0, width)
        for x in interval:
            if img[y][x] == 255:
                points.append((x, y))
                break
    return points


'''
* param {img} : binary shadow image
* param {direction} : get left or right part of shadow
* return points selected from shadow
'''


def getPointsHorizontal(img, direction):

    height, width = img.shape[:2]

    x_left = 5
    x_right = width - 5

    points = []
    y_left = -1
    y_right = -1

    for x in range(x_left, x_right):
        interval = range(
            height-1, -1, -1) if direction == "up" else range(0, height)
        for y in interval:
            if img[y][x] == 255:
                points.append((x, y))
                break
    return points


'''
* param {img} : binary shadow image
* param {orientation} : which orientaion is shadow, horizontal or vertical
* param {direction} : which part of shadow
* return points selected from shadow
'''


def getPoints(points, orientation="vertical", direction="left"):
    if(orientation == "horizontal"):
        return getPointsHorizontal(points, direction)
    return getPointsVertical(points, direction)


'''
* param {poins} : shadow points
return binary image representing shadow points
'''


def drawPoints(points):
    line = np.zeros((height, width, 3), np.uint8)
    for p in points:
        line[p[1]][p[0]] = (0, 0, 255)
    return line


'''
* param {frame_0} : initial frame without shadow
* param {frame} : frame with shadow 
* return lower points of an horizontal shadow 
'''


def getShadowPoints_2(frame_0, frame_n, showSteps):

    frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
    frame_n = cv2.cvtColor(frame_n, cv2.COLOR_BGR2GRAY)

    height, width = frame_0.shape[:2]

    processed = processImage(frame_0, frame_n)
    points = getPoints(processed, "horizontal", "down")

    if showSteps:
        cv2.imshow('img', processed)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if showSteps:
        line_img = np.zeros((height, width, 3), np.uint8)
        for p in points:
            line_img[p[1]][p[0]] = 255
        cv2.imshow('img', line_img)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return points
