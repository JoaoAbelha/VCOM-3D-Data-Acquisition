import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

width = 360
height = 200

def processImage(frame_0,frame):
    frame_0 = cv2.resize(frame_0,(width,height))
    frame = cv2.resize(frame,(width,height))

    absdiff = cv2.absdiff(frame_0,frame)
    ret, absdiff_thresh = cv2.threshold(absdiff,40,255,cv2.THRESH_BINARY)
    result = cv2.morphologyEx(absdiff_thresh, cv2.MORPH_CLOSE, (5,5))
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, (5,5))

    return result

y_top = 5
y_bot = height - 5

def getPointsVertical(img):
    points = []
    x_top = -1
    x_bot = -1

    if img[y_top][width-1] == 255 or img[y_bot][width-1] == 255:
        return points
    for x_top in range(width-2,-1,-1):
        if img[y_top][x_top] == 255:
            break
    for x_bot in range(width-2,-1,-1):
        if img[y_bot][x_bot] == 255:
            break
        
    if x_top == 0 or x_bot == 0:
        return points

    for y in range(y_top,y_bot):
        interval = range(width-1,-1,-1) if direction == "left" else range(0,width)
        for x in interval:
            if img[y][x] == 255:
                points.append( (x,y) )
                break
    return points

x_left = 5
x_right = width - 5

def getPointsHorizontal(img, direction):
    points = []
    y_left = -1
    y_right = -1

    if img[height-1][x_left] == 255 or img[height-1][x_right] == 255:
        return points
    for y_left in range(height-2,-1,-1):
        if img[y_left][x_left] == 255:
            break
    for y_right in range(height-2,-1,-1):
        if img[y_right][x_right] == 255:
            break
        
    if y_left == 0 or y_right == 0:
        return points

    for x in range(x_left,x_right):
        interval = range(height-1,-1,-1) if direction == "up" else range(0,height)
        for y in interval:
            if img[y][x] == 255:
                points.append( (x,y) )
                break
    return points

def getPoints(points, orientation="vertical", direction="left"):
    if(orientation == "horizontal"):
        return getPointsHorizontal(points, direction)
    return getPointsVertical(points, direction)

def drawPoints(points):
    line = np.zeros((height,width,3), np.uint8)
    for p in points:
        line[p[1]][p[0]] = (0,0,255)
    return line




frame_0 = cv2.imread('./imgs/charger_0.jpg',0)
frame = cv2.imread('./imgs/charger_1.jpg',0)

processed = processImage(frame_0,frame)

points1 = getPoints(processed,"horizontal","down")
points2 = getPoints(processed,"horizontal","up")

cv2.imshow('frame',np.concatenate((cv2.cvtColor(processed,cv2.COLOR_GRAY2BGR), drawPoints(points1),drawPoints(points2)), axis=0))

key = cv2.waitKey(0)
while key not in [ord('q'), ord('k')]:
    key = cv2.waitKey(0)