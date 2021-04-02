import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math

cap = cv2.VideoCapture('./imgs/video_rotated.mp4')

frame_0 = None
frame_n_1 = None

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

width = 360
height = 200

def toGray(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def processImage(frame_n):
    global frame_0,frame_n_1

    frame_n = cv2.GaussianBlur(frame_n,(7,7),1)
    frame_n = cv2.resize(frame_n,(width,height))
    
    original = frame_n

    if frame_0 is None:
        frame_0 = frame_n
        frame_n_1 = frame_0
    
    absdiff = cv2.absdiff(frame_0,frame_n)
    ret, absdiff_thresh = cv2.threshold(absdiff,40,255,cv2.THRESH_BINARY)

    ret,thresh_0 = cv2.threshold(frame_0,150,255,cv2.THRESH_BINARY_INV)
    ret,thresh_n = cv2.threshold(frame_n,150,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   

    thresh_absdiff = cv2.absdiff(thresh_0,thresh_n)
    shadow = cv2.bitwise_and(absdiff_thresh , thresh_absdiff)
    
    #shadow = cv2.morphologyEx(shadow, cv2.MORPH_ERODE, (11,11),2)
    #frame_0 = frame_n

    points = get_points(absdiff_thresh)
    line = np.zeros((height,width,3), np.uint8)
    for p in points:
        line[p[1]][p[0]] = (0,0,255)

    ret1 = np.concatenate((toGray(original), toGray(shadow)), axis=1)
    ret2 = np.concatenate((toGray(absdiff), toGray(absdiff_thresh)), axis=1)
    ret3 = np.concatenate((toGray(thresh_0), toGray(thresh_n)), axis=1)
    ret4 = np.concatenate((toGray(thresh_absdiff), line), axis=1)
    
    
    ret = np.concatenate((ret1, ret2, ret3, ret4), axis=0)
    return shadow , ret


y_top = 5
y_bot = height - 5

def get_points(img):
    points = []
    x_top = -1
    x_bot = -1

    if img[y_top][width-1] == 255:
        return points
    if img[y_bot][width-1] == 255:
        return points
    for x_top in range(width-2,-1,-1):
        if img[y_top][x_top] == 255:
            break
    for x_bot in range(width-2,-1,-1):
        if img[y_bot][x_bot] == 255:
            break
        
    if x_top == 0 or x_bot == 0:
        return points
    print( "top : " , x_top , y_top)
    print( "bot : " , x_bot , y_bot)

    for y in range(y_top,y_bot):
        for x in range(width-1,-1,-1):
            if img[y][x] == 255:
                points.append( (x,y) )
                break
    return points


while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img , imgs = processImage(gray)
    cv2.imshow('frame',imgs)
    #points = get_points(img)
    #print(points)

    key = cv2.waitKey(0)
    while key not in [ord('q'), ord('k')]:
        key = cv2.waitKey(0)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()