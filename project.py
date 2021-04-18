import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from step1 import camera_calibration
from step2 import camera_position
from step3 import light_calibration, calibrate_planes
from step5_2 import getShadowPoints
from step5 import getShadowPoints_2
from step6 import shadow3DPoints

MULTIPLE_PLANES = True

CAMERA_POSITION_IMG = './imgs/objs2/IMG_20210414_122106.jpg'
IMG = './imgs/objs2/IMG_20210414_123048.jpg'
IMG_NO_SHADOW = './imgs/objs2/IMG_20210414_122921.jpg'
PLANES_IMG = None

if MULTIPLE_PLANES:
    CAMERA_POSITION_IMG = './imgs/alternate4/checkerboard.png'
    IMG = './imgs/alternate4/i (12).png'
    IMG_NO_SHADOW = './imgs/alternate4/planes.png'
    PLANES_IMG = './imgs/alternate4/i (1).png'


def main():
    #print("Hello World!")
    #camera_calibration()
    chessBoardImage = cv2.imread(CAMERA_POSITION_IMG)
    _, mtx, rotM, real_word_position, projection_matrix = camera_position(chessBoardImage)
    image = cv2.imread(IMG)
    image_no_shadow = cv2.imread(IMG_NO_SHADOW)

    objectPoints = None
    if MULTIPLE_PLANES:
        planes_img = cv2.imread(PLANES_IMG)
        bg_mask, fg_mask, planes = calibrate_planes(mtx, rotM, real_word_position, planes_img, control_image=image_no_shadow)

        shadow_image = cv2.absdiff(image_no_shadow, image)
        
        shadow_image_background = cv2.bitwise_not(cv2.bitwise_and(shadow_image, shadow_image, mask=bg_mask))
        cv2.imshow("sadback", shadow_image_background)
        shadow_plane = light_calibration(shadow_image_background, mtx, rotM, real_word_position, planes, mask = bg_mask)
        
        shadow_image_foreground = cv2.bitwise_not(cv2.bitwise_and(shadow_image, shadow_image, mask=fg_mask))
        cv2.imshow("sadfore", shadow_image_foreground)
        shadowPoints = getShadowPoints(shadow_image_foreground)

        objectPoints = shadow3DPoints(shadowPoints, projection_matrix, shadow_plane)
    else:
        shadowPoints = getShadowPoints(image)
        objectPoints = shadow3DPoints(shadowPoints, projection_matrix)

    #print(objectPoints)
    points = []

    current_point = objectPoints[0]

    for p in objectPoints:
        if math.dist(current_point, p) >= 5:
            points.append(current_point)
            current_point = p

    print(len(objectPoints))
    print(len(points))
    #print(points)
    t= []
    for p in points:
        t.append(p[2])

    n, bins, patches = plt.hist(x=t, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.show()

    ##
    final_list = np.array([])
    binlist = np.c_[bins[:-1],bins[1:]]
    d = np.array(t)
    for i in range(len(binlist)):
        if i == len(binlist)-1:
            l = d[(d >= binlist[i,0]) & (d <= binlist[i,1])]
        else:
            l = d[(d >= binlist[i,0]) & (d < binlist[i,1])]
        if l.shape[0] < 10:
            final_list = np.append(final_list, l)
    ##

    fig = plt.figure()
    ax = fig.add_subplot()
    print(len(final_list))


    
    for p in points:
        if p[2] not in final_list or True:
            ax.scatter(p[0], -p[2])
    plt.ylim([-200, 0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    

    plt.show()
    


if __name__ == "__main__":
    main()
