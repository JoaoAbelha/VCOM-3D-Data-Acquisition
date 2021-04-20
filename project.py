import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from step1 import camera_calibration
from step2 import camera_position
from step3 import light_calibration, calibrate_planes, calculate_plane, project_image_point_to_plane
from step5_2 import getShadowPoints
from step5 import getShadowPoints_2
from step6 import shadow3DPoints

'''
    Global configuration  and variables shared in steps
'''
MULTIPLE_PLANES = False

CAMERA_POSITION_IMG = './imgs/objs2/IMG_20210414_122106.jpg'
IMG = './imgs/objs2/IMG_20210414_123048.jpg'
IMG_NO_SHADOW = './imgs/objs2/IMG_20210414_122921.jpg'
DISTANCE_BETWEEN_POINTS = 50
PLANES_IMG = None

if MULTIPLE_PLANES:
    CAMERA_POSITION_IMG = './imgs/alternate4/checkerboard.png'
    IMG = './imgs/alternate4/i (12).png'
    IMG_NO_SHADOW = './imgs/alternate4/planes.png'
    PLANES_IMG = './imgs/alternate4/i (1).png'


# PATTERN_SIZE: the pattern the algorithm is going to look for in the chessboard
PATTERN_SIZE = (9, 6)
#  PATH_SAVE_INTRINSIC_PARAMS: where the intrinsic parameters are going to be saved
PATH_SAVE_INTRINSIC_PARAMS = "./calibration/wide_dist_pickle.p"
PATH_SAVE_INTRINSIC_PARAMS = "./calibration/alternate/wide_dist_pickle.p"
#  SAVE_PARAMETERS: if true, the parameters are saved in a file
SAVE_PARAMETERS = False

'''
    * param {objectPoints}: the points that were found through shadow segmentation
    * post processing: deletes points that are too closed since visually it does not make any difference but
      computationally it eases the display of points
    returns a new set of points
'''


def reducePoints(objectPoints):
    points = []
    current_point = objectPoints[0]

    for p in objectPoints:
        if math.dist(current_point, p) >= 5:
            points.append(current_point)
            current_point = p
    return points


def main():
    parser = argparse.ArgumentParser(
        description='3D Data Acquisition using a Structured Light Technique')
    args = parser.parse_args()

    if SAVE_PARAMETERS:
        camera_calibration(PATTERN_SIZE, PATH_SAVE_INTRINSIC_PARAMS)
    #print("Hello World!")
    # camera_calibration()
    chessBoardImage = cv2.imread(CAMERA_POSITION_IMG)
    _, mtx, rotM, real_word_position, projection_matrix = camera_position(chessBoardImage, PATH_SAVE_INTRINSIC_PARAMS, PATTERN_SIZE)
    image = cv2.imread(IMG)
    image_no_shadow = cv2.imread(IMG_NO_SHADOW)

    objectPoints = None
    if MULTIPLE_PLANES:
        planes_img = cv2.imread(PLANES_IMG)
        bg_mask, fg_mask, planes = calibrate_planes(mtx, rotM, real_word_position, planes_img, control_image=image_no_shadow)

        shadow_image = cv2.absdiff(image_no_shadow, image)
        
        shadow_image_background = cv2.bitwise_not(cv2.bitwise_and(shadow_image, shadow_image, mask=bg_mask))
        shadow_plane = light_calibration(shadow_image_background, mtx, rotM, real_word_position, planes, mask = bg_mask)
        
        shadow_image_foreground = cv2.bitwise_not(cv2.bitwise_and(shadow_image, shadow_image, mask=fg_mask))
        cv2.imshow("sadfore", shadow_image_foreground)
        shadowPoints = getShadowPoints(shadow_image_foreground)

        objectPoints = shadow3DPoints(shadowPoints, projection_matrix, shadow_plane)
    else:
        shadowPoints = getShadowPoints(image)

        point1 = max(shadowPoints,key=lambda x: x[0])
        point2 = min(shadowPoints,key=lambda x: x[0])
        point3 = min(shadowPoints,key=lambda x: x[1])

        print((point1,point2,point3))

        basePoints = shadow3DPoints([point1,point2], projection_matrix, [1,0,0,0])
        uv1 = np.array([[point1[0],point1[1],1]], dtype=np.float32).T
        uv2 = np.array([[point2[0],point2[1],1]], dtype=np.float32).T
        uv3 = np.array([[point3[0],point3[1],1]], dtype=np.float32).T
        p1 = project_image_point_to_plane(uv1, [1,0,0,0], mtx, rotM, real_word_position)
        p2 = project_image_point_to_plane(uv2, [1,0,0,0], mtx, rotM, real_word_position) 
        p3 = project_image_point_to_plane(uv3, [1,0,0,10], mtx, rotM, real_word_position) 
        flat1 = np.squeeze(p1)
        flat2 = np.squeeze(p2)
        flat3 = np.squeeze(p3)

        objectPoints = shadow3DPoints([point3], projection_matrix, [1,0,0,10])

        print((flat1,flat2))
        print(flat3)

        plane = calculate_plane(flat1,flat2,flat3)
        print(plane)


    #print(objectPoints)
    points = []

    current_point = objectPoints[0]

    for p in objectPoints:
        if math.dist(current_point, p) >= 5:
            points.append(current_point)
            current_point = p

    print(len(objectPoints))
    print(len(points))
    # print(points)
    t = []
    for p in points:
        t.append(p[2])

    n, bins, patches = plt.hist(x=t, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.show()

    ##
    final_list = np.array([])
    binlist = np.c_[bins[:-1], bins[1:]]
    d = np.array(t)
    for i in range(len(binlist)):
        if i == len(binlist)-1:
            l = d[(d >= binlist[i, 0]) & (d <= binlist[i, 1])]
        else:
            l = d[(d >= binlist[i, 0]) & (d < binlist[i, 1])]
        if l.shape[0] < 10:
            final_list = np.append(final_list, l)
    ##

    fig = plt.figure()
    ax = plt.axes()
    print(len(final_list))

    for curr, nxt in zip(points, points[1:]):
        distance = math.sqrt(
            ((curr[0]-nxt[0])**2)+((curr[1]-nxt[1])**2)+((curr[2]-nxt[2])**2))
        if distance < DISTANCE_BETWEEN_POINTS:
            yline = np.linspace(curr[0], nxt[0], num=2)
            zline = np.linspace(-curr[2], -nxt[2], num=2)

            ax.plot(yline, zline, 'gray')
        else:
            ax.scatter(curr[0], -curr[2], c=['gray'])
            ax.scatter(nxt[0], -nxt[2], c=['gray'])

    plt.ylim([-200, 0])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    plt.show()


if __name__ == "__main__":
    main()
