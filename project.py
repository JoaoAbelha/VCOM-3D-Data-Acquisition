import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from step1 import camera_calibration
from step2 import camera_position
from step5_2 import getShadowPoints
from step5 import getShadowPoints_2
from step6 import shadow3DPoints

'''
    Global configuration  and variables shared in steps
'''
CAMERA_POSITION_IMG = './imgs/objs2/IMG_20210414_122106.jpg'
IMG = './imgs/objs2/IMG_20210414_123048.jpg'
IMG_NO_SHADOW = './imgs/objs2/IMG_20210414_122921.jpg'
DISTANCE_BETWEEN_POINTS = 50


# PATTERN_SIZE: the pattern the algorithm is going to look for in the chessboard
PATTERN_SIZE = (9, 6)
#  PATH_SAVE_INTRINSIC_PARAMS: where the intrinsic parameters are going to be saved
PATH_SAVE_INTRINSIC_PARAMS = "calibration/wide_dist_pickle.p"
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
    position_normalized, projection_matrix = camera_position(
        chessBoardImage, PATH_SAVE_INTRINSIC_PARAMS, PATTERN_SIZE)
    image = cv2.imread(IMG)
    image_no_shadow = cv2.imread(IMG_NO_SHADOW)
    shadowPoints = getShadowPoints(image)
    #shadowPoints = getShadowPoints_2(image_no_shadow,image)
    objectPoints = shadow3DPoints(shadowPoints, projection_matrix)
    # print(objectPoints)
    points = reducePoints(objectPoints)

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
