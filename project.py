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

# PATTERN_SIZE: the pattern the algorithm is going to look for in the chessboard
PATTERN_SIZE=(9,6)
#  PATH_SAVE_INTRINSIC_PARAMS: where the intrinsic parameters are going to be saved
PATH_SAVE_INTRINSIC_PARAMS="calibration/wide_dist_pickle.p"
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
    if SAVE_PARAMETERS:
        camera_calibration(PATTERN_SIZE, PATH_SAVE_INTRINSIC_PARAMS)
    
    chessBoardImage = cv2.imread(CAMERA_POSITION_IMG)
    position_normalized, projection_matrix = camera_position(chessBoardImage, PATH_SAVE_INTRINSIC_PARAMS, PATTERN_SIZE)
    image = cv2.imread(IMG)
    image_no_shadow = cv2.imread(IMG_NO_SHADOW)
    shadowPoints = getShadowPoints(image)
    #shadowPoints = getShadowPoints_2(image_no_shadow,image)
    objectPoints = shadow3DPoints(shadowPoints, projection_matrix)
    #print(objectPoints)
    points = reducePoints(objectPoints)

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
