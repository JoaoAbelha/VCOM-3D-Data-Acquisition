import argparse
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import sys
import json
import numbers
from json import JSONDecodeError

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

IMG_NO_SHADOW = './imgs/alternate4/planes.png'
PLANES_IMG = './imgs/alternate4/i (1).png'

DISTANCE_BETWEEN_POINTS = 50

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

'''
* param {config}: config file data
* param {intrinsic}: boolean representing if an intrinsic param file was given
* validates if the configuration file values are correct
* returns void
'''
def validateConfig(config, intrinsic):
    if not os.path.isfile(config['Image']):
        sys.exit("Image in config file doesn't exist")

    if 'step2' in config:
        if not isinstance(config['step2']['Axis Size'], numbers.Number):
            sys.exit('Axis Size in step2 must be a number')
        if not os.path.isfile(config['step2']['Chessboard Image']):
            sys.exit("Chessboard Image in step2 doesn't exist")
    else:
        sys.exit('Missing properties required for step2. Validate your config file')

    if 'step1' in config:
        if not intrinsic:
            if config['step1']['Calibration Images Folder'] is None:
                sys.exit('Axis Size in step1 is required')
            if not isinstance(config['step1']['Chessboard Field Size'], numbers.Number):
                sys.exit('Chessboard Field Size in step1 must be a number')
            if config['step1']['Path Save Intrisic Params'] is None:
                sys.exit("Path Save Intrisic Params in step1 is required")
        if len(config['step1']['Chessboard Pattern Size']) != 2:
            sys.exit('Chessboard Pattern Size in step1 must have length 2')
        else:
            config['step1']['Chessboard Pattern Size'] = tuple(
                config['step1']['Chessboard Pattern Size'])
    else:
        sys.exit(
            'Missing properties required for step1. Validate your config file')


'''
* param {config}: config file path
* param {intrinsic}: boolean representing if an intrinsic param file was given
* reads the config file and then validates
* returns the data from the config file
'''
def parseConfig(config, intrinsic):
    if os.path.isfile(config):
        with open(config) as config_file:
            try:
                data = json.load(config_file)
                validateConfig(data, intrinsic)
                return data
            except JSONDecodeError:
                sys.exit(
                    'Error parsing the config file. Please validate your config file')
    else:
        sys.exit("Config file doesn't exist")

'''
* param {args}: command line args
* validates the command line args
* returns the data from the config file
'''
def validateArgs(args):
    if args.version2 is not None:
        if not os.path.isfile(args.version2):
            sys.exit("Image for version 2 doesn't exist")

    if args.intrinsic is not None:
        if not os.path.isfile(args.intrinsic):
            sys.exit("Intrinsic params file doesn't exist")
        return parseConfig(args.Config, True)

    return parseConfig(args.Config, False)

'''
* param {img}: image to undistort
* param {mtx}: camera matrix
* param {dist}: distortion parameters
* returns the image undistorted
'''
def undistort(img, mtx, dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description='3D Data Acquisition using a Structured Light Technique')

    parser.add_argument(
        '-s', '--steps', help='Show the results of the intermediate steps', action='store_true')
    parser.add_argument(
        '-v2', '--version2', help='Path to image without shadow used in version 2', action='store', type=str)
    parser.add_argument(
        '-i', '--intrinsic', help='Path to intrinsic params file', action='store', type=str)
    parser.add_argument('Config', metavar='config', type=str,
                        help='Configuration file with the necessary input to run the application')
    args = parser.parse_args()

    config = validateArgs(args)
    step1Config = config['step1']
    step2Config = config['step2']
    pathIntrinsic = args.intrinsic

    if args.intrinsic is None:
        camera_calibration(step1Config, args.steps)
        pathIntrinsic = step1Config['Path Save Intrisic Params']

    dist, mtx, rotM, real_word_position, projection_matrix = camera_position(
        step2Config, pathIntrinsic, step1Config['Chessboard Pattern Size'], args.steps)

    image = cv2.imread(config['Image'])
    print('here')
    image = undistort(image, mtx, dist)
    shadowPoints = []

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
        shadowPoints = getShadowPoints(shadow_image_foreground, args.steps)

        objectPoints = shadow3DPoints(shadowPoints, projection_matrix, shadow_plane)
    else:
        shadowPoints = None
        if args.version2 is None:
            shadowPoints = getShadowPoints(image, args.steps)
        else:
            image_no_shadow = cv2.imread(args.version2)
            shadowPoints = getShadowPoints_2(image_no_shadow,image, args.steps)

        point1 = max(shadowPoints,key=lambda x: x[0])
        point2 = min(shadowPoints,key=lambda x: x[0])
        point3 = min(shadowPoints,key=lambda x: x[1])

        print((point1,point2,point3))

        #basePoints = shadow3DPoints([point1,point2], projection_matrix, [1,0,0,0])
        uv1 = np.array([[point1[0],point1[1],1]], dtype=np.float32).T
        uv2 = np.array([[point2[0],point2[1],1]], dtype=np.float32).T
        uv3 = np.array([[point3[0],point3[1],1]], dtype=np.float32).T
        p1 = project_image_point_to_plane(uv1, [1,0,0,0], mtx, rotM, real_word_position)
        p2 = project_image_point_to_plane(uv2, [1,0,0,0], mtx, rotM, real_word_position) 
        p3 = project_image_point_to_plane(uv3, [1,0,0,10], mtx, rotM, real_word_position) 
        flat1 = np.squeeze(p1)
        flat2 = np.squeeze(p2)
        flat3 = np.squeeze(p3)

        print((flat1,flat2))
        print(flat3)

        plane = calculate_plane(flat1,flat2,flat3)
        print(plane)

        objectPoints = shadow3DPoints(shadowPoints, projection_matrix, [1, 0, 0, 10])

    #print(objectPoints)
    points = []

    current_point = objectPoints[0]

    for p in objectPoints:
        if math.dist(current_point, p) >= 5:
            points.append(current_point)
            current_point = p

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
