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
from step3 import shadowPlane
from step5_2 import getShadowPoints
from step5 import getShadowPoints_2
from step6 import get3DPoints

'''
    Global configuration  and variables shared in steps
'''

DISTANCE_BETWEEN_POINTS = 1000

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
        if math.dist(current_point, p) >= 2:
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

    if 'step3' in config:
        if not isinstance(config['step3']['Object Height'], numbers.Number):
            sys.exit('Object Height in step3 must be a number')
        if not os.path.isfile(config['step3']['Known Object Image']):
            sys.exit("Known Object Image in step3 doesn't exist")
    else:
        sys.exit(
            'Missing properties required for step3. Validate your config file')


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
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
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
    step3Config = config['step3']
    pathIntrinsic = args.intrinsic

    if args.intrinsic is None:
        camera_calibration(step1Config, args.steps)
        pathIntrinsic = step1Config['Path Save Intrisic Params']

    dist, mtx, projection_matrix = camera_position(
        step2Config, pathIntrinsic, step1Config['Chessboard Pattern Size'], step1Config['Chessboard Field Size'], args.steps)

    step3Image = cv2.imread(step3Config['Known Object Image'])
    step3Image = undistort(step3Image, mtx, dist)
    sPlane = shadowPlane(
        step3Config, step3Image, projection_matrix, mtx, dist, args.steps)

    print(sPlane)

    image = cv2.imread(config['Image'])
    image = undistort(image, mtx, dist)
    shadowPoints = []

    if args.version2 is None:
        shadowPoints = getShadowPoints(image, args.steps)
    else:
        image_no_shadow = cv2.imread(args.version2)
        shadowPoints = getShadowPoints_2(
            image_no_shadow, image, args.steps)

    objectPoints = get3DPoints(
        shadowPoints, projection_matrix, sPlane)

    points = []
    current_point = objectPoints[0]

    for p in objectPoints:
        if math.dist(current_point, p) >= 5:
            points.append(current_point)
            current_point = p

    points = reducePoints(objectPoints)
    # print(points)
    fig = plt.figure()
    ax = plt.axes()

    for curr, nxt in zip(points, points[1:]):
        distance = math.sqrt(
            ((curr[0]-nxt[0])**2)+((curr[1]-nxt[1])**2)+((curr[2]-nxt[2])**2))
        if distance < DISTANCE_BETWEEN_POINTS:
            yline = np.linspace(curr[1], nxt[1], num=2)
            zline = np.linspace(-curr[2], -nxt[2], num=2)

            ax.plot(yline, zline, 'gray')
        else:
            ax.scatter(curr[1], -curr[2], c=['gray'])
            ax.scatter(nxt[1], -nxt[2], c=['gray'])

    #plt.ylim([-100, 200])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    plt.show()


if __name__ == "__main__":
    main()
