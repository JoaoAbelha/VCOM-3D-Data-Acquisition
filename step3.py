from step2 import camera_position, readIntrinsicParameters
from step1 import camera_calibration
from step5_2 import getShadowPoints
from step6 import calculate3DPoint
import cv2 as cv
import numpy as np
import random as rng

BASE_PLANE = [1, 0, 0, 0]


'''
* param {p1}: 3D coordinates for p1
* param {p2}: 3D coordinates for p2
* param {p3}: 3D coordinates for p3
* returns the plane 
'''


def calculate_plane(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return [a, b, c, d]

# returns the A, B, C, and D components of a plane that's perpendicular to x = 0 and contains the points passed as arguments


'''
* param {step3Config}: step3 params from config file
* param {image}: image with known object
* param {projection_matrix}: projection matrix obtained in the previous steps
* param {steps}: show intermediate steps
* returns the shadow plane 
'''


def shadowPlane(step3Config, image, projection_matrix, steps):
    shadowPoints = []
    shadowPoints = getShadowPoints(image, steps)

    point1 = max(shadowPoints, key=lambda x: x[0])
    point2 = min(shadowPoints, key=lambda x: x[0])
    point3 = min(shadowPoints, key=lambda x: x[1])

    if steps:
        #img = np.zeros(image)
        cv.circle(image, point1,
                  radius=5, color=(0, 255, 0), thickness=1)
        cv.circle(image, point2,
                  radius=5, color=(255, 0, 0), thickness=1)
        cv.circle(image, point3,
                  radius=5, color=(0, 0, 255), thickness=1)
        cv.imshow('Points to calulate shadow plane', image)
        cv.waitKey(5000)
        cv.destroyAllWindows()

    point13D = calculate3DPoint(
        point1[0], point1[1], projection_matrix, BASE_PLANE)
    point23D = calculate3DPoint(
        point2[0], point2[1], projection_matrix, BASE_PLANE)
    point33D = calculate3DPoint(point3[0], point3[1], projection_matrix, [
                                1, 0, 0, step3Config['Object Height']])

    #print(point13D, point23D, point33D)

    return calculate_plane(np.asarray(point13D), np.asarray(point23D), np.asarray(point33D))
