import cv2
import numpy as np
from step2 import camera_position
from utils import gaussMethod

DEFAUL_SHADOW_PLANE = np.array([0, 1, 0, 0])


def calculate3DPoint(i, j, projectionMatrix, shadowPlane=DEFAUL_SHADOW_PLANE):
    firstEquation = np.array([
        projectionMatrix[2][0] * i - projectionMatrix[0][0],
        projectionMatrix[2][1] * i - projectionMatrix[0][1],
        projectionMatrix[2][2] * i - projectionMatrix[0][2],
        projectionMatrix[0][3] - projectionMatrix[2][3] * i
    ])

    secondEquation = np.array([
        projectionMatrix[2][0] * j - projectionMatrix[1][0],
        projectionMatrix[2][1] * j - projectionMatrix[1][1],
        projectionMatrix[2][2] * j - projectionMatrix[1][2],
        projectionMatrix[1][3] - projectionMatrix[2][3] * j
    ])

    equations = np.array([firstEquation, secondEquation, shadowPlane])
    # solve the system of equations
    gaussResult = gaussMethod(equations)
    # return the x,y and z values obtained by solving the system of equations
    return [gaussResult[0][3], gaussResult[1][3], gaussResult[2][3]]


def shadow3DPoints(points, projectionMatrix, shadowPlane=DEFAUL_SHADOW_PLANE):
    shadowPoints3D = []
    for p in points:
        point3D = calculate3DPoint(p[0], p[1], projectionMatrix)
        #shadowPoints3D = np.append(shadowPoints3D, point3D)
        shadowPoints3D.append(point3D)
    return shadowPoints3D
