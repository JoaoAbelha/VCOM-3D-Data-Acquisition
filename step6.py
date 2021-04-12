import cv2
import numpy as np
from step2 import camera_position
from utils import gaussMethod
from step5 import getPoints, processImage

DEFAUL_SHADHOW_PLANE = npm.matrix([0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0])


def calculate3DPoint(i, j, ip, projectionMatrix, shadowPlane=DEFAUL_SHADHOW_PLANE):
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

    thirdEquation = np.array([
        shadowPlane[2][0] * ip - shadowPlane[0][0],
        shadowPlane[2][1] * ip - shadowPlane[0][1],
        shadowPlane[2][2] * ip - shadowPlane[0][2],
        shadowPlane[0][3] - shadowPlane[2][3] * ip
    ])

    equations = np.matrix([firstEquation, secondEquation, thirdEquation])
    # solve the system of equations
    gaussResult = gaussMethod(equations)
    # return the x,y and z values obtained by solving the system of equations
    return np.array([[gaussResult[0][3], gaussResult[1][3], gaussResult[2][3]]])


def shadow3DPoints(points, projectionMatrix):
    points3D = np.array([])
    for p in points:
        point3D = calculate3DPoints(p[1], p[2], 1, projectionMatrix)
        points3D = np.append(points3D, point3D)
    return points3D


frame_0 = cv2.imread('./imgs/charger_0.jpg', 0)
processed = processImage(frame_0, frame)
points1 = getPoints(processed, "horizontal", "down")

projectionMatrix = camera_position(img)[1]
shadow3DPoints(points1, projectionMatrix)
