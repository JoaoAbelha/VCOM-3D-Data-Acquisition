import cv2
import numpy as np
from step2 import camera_position
from utils import gaussMethod


def calculate3DPoints(i, j, ip, projectionMatrix, shadowPlane=[]):
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
        shadowPlane[3][1] * ip - shadowPlane[1][1],
        shadowPlane[3][2] * ip - shadowPlane[1][2],
        shadowPlane[3][3] * ip - shadowPlane[1][3],
        shadowPlane[1][4] - shadowPlane[3][4] * ip
    ])

    equations = np.matrix([firstEquation, secondEquation, thirdEquation])
    # solve the system of equations
    gaussResult = gaussMethod(equations)
    # return the x,y and z values obtained by solving the system of equations
    return np.array([gaussResult[0][3], gaussResult[1][3], gaussResult[2][3]])


img = cv2.imread('./calibration/GOPR0032.jpg')
projectionMatrix = camera_position(img)[1]
calculate3DPoints(315, 5, projectionMatrix)
