import cv2 as cv
from step2 import camera_position
import numpy as np

def calculate3DPoints(i , j, projectionMatrix, shadowPlane = []):
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
    
    
img = cv.imread('./calibration/GOPR0032.jpg')
projectionMatrix = camera_position(img)[1]
calculate3DPoints(315, 5, projectionMatrix)