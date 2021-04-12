import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from step1 import camera_calibration
from step2 import camera_position
from step5_2 import getShadowPoints
from step6 import shadow3DPoints

CAMERA_POSITION_IMG = 'calibration/GOPR0032.jpg'
IMG = './imgs/alternate/Untitled_000037.png'


def main():
    #print("Hello World!")
    #camera_calibration()
    chessBoardImage = cv2.imread(CAMERA_POSITION_IMG)
    position_normalized, projection_matrix = camera_position(chessBoardImage)
    image = cv2.imread(IMG)
    shadowPoints = getShadowPoints(image)
    objectPoints = shadow3DPoints(shadowPoints, projection_matrix)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for p in objectPoints:
        ax.scatter(p[0], p[1], p[2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    main()
