import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from step1 import camera_calibration
from step2 import camera_position
from step5 import getPoints
from step6 import shadow3DPoints

CAMERA_POSITION_IMG = 'calibration/GOPR0032.jpg'


def main():
    print("Hello World!")
    camera_calibration()
    image = cv2.imread(CAMERA_POSITION_IMG)
    position_normalized, projection_matrix = camera_position(image)
    shadow_plane = []


if __name__ == "__main__":
    main()
