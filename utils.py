import cv2 as cv
import numpy as np

def getProjectionMatrix(mtx, rvec, tvec):
    rotM = cv.Rodrigues(rvec)[0]
    extrinsicMatrix = np.column_stack((rotM, tvec))
    # multiply the extrinsic with the intrinsic
    projectionMatrix = np.matmul (mtx , extrinsicMatrix)
    return projectionMatrix

