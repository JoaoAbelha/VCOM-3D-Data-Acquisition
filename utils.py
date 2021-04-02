import cv2 as cv
import numpy as np

def getProjectionMatrix(mtx, rvec, tvec):
    rotM = cv.Rodrigues(rvec)[0]
    extrinsicMatrix = np.column_stack((rotM, tvec))
    # multiply the extrinsic with the intrinsic
    projectionMatrix = np.matmul (mtx , extrinsicMatrix)
    return projectionMatrix


# apply gauss method to calculate a system of linear equations
def rowOperation(m, a, b ,k):
    result = np.array(m, dtype = float)
    for i in range(len(m[0])):
        result[a][i] -= result[b][i] * k
    return result

# solve gauss
def gaussMethod(m):
    for i in range(len(m)):
        m = rowOperation(m, i, i, 1 - 1.0/m[i][i])
        for j in range(len(m)):
            if i != j:
                m = rowOperation(m, j, i, m[j][i])
    # check column to see intersection
    print(m)


gaussMethod([
    [9 , 8   , 1  , 2  , 120],
    [ 4  , 100 , 10 , 2  , 119],
    [6  , 8   , 55 , 4  , 70],
    [10 , 2   , 45  , 80 , 96]
])
