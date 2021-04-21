import pickle
import cv2 as cv
import numpy as np
from utils import getProjectionMatrix

'''
    Second step: Find the extrinsic parameters, in other words, 
    the camera pose

'''


'''
 * reads from the file, the intrinsic parameters of the camera
 * params {pathIntrinsic}: the path where the intrinsic parameters were saved
 * returns a tuple with those values
'''


def readIntrinsicParameters(pathIntrinsic):
    objects = []
    with (open(pathIntrinsic, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    if len(objects) != 1:
        print("It only should have a object. Invalid file. It has {} objects".format(
            len(objects)))
        exit(1)

    intrinsic = objects[0]

    if "mtx" not in intrinsic or "dist" not in intrinsic:
        print("Invalid object read from file")
        exit(1)

    return (intrinsic["mtx"], intrinsic["dist"])


'''
* param {objpoints} : relative object points
* param {imgpoints}:  image points
* param {mtx}: camera matrix
* param {dist}: distortion parameters
* param {rvecs}: rotation vectors
* param {tvecs}: translation vectors
* prints and calculates the reprojection error
* return: void
'''


def reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Reprojection error= {}".format(mean_error / len(objpoints)))


'''
* param {img}: the image where the camera pose was calculated
* param {corners}: the corners of the chessboard
* param {imgpts}: the image points
* it draws in the img the axis that is centered in the first chess corner
* it returns the img with the axis drawn on it
'''


def draw(img, corners, imgpts):
    """Draw Axes"""
    corner = tuple(corners[0].ravel())
    cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


'''
 * params {step2Config}: values extracted from the config file needed for this step
 * params {pathIntrinsic}: the path where the intrinsic parameters were saved
 * params {patternSize}: the pattern the algorithm is going to look for in the chessboard
 * it calculates the camera position and the projection matrix
 * returns the camera position and the projection matrix
'''


def camera_position(step2Config, pathIntrinsic, patternSize, showSteps):
    img = cv.imread(step2Config['Chessboard Image'])
    (mtx, dist) = readIntrinsicParameters(pathIntrinsic)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, patternSize, None)

    # 3d points
    objp = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patternSize[0],
                           0:patternSize[1]].T.reshape(-1, 2) * 22

    objPoints = []
    imgPoints = []
    # solvePnP requires camera calibraiton
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    if ret == True:
        objPoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
        ret, rvec, tvec, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)
 
        imgPoints.append(corners) ## heere

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]
                          ).reshape(-1, 3) * step2Config['Axis Size']
        axis_img, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

        if showSteps:
            draw(img, corners, axis_img)
            cv.imshow('img', img)
            cv.waitKey()
            cv.destroyAllWindows()

        rotM = cv.Rodrigues(rvec)[0]
        real_word_position = -np.matrix(rotM).T * np.matrix(tvec)
        #position_normalized = np.linalg.inv(mtx) * real_word_position

        print('real world position (X,Y,Z)= ({}, {}, {})'.format(
            real_word_position[0, 0], real_word_position[1, 0], real_word_position[2, 0]))
        # we can normalize the point: focal length = 1 and moves the origin to the centre of the image
        print('projection matrix : ')
        print(getProjectionMatrix(mtx, rvec, tvec))

        reprojection_error(objPoints, imgPoints, mtx, dist, [rvec], [tvec])
        return (dist, mtx, rotM, real_word_position, getProjectionMatrix(mtx, rvec, tvec))
    else:
        print('could not find position')
