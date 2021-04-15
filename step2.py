import pickle
import cv2 as cv
import numpy as np
from utils import getProjectionMatrix

'''
    Second step: Find the extrinsic parameters, in other words, 
    the camera pose

'''

'''
 Global configuration variables that are used in this second step
 * AXIS_SIZE: if the image is show, this represents the size of the axis drawn in the image
 * SHOW_AXIS_IMAGE: displays the image in which the camera pose was calculated
'''
AXIS_SIZE = 10
SHOW_AXIS_IMAGE = True

'''
 * reads from the file, the intrinsic parameters of the camera
 * params {PATH_SAVE_INTRINSIC_PARAMETERS}: the path where the intrinsic parameters were saved
 * returns a tuple with those values
'''
def readIntrinsicParameters(PATH_SAVE_INTRINSIC_PARAMETERS):
    objects = []
    with (open(PATH_SAVE_INTRINSIC_PARAMETERS, "rb")) as openfile:
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
 * param {img}: the image that we want to find the camera pose
 * params {PATH_SAVE_INTRINSIC_PARAMETERS}: the path where the intrinsic parameters were saved
 * params {PATTERN_SIZE}: the pattern the algorithm is going to look for in the chessboard
 * it calculates the camera position and the projection matrix
 * returns the camera position and the projection matrix
'''
def camera_position(img, PATH_SAVE_INTRINSIC_PARAMETERS, PATTERN_SIZE):
    (mtx, dist) = readIntrinsicParameters(PATH_SAVE_INTRINSIC_PARAMETERS)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE, None)

    # 3d points
    objp = np.zeros((PATTERN_SIZE[0]* PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2) * 22

    # solvePnP requires camera calibraiton
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
        ret, rvec, tvec, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]
                          ).reshape(-1, 3) * AXIS_SIZE
        axis_img, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)

        if SHOW_AXIS_IMAGE:
            draw(img, corners, axis_img)
            cv.imshow('img', img)
            cv.waitKey()
            cv.destroyAllWindows()

        rotM = cv.Rodrigues(rvec)[0]
        real_word_position = -np.matrix(rotM).T * np.matrix(tvec)
        position_normalized = np.linalg.inv(mtx) * real_word_position

        print('real world position (X,Y,Z)= ({}, {}, {})'.format(
            real_word_position[0, 0], real_word_position[1, 0], real_word_position[2, 0]))
        # we can normalize the point: focal length = 1 and moves the origin to the centre of the image
        print('normalize real world position (X,Y,Z)= ({}, {}, {})'.format(
            position_normalized[0, 0], position_normalized[1, 0], position_normalized[2, 0]))
        print('projection matrix : ')
        print(getProjectionMatrix(mtx, rvec, tvec))
        return (position_normalized, getProjectionMatrix(mtx, rvec, tvec))
    else:
        print('could not find position')