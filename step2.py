####### camera pose - extrinsic parameters ##########
import pickle
import cv2 as cv
import numpy as np
from utils import getProjectionMatrix

AXIS_SIZE = 3
SHOW_AXIS_IMAGE = False


def readIntrinsicParameters():
    objects = []
    with (open("calibration/alternate/wide_dist_pickle.p", "rb")) as openfile:
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


def draw(img, corners, imgpts):
    """Draw Axes"""
    corner = tuple(corners[0].ravel())
    cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

# if our xy plane is known (extrinsic and intrinsic parameters for a certain camara pose)
# we can intersect the ray formed by two different points
# if the point is strictly in the xy plane we can intersect the xy plane with the ray that passes through the optical centre and
# the respective point in the image plane that is in the base


def camera_position(img):
    (mtx, dist) = readIntrinsicParameters()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # 3d points
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 15

    #   - solvePnP requires camera calibraiton
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret, rvec, tvec = cv.solvePnP(objp, corners2, mtx, dist)
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
        return (position_normalized, rvec, tvec, rotM, real_word_position, getProjectionMatrix(mtx, rvec, tvec))
    else:
        print('could not find position')
