import glob
import numpy as np
import cv2 as cv
import pickle

############# intrinsic parameters ##################
# this will enable us to correct the distortions

IMAGES_PATH_EXPRESSION = './imgs/chessboard5/*.jpg'
PATTERN_SIZE = (9, 6)
CHECKBOARD_SIZE = (9, 6)
FIELD_SIZE = 22 # it can represented in mm. For squares
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
SHOW_IMAGES = False
PATH_SAVE_INTRINSIC_PARAMS = "calibration/wide_dist_pickle.p"
SAVE_PARAMETERS = True

# if we are going to the always the same with the same camera we can save the values to use later
def save_intrinsicParameters(mtx, dist):
    to_save = {}
    to_save['mtx'] = mtx
    to_save['dist'] = dist
    pickle.dump(to_save,  open( PATH_SAVE_INTRINSIC_PARAMS, "wb" ))
    

def reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Reprojection error= {}".format(mean_error / len(objpoints)))

def print_intrinsic_parameters(cameraMatrix, distCoeffs):
    print ("Camera Matrix = |fx  0 cx|")
    print ("                | 0 fy cy|")
    print ("                | 0  0  1|")
    print("fx= {} | fy = {}".format(cameraMatrix[0][0], cameraMatrix[1][1]))
    print("cx= {} | cy = {}".format(cameraMatrix[0][2], cameraMatrix[1][2]))

    print ("\nDistortion Coefficients = [k1, k2, p1, p2, k3]: {}".format(distCoeffs[0]))


def camera_calibration():
    images = glob.glob(IMAGES_PATH_EXPRESSION)
    if len(images) == 0 :
        print('The folder calibration should not be empty if you wish to calibrate the camera')
        exit(1)
    elif len(images) < 10 :
        print('warning: Having at least 10 different images with different orientation and positions is advisable')

    # z-depth = 0 assumed
    object_points = np.zeros((CHECKBOARD_SIZE[0] * CHECKBOARD_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:CHECKBOARD_SIZE[0], 0: CHECKBOARD_SIZE[1]].T.reshape(-1,2) *  FIELD_SIZE

    objPoints = [] 
    imgPoints = [] 

    print("Analysing chessboard images...")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, CHECKBOARD_SIZE,\
                                                 cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objPoints.append(object_points)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), TERMINATION_CRITERIA)
            imgPoints.append(corners)
            if SHOW_IMAGES:
                cv.drawChessboardCorners(img, (9,6), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        else:
            print(fname," no point found")


    if SHOW_IMAGES:
        cv.destroyAllWindows()

    if len(objPoints) > 0 :
        print ("Running Calibration...")
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, \
                                                                             CHECKBOARD_SIZE, None, None)
        print_intrinsic_parameters(cameraMatrix, distCoeffs)
        reprojection_error(objPoints, imgPoints, cameraMatrix, distCoeffs, rvecs, tvecs)

        if SAVE_PARAMETERS:
            save_intrinsicParameters(cameraMatrix, distCoeffs)
    else:
        print("no points")
