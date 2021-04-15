import glob
import numpy as np
import cv2 as cv
import pickle


'''
    First step: Camera calibration, namely finding the intrinsic parameters of
    the camera

'''


'''
 Global configuration variables that are used in this first step
 * IMAGES_PATH_EXPRESSION: where the chessboard images that are going to be used to calibrate da camera are
 * FIELD_SIZE: the real size of the boarders squares in mm, just to ease interpretation in later stages
 * SHOW_IMAGES: if true, it shows the patterns found in the chessboards
'''
IMAGES_PATH_EXPRESSION = './imgs/chessboard5/*.jpg'
FIELD_SIZE = 22 # it can represented in mm. For squares
TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
SHOW_IMAGES = False
'''
* param {mtx}: camera matrix
* param {dist}: distortion parameters
* params {PATH_SAVE_INTRINSIC_PARAMETERS}: the path where the intrinsic parameters are to be saved
* it saves the camera matrix and distortion parameters in a file
* return void
'''
def save_intrinsicParameters(mtx, dist, PATH_SAVE_INTRINSIC_PARAMS):
    to_save = {}
    to_save['mtx'] = mtx
    to_save['dist'] = dist
    pickle.dump(to_save,  open( PATH_SAVE_INTRINSIC_PARAMS, "wb" ))
    
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
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Reprojection error= {}".format(mean_error / len(objpoints)))


'''
* param {cameraMatrix} : the camera matrix
* param {distCoeffs}: distortion coefficient of the camera 
* prints the camera matrix and distortion coefficients of the camera
* return: void
'''
def print_intrinsic_parameters(cameraMatrix, distCoeffs):
    print ("Camera Matrix = |fx  0 cx|")
    print ("                | 0 fy cy|")
    print ("                | 0  0  1|")
    print("fx= {} | fy = {}".format(cameraMatrix[0][0], cameraMatrix[1][1]))
    print("cx= {} | cy = {}".format(cameraMatrix[0][2], cameraMatrix[1][2]))

    print ("\nDistortion Coefficients = [k1, k2, p1, p2, k3]: {}".format(distCoeffs[0]))


'''
 * params {PATH_SAVE_INTRINSIC_PARAMETERS}: the path where the intrinsic parameters are going to be saved
 * params {PATTERN_SIZE}: the pattern the algorithm is going to look for in the chessboard
 * calibrate the camera by using a checkboard, it calculates the camera intrinsic parameters and distortion parameters
   the parameters are saved in a file since we only need to calculate the parameters once for each camera 
 * return void
'''
def camera_calibration(PATTERN_SIZE, PATH_SAVE_INTRINSIC_PARAMS):
    images = glob.glob(IMAGES_PATH_EXPRESSION)
    if len(images) == 0 :
        print('The folder calibration should not be empty if you wish to calibrate the camera')
        exit(1)
    elif len(images) < 10 :
        print('warning: Having at least 10 different images with different orientation and positions is advisable')

    # z-depth = 0 assumed
    object_points = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0: PATTERN_SIZE[1]].T.reshape(-1,2) *  FIELD_SIZE

    objPoints = [] 
    imgPoints = [] 

    print("Analysing chessboard images...")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, PATTERN_SIZE,\
                                                 cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)                                
        if ret:
            objPoints.append(object_points)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), TERMINATION_CRITERIA)
            imgPoints.append(corners)

            # Draw and display the corners if flag is true
            if SHOW_IMAGES:
                cv.drawChessboardCorners(img, PATTERN_SIZE, corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        else:
            print(fname," no point found")


    if SHOW_IMAGES:
        cv.destroyAllWindows()

    if len(objPoints) > 0 :
        print ("Running Calibration...")
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, \
                                                                             PATTERN_SIZE, None, None)
        print_intrinsic_parameters(cameraMatrix, distCoeffs)
        reprojection_error(objPoints, imgPoints, cameraMatrix, distCoeffs, rvecs, tvecs)

        save_intrinsicParameters(cameraMatrix, distCoeffs, PATH_SAVE_INTRINSIC_PARAMS)
    else:
        print("no points")
