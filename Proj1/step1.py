import glob
import numpy as np
import cv2 as cv
import pickle


'''
    First step: Camera calibration, namely finding the intrinsic parameters of
    the camera

'''

TERMINATION_CRITERIA = (cv.TERM_CRITERIA_EPS +
                        cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
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
    pickle.dump(to_save,  open(PATH_SAVE_INTRINSIC_PARAMS, "wb"))


'''
* param {cameraMatrix} : the camera matrix
* param {distCoeffs}: distortion coefficient of the camera 
* prints the camera matrix and distortion coefficients of the camera
* return: void
'''


def print_intrinsic_parameters(cameraMatrix, distCoeffs):
    print("Camera Matrix = |fx  0 cx|")
    print("                | 0 fy cy|")
    print("                | 0  0  1|")
    print("fx= {} | fy = {}".format(cameraMatrix[0][0], cameraMatrix[1][1]))
    print("cx= {} | cy = {}".format(cameraMatrix[0][2], cameraMatrix[1][2]))

    print("\nDistortion Coefficients = [k1, k2, p1, p2, k3]: {}".format(
        distCoeffs[0]))


'''
 * params {step1Config}: values extracted from the config file needed for this step
 * calibrate the camera by using a checkboard, it calculates the camera intrinsic parameters and distortion parameters
   the parameters are saved in a file since we only need to calculate the parameters once for each camera 
 * return void
'''


def camera_calibration(step1Config, showSteps):
    images = glob.glob(step1Config['Calibration Images Folder'])
    if len(images) == 0:
        print(
            'The folder calibration should not be empty if you wish to calibrate the camera')
        exit(1)
    elif len(images) < 10:
        print('warning: Having at least 10 different images with different orientation and positions is advisable')

    # z-depth = 0 assumed
    object_points = np.zeros(
        (step1Config['Chessboard Pattern Size'][0] * step1Config['Chessboard Pattern Size'][1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0:step1Config['Chessboard Pattern Size'][0],
                                    0: step1Config['Chessboard Pattern Size'][1]].T.reshape(-1, 2) * step1Config['Chessboard Field Size']

    objPoints = []
    imgPoints = []

    print("Analysing chessboard images...")
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, step1Config['Chessboard Pattern Size'],
                                                cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
            objPoints.append(object_points)
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
            imgPoints.append(corners)

            # Draw and display the corners if flag is true
            if showSteps:
                cv.drawChessboardCorners(
                    img, step1Config['Chessboard Pattern Size'], corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        else:
            print(fname, " no point found")

    if showSteps:
        cv.destroyAllWindows()

    if len(objPoints) > 0:
        print("Running Calibration...")
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints,
                                                                            step1Config['Chessboard Pattern Size'], None, None)
        print_intrinsic_parameters(cameraMatrix, distCoeffs)


        save_intrinsicParameters(
            cameraMatrix, distCoeffs, step1Config['Path Save Intrisic Params'])
    else:
        print("no points")
