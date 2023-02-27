import cv2 as cv
import numpy as np
import glob

fs = cv.FileStorage('data/checkerboard.xml', cv.FILE_STORAGE_READ)

width = fs.getNode('CheckerBoardWidth').real()
height = fs.getNode('CheckerBoardHeight').real()
square_size = fs.getNode('CheckerBoardSquareSize').real()

chesssize = (int(width), int(height))
framesize = (644, 486)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points (0,0,0), (1,0,0) etc...
objp = np.zeros((chesssize[0] * chesssize[1], 3), np.float32)
objp[:, :2] = (square_size*np.mgrid[0:chesssize[0], 0:chesssize[1]].T.reshape(-1, 2))

# we use some arrays to store the object points from all the images
objPoints = []  # 3D points of the image plane
imgPoints = []  # 2D points of the image plane

for i in range(2, 3):
    images = glob.glob('intrinsicsFrames/cam{:d}/*.jpg'.format(i))
    for image in images:
        print(image)
        img = cv.imread(image, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # now we find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chesssize, None)

        # if found we need to add the object points and image points (we do refine them)
        if ret:
            objPoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgPoints.append(corners2)

            # Draw and display corners
            cv.drawChessboardCorners(img, chesssize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(300)

    # Calibration

    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
        objPoints, imgPoints, framesize, None, None)

    print("Calibrated Camera:", ret)
    print("\nCamera Matrix:\n", cameraMatrix)
    print("\nDistortion Parameters:\n", dist)
    print("\nRotation Vector:\n", rvecs)
    print("\nTranslation Vector:\n", tvecs)

    s = cv.FileStorage('data/cam{:d}/intrinsics.xml'.format(i), cv.FileStorage_WRITE)
    s.write('CameraMatrix', cameraMatrix)
    s.write('DistortionCoeffs', dist)
    s.release()
