import cv2 as cv
import numpy as np
import glob

import utils

fs = cv.FileStorage('data/checkerboard.xml', cv.FILE_STORAGE_READ)

width = fs.getNode('CheckerBoardWidth').real()
height = fs.getNode('CheckerBoardHeight').real()
square_size = fs.getNode('CheckerBoardSquareSize').real()

chesssize = (int(width), int(height))
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        global interpolatedCoords
        global userCoordinates
        global inv_p_matrix

        cv.circle(img, (x, y), 5, (255, 0, 0), -1)

        userCoordinates = np.append(userCoordinates, np.array([x, y]))
        cv.imshow('img', img)

        if userCoordinates.size == 8:
            #interpolatedCoords = utils.testWarp(img, userCoordinates)
            interpolatedCoords = utils.newInterpolation(img,userCoordinates)


for i in range(1, 5):
    # prepare object points (0,0,0), (1,0,0) etc...
    objp = np.zeros((chesssize[0] * chesssize[1], 3), np.float32)
    objp[:, :2] = (square_size * np.mgrid[0:8, 0:6]).T.reshape(-1, 2)

    # we use some arrays to store the object points from all the images
    objPoints = []  # 3D points of the image plane
    imgPoints = []  # 2D points of the image plane

    cap = cv.VideoCapture('data/cam{:d}/checkerboard.avi'.format(i))
    ret, frame1 = cap.read()
    if ret:
        img = np.copy(frame1)
        #img = utils.rescaleFrame(img, 2)
        interpolatedCoords = np.array([])
        userCoordinates = np.array([])
        inv_p_matrix = np.array([])
        cv.imshow('img', img)
        cv.setMouseCallback('img', click_event)
        cv.waitKey(0)
            #img = utils.rescaleFrame(img, .5)

        objPoints.append(objp)

        cv.destroyAllWindows()

        imgPoints.append(np.float32(interpolatedCoords))

        fs = cv.FileStorage('data/cam{:d}/intrinsics.xml'.format(i), cv.FILE_STORAGE_READ)

        CameraMatrix = fs.getNode('CameraMatrix').mat()
        dist = fs.getNode('DistortionCoeffs').mat()

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, 3]]).reshape(-1, 3)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners2 = cv.cornerSubPix(gray, interpolatedCoords, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, CameraMatrix, dist)
        if ret:
            # R, _ = cv.Rodrigues(rvecs)

            # Find the rotation and translation vectors
            # Project 3D points to the image plain
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, CameraMatrix, dist)
            # Draw axis
            # utils.drawAxes(img, corners2, imgpts)
            cv.drawFrameAxes(frame1, CameraMatrix, dist, rvecs, tvecs, 400)
            frame1 = utils.rescaleFrame(frame1, 2)
            cv.imshow('frame', frame1)

            s = cv.FileStorage('data/cam{:d}/config.xml'.format(i), cv.FileStorage_WRITE)
            s.write('CameraMatrix', CameraMatrix)
            s.write('DistortionCoeffs', dist)
            s.write('Rvecs', rvecs)
            s.write('Tvecs', tvecs)
            s.release()

        cv.waitKey(0)
        cv.destroyAllWindows()
