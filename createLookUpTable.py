import glm
import numpy as np
import cv2 as cv

s1 = cv.FileStorage('data/cam1/config.xml', cv.FileStorage_READ)
cmatrix1 = s1.getNode('CameraMatrix').mat()
dist1 = s1.getNode('DistortionCoeffs').mat()
rvecs1 = s1.getNode('Rvecs').mat()
tvecs1 = s1.getNode('Tvecs').mat()

s2 = cv.FileStorage('data/cam2/config.xml', cv.FileStorage_READ)
cmatrix2 = s2.getNode('CameraMatrix').mat()
dist2 = s2.getNode('DistortionCoeffs').mat()
rvecs2 = s2.getNode('Rvecs').mat()
tvecs2 = s2.getNode('Tvecs').mat()

s3 = cv.FileStorage('data/cam3/config.xml', cv.FileStorage_READ)
cmatrix3 = s3.getNode('CameraMatrix').mat()
dist3 = s3.getNode('DistortionCoeffs').mat()
rvecs3 = s3.getNode('Rvecs').mat()
tvecs3 = s3.getNode('Tvecs').mat()

s4 = cv.FileStorage('data/cam4/config.xml', cv.FileStorage_READ)
cmatrix4 = s4.getNode('CameraMatrix').mat()
dist4 = s4.getNode('DistortionCoeffs').mat()
rvecs4 = s4.getNode('Rvecs').mat()
tvecs4 = s4.getNode('Tvecs').mat()

s5 = cv.FileStorage('data/checkerboard.xml', cv.FileStorage_READ)
squareSize = s5.getNode('CheckerBoardSquareSize').real()

s6 = cv.FileStorage('data/masks.xml', cv.FileStorage_READ)
mask1 = s6.getNode('mask1').mat()
mask2 = s6.getNode('mask2').mat()
mask3 = s6.getNode('mask3').mat()
mask4 = s6.getNode('mask4').mat()

images = [cv.imread('data/cam1/foreground.jpg'), cv.imread('data/cam2/foreground.jpg'),
          cv.imread('data/cam3/foreground.jpg'), cv.imread('data/cam4/foreground.jpg')]

rvecs = [rvecs1, rvecs2, rvecs3, rvecs4]
tvecs = [tvecs1, tvecs2, tvecs3, tvecs4]
cmatrices = [cmatrix1, cmatrix2, cmatrix3, cmatrix4]
dists = [dist1, dist2, dist3, dist4]
masks = [mask1, mask2, mask3, mask4]

data = np.empty((0, 4), int)
s = cv.FileStorage('data/lookuptable.xml', cv.FileStorage_WRITE)
for x in range(0, 1800, 40):
    for y in range(0, 1800, 40):
        for z in range(0, 1800, 40):
            flag = True
            point = np.float32([x, y, z])
            global table
            global point2d
            projectedPoints = []
            for i in range(0, 4):
                img = images[i]
                rvec = rvecs[i]
                tvec = tvecs[i]
                cmatrix = cmatrices[i]
                dist = dists[i]
                table = masks[i]
                point2d, _ = cv.projectPoints(point, rvec, tvec, cmatrix, dist)
                point2d = tuple(map(int, point2d.ravel()))
                projectedPoints.append([point2d])
                # TODO change the if statement
                if table[point2d[1], point2d[0]] == 0:
                    flag = False
            # TODO add projected points to matrix?
            #data = np.append(data, np.array([[[x * .01, z * .01, -y * .01], projectedPoints, flag]]), axis=0)
            data = np.append(data, np.array([[x * .01, z * .01, -y * .01, flag]]), axis=0)

s.write('table', data)
