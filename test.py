import glm
import numpy as np
import cv2 as cv

import utils

s1 = cv.FileStorage('data/cam1/config.xml', cv.FileStorage_READ)
cmatrix1 = s1.getNode('CameraMatrix').mat()
dist1 = s1.getNode('DistortionCoeffs').mat()
rvecs1 = s1.getNode('Rvecs').mat()

print(utils.changeRodrigues(rvecs1))
