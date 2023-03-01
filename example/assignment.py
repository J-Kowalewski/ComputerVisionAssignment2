import glm
import random
import numpy as np
import cv2 as cv
import utils

block_size = 1.0

s1 = cv.FileStorage('../data/cam1/config.xml', cv.FileStorage_READ)
cmatrix1 = s1.getNode('CameraMatrix').mat()
dist1 = s1.getNode('DistortionCoeffs').mat()
rvecs1 = s1.getNode('Rvecs').mat()
tvecs1 = s1.getNode('Tvecs').mat()

s2 = cv.FileStorage('../data/cam2/config.xml', cv.FileStorage_READ)
cmatrix2 = s2.getNode('CameraMatrix').mat()
dist2 = s2.getNode('DistortionCoeffs').mat()
rvecs2 = s2.getNode('Rvecs').mat()
tvecs2 = s2.getNode('Tvecs').mat()

s3 = cv.FileStorage('../data/cam3/config.xml', cv.FileStorage_READ)
cmatrix3 = s3.getNode('CameraMatrix').mat()
dist3 = s3.getNode('DistortionCoeffs').mat()
rvecs3 = s3.getNode('Rvecs').mat()
tvecs3 = s3.getNode('Tvecs').mat()

s4 = cv.FileStorage('../data/cam4/config.xml', cv.FileStorage_READ)
cmatrix4 = s4.getNode('CameraMatrix').mat()
dist4 = s4.getNode('DistortionCoeffs').mat()
rvecs4 = s4.getNode('Rvecs').mat()
tvecs4 = s4.getNode('Tvecs').mat()

s5 = cv.FileStorage('../data/checkerboard.xml', cv.FileStorage_READ)
squareSize = s5.getNode('CheckerBoardSquareSize').real()

s6 = cv.FileStorage('../data/masks.xml', cv.FileStorage_READ)
mask1 = s6.getNode('mask1').mat()
mask2 = s6.getNode('mask2').mat()
mask3 = s6.getNode('mask3').mat()
mask4 = s6.getNode('mask4').mat()


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def set_voxel_positions(width, height, depth):
    storage = cv.FileStorage('../data/lookuptable.xml', cv.FileStorage_READ)
    table = storage.getNode('table').mat()
    shape = table.shape
    data = []
    for i in range(0, shape[0]):
        if table[i][3] == 1:
            data.append([table[i][0], table[i][1], table[i][2]])
    return data


def get_cam_positions():
    rvec = np.array((rvecs1[0], -rvecs1[2], -rvecs1[1]))
    R1 = cv.Rodrigues(rvec)[0]
    x1, y1, z1 = -np.array(R1).T * np.matrix(tvecs1)

    rvec = np.array((rvecs2[0], -rvecs2[2], -rvecs2[1]))
    R2 = cv.Rodrigues(rvec)[0]
    x2, y2, z2 = -np.array(R2).T * np.matrix(tvecs2)

    rvec = np.array((rvecs3[0], -rvecs3[2], -rvecs3[1]))
    R3 = cv.Rodrigues(rvec)[0]
    x3, y3, z3 = -np.array(R3).T * np.matrix(tvecs3)

    rvec = np.array((rvecs4[0], -rvecs4[2], -rvecs4[1]))
    R4 = cv.Rodrigues(rvec)[0]
    x4, y4, z4 = -np.array(R4).T * np.matrix(tvecs4)

    return [[x1 * block_size / squareSize, z1 * block_size / squareSize, -y1 * block_size / squareSize],
            [x2 * block_size / squareSize, z2 * block_size / squareSize, -y2 * block_size / squareSize],
            [x3 * block_size / squareSize, z3 * block_size / squareSize, -y3 * block_size / squareSize],
            [x4 * block_size / squareSize, z4 * block_size / squareSize, -y4 * block_size / squareSize]]


def get_cam_rotation_matrices():
    rvecs = [rvecs1, rvecs2, rvecs3, rvecs4]
    rMatrices = []
    for c in range(len(rvecs)):
        rvec = np.array((rvecs[c][0], -rvecs[c][2], -rvecs[c][1]))
        rMatrix = cv.Rodrigues(rvec)[0]
        rMatrix2 = np.identity(4)
        rMatrix2[:3, :3] = rMatrix
        rMatrices.append(rMatrix2)

    cam_angles = [[0, 0, -90], [0, 0, -90], [0, 0, -90], [0, 0, -90]]
    cam_rotations = [glm.mat4(rMatrices[0]), glm.mat4(rMatrices[1]), glm.mat4(rMatrices[2]), glm.mat4(rMatrices[3])]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
