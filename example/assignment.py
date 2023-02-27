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


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    R1, _ = cv.Rodrigues(rvecs1)
    x1, y1, z1 = -np.array(R1).T * np.matrix(tvecs1)
    R2, _ = cv.Rodrigues(rvecs2)
    x2, y2, z2 = -np.array(R2).T * np.matrix(tvecs2)
    R3, _ = cv.Rodrigues(rvecs3)
    x3, y3, z3 = -np.array(R3).T * np.matrix(tvecs3)
    R4, _ = cv.Rodrigues(rvecs4)
    x4, y4, z4 = -np.array(R4).T * np.matrix(tvecs4)

    return [[x1 * block_size / squareSize, z1 * block_size / squareSize, y1 * block_size / squareSize],
            [x2 * block_size / squareSize, z2 * block_size / squareSize, y2 * block_size / squareSize],
            [x3 * block_size / squareSize, z3 * block_size / squareSize, y3 * block_size / squareSize],
            [x4 * block_size / squareSize, z4 * block_size / squareSize, y4 * block_size / squareSize]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    R1, _ = cv.Rodrigues(rvecs1)
    R2, _ = cv.Rodrigues(rvecs2)
    R3, _ = cv.Rodrigues(rvecs3)
    R4, _ = cv.Rodrigues(rvecs4)

    cam_angles = [R1, R2, R3, R4]
    print(cam_angles)
    tvecArr = [tvecs1, tvecs2, tvecs3, tvecs4]
    cam_rotations = []
    for c in range(len(cam_angles)):
        glm_mat = glm.mat4(float(cam_angles[c][0][0]), float(cam_angles[c][2][0]), float(cam_angles[c][1][0]), 0,
                           float(cam_angles[c][0][1]), float(cam_angles[c][2][1]), float(cam_angles[c][1][1]), 0,
                           float(cam_angles[c][0][2]), float(cam_angles[c][2][2]), float(cam_angles[c][1][2]), 0,
                           0, 0, 0, 1)
        print('----')
        print(glm_mat)
        glm_mat = glm.rotate(glm_mat, glm.radians(90), (0, 1, 1))
        cam_rotations.append(glm_mat)

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], tvecArr[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], tvecArr[c][2] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], tvecArr[c][1] * np.pi / 180, [0, 0, 1])
    return cam_rotations
