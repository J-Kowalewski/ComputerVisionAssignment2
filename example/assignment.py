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


# def set_voxel_positions(width, height, depth):
#     # Generates random voxel locations
#     # TODO: You need to calculate proper voxel arrays instead of random ones.
#     data = []
#     for x in range(width):
#         for y in range(height):
#             for z in range(depth):
#                 if random.randint(0, 1000) < 5:
#                     data.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])
#     return data
def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    images = [cv.imread('../data/cam1/foreground.jpg'), cv.imread('../data/cam2/foreground.jpg'),
              cv.imread('../data/cam3/foreground.jpg'), cv.imread('../data/cam4/foreground.jpg')]
    rvecs = [rvecs1, rvecs2, rvecs3, rvecs4]
    tvecs = [tvecs1, tvecs2, tvecs3, tvecs4]
    cmatrices = [cmatrix1, cmatrix2, cmatrix3, cmatrix4]
    dists = [dist1, dist2, dist3, dist4]
    masks = [mask1, mask2, mask3, mask4]
    data = []
    for x in range(0, 1800, 40):
        for y in range(0, 1800, 40):
            for z in range(0, 1800, 40):
                flag = True
                point = np.float32([x, y, z])
                for i in range(0, 4):
                    img = images[i]
                    rvec = rvecs[i]
                    tvec = tvecs[i]
                    cmatrix = cmatrices[i]
                    dist = dists[i]
                    table = masks[i]
                    point2d, _ = cv.projectPoints(point, rvec, tvec, cmatrix, dist)
                    point2d = tuple(map(int, point2d.ravel()))
                    if table[point2d[1], point2d[0]] == 0:
                        flag = False
                if flag:
                    data.append([x * .1, z * .1, y * .1])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    rvec = np.array((rvecs1[0], -rvecs1[2], -rvecs1[1]))
    R1 = cv.Rodrigues(rvec)[0]
    x1, y1, z1 = -np.array(R1).T * np.matrix(tvecs1)

    rvec = np.array((rvecs2[0], -rvecs2[2], rvecs2[1]))
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
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

    rvecs = [rvecs1, rvecs2, rvecs3, rvecs4]
    rMats = []
    for c in range(len(rvecs)):
        rvec = np.array((rvecs[c][0], -rvecs[c][2], -rvecs[c][1]))
        rMat = cv.Rodrigues(rvec)[0]
        rMat2 = np.identity(4)
        rMat2[:3, :3] = rMat
        rMats.append(rMat2)

    cam_angles = [[0, 0, -90], [0, 0, -90], [0, 0, -90], [0, 0, -90]]
    cam_rotations = [glm.mat4(rMats[0]), glm.mat4(rMats[1]), glm.mat4(rMats[2]), glm.mat4(rMats[3])]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
