import cv2 as cv
import numpy as np
import glm


# Rescale image
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Sorts user coordinates
def sortUserCoord(coordinates):
    coordinates = np.reshape(coordinates, (-1, 2))
    # sort all rows by y value
    sortedArr = coordinates[coordinates[:, 1].argsort()]

    # split array into 2 and sort them by x value
    upArray = sortedArr[[0, 1], :]
    upArray = upArray[upArray[:, 0].argsort()]

    downArray = sortedArr[[2, 3], :]
    downArray = downArray[downArray[:, 0].argsort()]
    # flip array vertically so bottom-left point is last
    downArray = np.flipud(downArray)
    # return merged and sorted array (TL,TR,BR,BL)
    return np.vstack((upArray, downArray))


# Transforms the image (warps)
def four_point_transform(image, pts):
    rect = np.float32(np.reshape(pts, (-1, 2)))
    print(rect)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    inv_p_matrix = np.linalg.inv(M)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    # cv.imshow('warped', warped)

    return inv_p_matrix, warped, maxWidth, maxHeight


def drawChess(img):
    chesssize = (8, 6)

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points (0,0,0), (1,0,0) etc...
    objp = np.zeros((chesssize[0] * chesssize[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chesssize[0], 0:chesssize[1]].T.reshape(-1, 2)
    objPoints = []  # 3D points of the image plane
    imgPoints = []  # 2D points of the image plane

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


# Linear interpolation
def linearInter(img_transformed, maxHeight, maxWidth):
    # TL - [0], TR, BR, BL - [3]
    board_size = (7, 9)
    square_sizeX = maxWidth / board_size[0]
    square_sizeY = maxHeight / board_size[1]
    positions = np.array([])
    for j in range(board_size[0] - 1):
        for i in range(board_size[1] - 1):
            x = int((j + 1) * square_sizeX)
            y = int((i + 1) * square_sizeY)
            positions = np.append(positions, np.array([x, y]))

    positions = np.int32(positions).reshape(-1, 2)
    # Draw the positions on the transformed image
    for idx, pos in enumerate(positions):
        cv.circle(img_transformed, pos, 2, (255, 0, 0), -1)

    cv.imshow('chess', img_transformed)
    return positions


def drawAxes(img, corners, imgpts):
    corner = np.int32(tuple(corners[0].ravel()))
    imgpts = np.int32(imgpts).reshape(-1, 2)
    print(corner)
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def testWarp(img, points):
    points = np.array(sortUserCoord(points), np.float32)
    grid_size = (8, 6)  # rows, cols
    # Creating a linear matrix of corner positions
    chessboard_points = []
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            chessboard_points.append([i * 75, j * 75])

    srs = np.float32([
        [0, 0],
        [(grid_size[1] - 1) * 75, 0],
        [(grid_size[1] - 1) * 75, (grid_size[0] - 1) * 75],
        [0, (grid_size[0] - 1) * 75]])
    # Compute the perspective transform matrix
    # Points = manually given points, srs = uniform corners
    M = cv.getPerspectiveTransform(srs, points)
    chessboard_points2 = np.float32(chessboard_points)
    # Using the matrix M to warp the points to the perspective we want
    w_points = cv.perspectiveTransform(chessboard_points2[None, :, :], M)
    for idx, pos in enumerate(w_points[0]):
        cv.circle(img, np.int32(pos), 2, (255, 0, 0), -1)
    cv.imshow('points', img)
    return w_points[0]


def newInterpolation(image, pts):
    inv_p_matrix, warped, maxWidth, maxHeight = four_point_transform(image, pts)
    board_size = (6, 8)
    xlen = board_size[0]
    ylen = board_size[1]

    square_sizeX = maxWidth / (xlen + 1)
    square_sizeY = maxHeight / (ylen + 1)
    positions = np.array([])
    for j in range(xlen):
        for i in range(ylen):
            x = int((j + 1) * square_sizeX)
            y = int((i + 1) * square_sizeY)
            positions = np.append(positions, np.array([x, y]))

    positions = np.int32(positions).reshape(-1, 2)
    for idx, pos in enumerate(positions):
        cv.circle(warped, pos, 2, (255, 0, 0), -1)

    cv.imshow('points', warped)

    positions = np.float32(positions)
    positions = cv.perspectiveTransform(positions[None, :, :], inv_p_matrix)
    for idx, pos in enumerate(np.int32(positions[0])):
        cv.circle(image, pos, 2, (255, 0, 0), -1)
    cv.imshow('points2', image)
    return positions[0]


def changeRodrigues(R):
    R = cv.Rodrigues(R)
    R = R[0]
    final = glm.mat4(float(R[0][0]), float(R[1][0]), float(R[2][0]), 0,
                     float(R[0][1]), float(R[1][1]), float(R[2][1]), 0,
                     float(R[0][2]), float(R[1][2]), float(R[2][2]), 0,
                     0, 0, 0, 1)

    return final
