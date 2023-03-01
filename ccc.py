import cv2 as cv
import numpy as np

images = [cv.imread('data/cam1/foreground.jpg'), cv.imread('data/cam2/foreground.jpg'),
          cv.imread('data/cam3/foreground.jpg'), cv.imread('data/cam4/foreground.jpg')]
s = cv.FileStorage('data/masks.xml', cv.FileStorage_WRITE)
for x in range(0, 4):
    img = images[x]
    imheight, imwidth, imdepth = img.shape
    table = np.zeros([imheight, imwidth])
    for i in range(0, imheight):
        for j in range(0, imwidth):
            if np.any(img[i, j] != 0):
                table[i][j] = 1

    s.write('mask{:d}'.format(x + 1), table)

s.release()
