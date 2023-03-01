from __future__ import print_function
import cv2 as cv

MOG = False

for i in range(1, 5):
    capture = cv.VideoCapture('data/cam{:d}/background.avi'.format(i))
    if MOG:
        backSub = cv.createBackgroundSubtractorMOG2()
    else:
        backSub = cv.createBackgroundSubtractorKNN(history=500, dist2Threshold=400.0, detectShadows=False)
    if not capture.isOpened():
        print('Unable to open')
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', fgMask)

        keyboard = cv.waitKey(30)

    background_model = backSub.getBackgroundImage()
    cv.imwrite('data/cam{:d}/background_model.jpg'.format(i), background_model)
