import numpy as np
import cv2 as cv

# Open Video
cap = cv.VideoCapture('data/cam3/video.avi')
# Open background image
medianFrame = cv.imread('data/cam3/background_model.jpg')
# Convert background to grayscale
hsv_bg = cv.cvtColor(medianFrame, cv.COLOR_BGR2HSV)
# Loop over all frames
while True:
    # Read frame
    ret, frame = cap.read()
    if frame is None:
        break
    # Convert current frame to HSV
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Calculate absolute difference of current frame and the background frame
    dframe = cv.absdiff(frame, hsv_bg)
    cv.imshow('absdiff', dframe)
    # Blur the image
    img = cv.medianBlur(dframe, 5)
    cv.imshow('blur', img)
    # Mask out low value and low saturated parts
    lower1 = np.array([0, 5, 5])
    upper1 = np.array([255, 255, 255])
    mask = cv.inRange(img, lower1, upper1)
    result = cv.bitwise_and(img, img, mask=mask)
    cv.imshow('mask', result)
    # Convert to grayscale
    grayImage = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', grayImage)
    # Convert to black and white and apply threshold
    (thresh, blackAndWhiteImage) = cv.threshold(grayImage, 30, 255, cv.THRESH_BINARY)
    cv.imshow('blacknwhite', blackAndWhiteImage)
    # Find contours and fill all the gaps
    contour, hier = cv.findContours(blackAndWhiteImage, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv.drawContours(blackAndWhiteImage, [cnt], 0, 255, -1)
    cv.imshow('contours', blackAndWhiteImage)

    # ----------------------------------------------------------------------------
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(blackAndWhiteImage)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 5000

    # output image with only the kept components
    im_result = np.zeros(blackAndWhiteImage.shape)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255
    cv.imshow('removed', im_result)
    # -------------------------------------------------------------------------------------------

    cv.waitKey(20)

cv.waitKey(0)
# Release video object
cap.release()
cv.imwrite('data/cam3/foreground.jpg', im_result)
# Destroy all windows
cv.destroyAllWindows()
