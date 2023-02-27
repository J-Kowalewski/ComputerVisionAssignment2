import cv2 as cv

cap1 = cv.VideoCapture('data/cam1/intrinsics.avi')
cap2 = cv.VideoCapture('data/cam2/intrinsics.avi')
cap3 = cv.VideoCapture('data/cam3/intrinsics.avi')
cap4 = cv.VideoCapture('data/cam4/intrinsics.avi')

count = 0

while cap1.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    if count/150 == 20:
        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()
        break
    elif ret1 and ret2 and ret3 and ret4:
        cv.imwrite('intrinsicsFrames/cam1/frame{:d}.jpg'.format(int(count/150)), frame1)
        cv.imwrite('intrinsicsFrames/cam2/frame{:d}.jpg'.format(int(count / 150)), frame2)
        cv.imwrite('intrinsicsFrames/cam3/frame{:d}.jpg'.format(int(count / 150)), frame3)
        cv.imwrite('intrinsicsFrames/cam4/frame{:d}.jpg'.format(int(count / 150)), frame4)
        count += 150  # i.e. at 50 fps, this advances 3 seconds
        cap1.set(cv.CAP_PROP_POS_FRAMES, count)
        cap2.set(cv.CAP_PROP_POS_FRAMES, count)
        cap3.set(cv.CAP_PROP_POS_FRAMES, count)
        cap4.set(cv.CAP_PROP_POS_FRAMES, count)
    else:
        cap1.release()
        cap2.release()
        cap3.release()
        cap4.release()
        break
