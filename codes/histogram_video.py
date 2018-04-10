import numpy as np
import cv2

cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('height = {}, width = {}'.format(height, width))

nbins = 256

while True:
    ret, frame = cap.read()

    # convert the frame to black and white and then to floating point
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate the histogram
    hist = cv2.calcHist([frame_gray], [0], None, [nbins], [0, 256])

    # calculate the accumulated histogram
    accum_hist = []
    accum_hist.append(int(hist[0]))
    for i in range(1, len(hist)):
        accum_hist.append(accum_hist[i - 1] + int(hist[i]))

    # equalize the image
    eq_frame_gray = cv2.equalizeHist(frame_gray)

    cv2.imshow('Original', frame_gray)
    cv2.imshow('Equalized', eq_frame_gray)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close without saving
    if key == 27:
        break
    # if the space bar is pressed, save and close
    # elif key == 32:
    #     break

cap.release()
cv2.destroyAllWindows()