import numpy as np
import cv2


def adjust_threshold(slider_pos):
    global threshold
    threshold = slider_pos


def hist_calc_delay(slider_pos):
    global frames_delay
    frames_delay = slider_pos


cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

nbins = 256

# capture the first frame
ret, frame = cap.read()

# slightly blur the frame to reduce noise from the camera
frame = cv2.blur(frame, (3, 3))

# calculate the red component histogram and normalize it
old_r_hist = cv2.calcHist([frame], [2], None, [nbins], [0, 256])
cv2.normalize(old_r_hist, old_r_hist, cv2.NORM_MINMAX).flatten()

iteration = 0
threshold = 5
frames_delay = 10

# create the window an the trackbars
cv2.namedWindow('motion detection')
cv2.createTrackbar('Threshold adjuster', 'motion detection', 5, 50, adjust_threshold)
cv2.createTrackbar('Histogram calculation delay\n(in number of frames)', 'motion detection', 10, 50, hist_calc_delay)

while True:
    ret, frame = cap.read()

    iteration += 1

    # slightly blur the frame to reduce noise from the camera
    frame = cv2.blur(frame, (3, 3))

    # calculate the red component of the new histogram
    new_r_hist = cv2.calcHist([frame], [2], None, [nbins], [0, 256])
    cv2.normalize(new_r_hist, new_r_hist, cv2.NORM_MINMAX).flatten()

    # compare the histograms with chi-squared method
    result = cv2.compareHist(old_r_hist, new_r_hist, cv2.HISTCMP_CHISQR)

    # if the result is larger then the threshold, some movement was detected
    if result >= threshold:
        cv2.putText(frame, 'MOTION DETECTED!',
                    org=(10, height-10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2)

    # change the old histogram after the set amount of frames has passed
    if iteration >= frames_delay:
        # put the current histogram as the old histogram for the next comparison
        old_r_hist = new_r_hist.copy()
        iteration = 0

    # show the current frame
    cv2.imshow('motion detection', frame)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close the program
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()