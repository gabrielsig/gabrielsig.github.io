import numpy as np
import cv2


#=====================================================================

def on_menu_change(slider_pos):
    global filter_type
    filter_type = slider_pos


def on_absolute_change(slider_pos):
    global absolute
    absolute = slider_pos


def on_intensity_change(slider_pos):
    global filter_intensity
    filter_intensity = slider_pos + 1


def apply_filter(src_img, filter_type, n_passes=1):

    # create a generic 3x3 mask
    mask = np.zeros([3, 3])

    # check which filter was selected and apply the correponding mask
    if filter_type == 0:
        # no Filter
        return src_img
    elif filter_type == 1:
        # mean filter
        # cv2.blur()
        mask = np.ones([3, 3]) / (3**2)
    elif filter_type == 2:
        # gaussian filter
        # cv2.GaussianBlur()
        # cv2.getGaussianKernel()
        mask = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16
    elif filter_type == 3:
        # vertical sobel border detector
        # cv2.Sobel()
        mask = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    elif filter_type == 4:
        # horizontal sobel border detector
        # cv2.Sobel()
        mask = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    elif filter_type == 5:
        # laplacian filter
        # cv2.Laplacian()
        mask = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
    elif filter_type == 6:
        # laplacian of gaussian filter, so we first use the gaussian mask
        mask = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16
    else:
        print('[ERROR]: could not select one of the available filters')

    # apply the filter with the selected mask
    out_img = cv2.filter2D(src_img, -1, mask)

    # apply multiple times if n_passes > 1
    if n_passes > 1:
        for n in range(0, n_passes-1):
            out_img = cv2.filter2D(out_img, -1, mask)

    # if we select the laplacian of gaussian, we replace the gaussian mask with the laplacian
    # and apply the filter again
    if filter_type == 6:
        mask = np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
        out_img = cv2.filter2D(out_img, -1, mask)

    return out_img

# ====================================================================

cap = cv2.VideoCapture()

cap.open(1)

if not cap.isOpened():
    print('[ERROR]: Camera is unavailable')
    exit(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# default filter is no filter
filter_type = 0
absolute = 0
filter_intensity = 1
menu_slider_name = '0: No filter\n' \
                   '1: Mean\n' \
                   '2: Gaussian\n' \
                   '3: Vertical Sobel\n' \
                   '4: Horizontal Sobel\n' \
                   '5: Laplacian\n' \
                   '6: Laplacian of Gaussian'
absolute_slider_name = '0: Normalized\n' \
                       '1: Absolute'

cv2.namedWindow('spatial filter')
cv2.createTrackbar(menu_slider_name, 'spatial filter', 0, 6, on_menu_change)
cv2.createTrackbar(absolute_slider_name, 'spatial filter', 0, 1, on_absolute_change)
cv2.createTrackbar('Filter intensity (+1)', 'spatial filter', 0, 9, on_intensity_change)

while True:
    ret, frame = cap.read()

    # convert the frame to black and white and then to floating point
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray32 = np.array(frame_gray, dtype=np.float32)

    # apply the filter
    filter_output = apply_filter(frame_gray32, filter_type, filter_intensity)

    # check if the absolute option is True
    if absolute:
        filter_output = np.absolute(filter_output)
    else:
        filter_output = cv2.normalize(filter_output, None, 0, 255, cv2.NORM_MINMAX)

    # convert the result back to a uint8 image
    result = np.array(filter_output, dtype=np.uint8)

    # show processed image
    cv2.imshow('spatial filter', result)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()