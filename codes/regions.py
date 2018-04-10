import numpy as np
import cv2

def on_mouse_select(event, x, y, flags, param):
    global p1, p2, p1_set, p2_set, img_width, img_height

    # if the point is outside of the image, set its coordinates to the border
    if x > img_width:
        x = img_width
    if y > img_height:
        y = img_height

    # assign the coordinates to the points and set the flags
    if event == cv2.EVENT_LBUTTONDOWN:
        p1 = [y, x]
        p1_set = True
    elif event == cv2.EVENT_LBUTTONUP:
        p2 = [y, x]
        p2_set = True

# read image
img = cv2.imread('media/hermes2.jpeg')

# get the image height and width
img_height, img_width = img.shape[:2]

# create the points on the origin and on the farther edge of the image
p1 = [0, 0]
p2 = [img_height, img_width]

# flags to determine if either one of the points is set
p1_set = False
p2_set = False

# create the image and the mouse callback
cv2.namedWindow('Regions')
cv2.setMouseCallback('Regions', on_mouse_select)

print('Image size: x = {}, y = {}'.format(img_height,img_width))

while True:
    if p1_set and p2_set:
        # img = cv2.imread('media/hermes2.jpeg')
        # Apply the negative in the area composed by the 2 points provided by the user
        for i in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
            for j in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
                img[i, j] = 255 - img[i, j]
        # reset the flags
        p1_set = False
        p2_set = False

    # show the image
    cv2.imshow('Regions', img)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close the program
    # if the space bar is pressed, reset the image
    if key == 27:
        break
    elif key == 32:
        img = cv2.imread('media/hermes2.jpeg')

cv2.destroyAllWindows()
