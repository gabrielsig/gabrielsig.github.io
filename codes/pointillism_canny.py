import numpy as np
import cv2

# parameters that configure the behaviour of the pointillism
step = 5
jitter = 3
min_bg_radius = 3
max_bg_radius = 9
min_detail_radius = 1
max_detail_radius = 4

img = cv2.imread('media/pipa.jpg')
height, width = img.shape[:2]

# create a list to the points that will be used to apply the effect
# the number of points in each dimension is defined by the size of the image in that dimension divided by the step
x_range = np.arange(0, int(height/step))
y_range = np.arange(0, int(width/step))

for i in range(len(x_range)):
    x_range[i] = int(x_range[i] * step + step/2)
for j in range(len(y_range)):
    y_range[j] = int(y_range[j] * step + step/2)

points = np.full(img.shape, 255, dtype=np.uint8)

# apply the pointillism effect on the image with big points to serve as the background
np.random.shuffle(x_range)
for i in x_range:
    np.random.shuffle(y_range)
    for j in y_range:
        x = i + np.random.randint(-jitter, jitter+1)
        y = j + np.random.randint(-jitter, jitter+1)
        if x >= height:
            x = height-1
        if y >= width:
            y = width-1
        color = img[x, y].tolist()
        point_radius = np.random.randint(min_bg_radius, max_bg_radius)
        cv2.circle(points, (y, x), point_radius, color, -1, cv2.LINE_AA)


cv2.imshow('Pointillism before canny', points)

# create an array for the lower thresholds to be used by the canny algorithm
canny_min_thresh = np.arange(50, 301, 50)

# shuffle the threshold array
np.random.shuffle(canny_min_thresh)

# create a blurred copy of the image
blurred = cv2.blur(img, (5,5))

full_x_range = np.arange(0, height)
full_y_range = np.arange(0, width)


for threshold in canny_min_thresh:
    # create the image with the edges detected by the canny algorithm with the given thresholds
    edges = cv2.Canny(blurred, threshold, threshold*2)
    # shuffle the full_x_range so we go though the image randomly
    np.random.shuffle(full_x_range)
    for i in full_x_range:
        # shuffle the full_y_range
        np.random.shuffle(full_y_range)
        for j in full_y_range:
            if edges[i,j] == 255:
                x = i + np.random.randint(-jitter, jitter+1)
                y = j + np.random.randint(-jitter, jitter+1)
                if x >= height:
                    x = height-1
                if y >= width:
                    y = width-1
                color = img[x, y].tolist()
                point_radius = np.random.randint(min_detail_radius, max_detail_radius)
                cv2.circle(points, (y, x), point_radius, color, -1, cv2.LINE_AA)

cv2.imshow('Pointillism  after canny', points)
cv2.imwrite('pointillism.jpg', points)
cv2.waitKey()
