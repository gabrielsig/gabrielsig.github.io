import numpy as np
import cv2


# ==============================================================
# FUNCTIONS

def flip_quadrants(img):
    # get the height and width of the image
    height, width = img.shape[:2]

    height = height & -2
    width = width & -2

    img = img[:height, :width]

    # size of each quadrant of the image
    x_size = int(height/2)
    y_size = int(width/2)

    # crate quadrants based on the image
    q1 = img[0:x_size, 0:y_size]
    q2 = img[0:x_size, y_size: ]
    q3 = img[x_size: , 0:y_size]
    q4 = img[x_size: , y_size: ]

    # concatenate the top and bottom half
    new_img_top = np.concatenate((q4, q3), 1)
    new_img_bot = np.concatenate((q2, q1), 1)

    # concatenate again
    new_img = np.concatenate((new_img_top, new_img_bot))

    return new_img


def get_filter_mask(shape):
    dft_height, dft_width, _ = shape

    filter_mask = np.zeros(shape, dtype=np.float32)

    for x in range(dft_height):
        for y in range(dft_width):
            # get the distance of the current pixel to the center
            distance = np.sqrt((x - dft_height / 2) ** 2 + (y - dft_width / 2) ** 2)
            # avoid division by zero
            distance += 0.0001

            filter_mask[x, y] = (g_h - g_l)*(1 - np.exp(-c*(distance**2)/(radius**2))) + g_l


    # get the magnitude spectrum
    magnitude_spectrum = 20*np.log(cv2.magnitude(filter_mask[:,:,0],filter_mask[:,:,1]))
    # normalize it
    magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 1, norm_type=cv2.NORM_MINMAX)
    cv2.imshow('mask', magnitude_normalized)

    return filter_mask


# ==================================================================
# PROGRAM LOGIC

# filter parameters
g_h = 12
g_l = 0.8
c = 1
radius = 400

# load the image as grey scale and get its dimensions
img = cv2.imread('test5.jpg', 0)
height, width = img.shape[:2]

# get the optimal dft size based on the image size
dft_m = cv2.getOptimalDFTSize(height)
dft_n = cv2.getOptimalDFTSize(width)

# copy the image with padding to another image
padded_img = cv2.copyMakeBorder(img, 0, dft_m-height, 0, dft_n-width, cv2.BORDER_CONSTANT, value=0)

# create a coppy of the padded image as floating point
float_img = np.float32(padded_img)

# add 1 to the image to avoid log of zero
float_img = float_img + 1

# apply the natural log function to the image
log_img = np.log(float_img)

# calculate the dft of the image with the output being a complex matrix
dft = cv2.dft(log_img, flags=cv2.DFT_COMPLEX_OUTPUT)

# shift the dft quadrants
shifted_dft = flip_quadrants(dft)

# get the magnitude spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(shifted_dft[:,:,0],shifted_dft[:,:,1]))

# normalize it
magnitude_normalized = cv2.normalize(magnitude_spectrum, None, 0, 1, norm_type=cv2.NORM_MINMAX)

# generate a filter mask with the same shape as the dft
filter_mask = get_filter_mask(dft.shape)

# apply the filter to the shifted dft
shifted_dft_filtered = cv2.mulSpectrums(shifted_dft,filter_mask,0)
#shifted_dft_filtered = shifted_dft * filter_mask

# shift the quadrants back
dft_filtered = flip_quadrants(shifted_dft_filtered)

# apply the inverse dft to get the new image
new_img = cv2.idft(dft_filtered)
new_img = cv2.magnitude(new_img[:,:,0],new_img[:,:,1])
# new_img = new_img[:,:,0]

new_img = cv2.normalize(new_img, None, 0, 1, norm_type=cv2.NORM_MINMAX)

# apply the exponential funtion to the image
new_img = np.exp(new_img)

# subtract 1
new_img = new_img - 1

# normalize
new_img = cv2.normalize(new_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)

# convert back to uint8
new_img = np.uint8(new_img)

cv2.imshow('Original', img)
cv2.imshow('Filtered', new_img)
cv2.waitKey()
