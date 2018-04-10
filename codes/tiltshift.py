import numpy as np
import cv2


# x = linha da imagem onde o alpha deve ser calculado
# l1,l2 = linhas onde o valor de alpha eh 0.5
# d = forca do decaimento
def alpha(x, l1, l2, d):
    if d == 0:
        d += 0.0001
    return (1/2)*(np.tanh((x-l1)/d) - np.tanh((x-l2)/d))


def update_tiltshift():
    global mask, tiltshifted_img

    # update the mask and the processed image
    for x in range(0, img_height):
        alpha_x = alpha(x, l1, l2, decaimento)

        mask[x, :] = alpha_x
        tiltshifted_img[x, :] = cv2.addWeighted(img_copy[x, :], alpha_x, blurred_img[x, :], 1-alpha_x, 0)


def on_saturation_change(slider_pos):
    global img_copy
    # saturation multiplier ranging from 1.0 to 3.5 in increments of 0.5
    saturation_multiplier = (slider_pos * 2.5 / 5)+1
    # change the color space to HSV so we can easly modify the saturation
    img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    # split the planes
    (h, s, v) = cv2.split(img_copy)
    # multiply the saturation with the selected multiplier
    s = s * saturation_multiplier
    # nomalize values that fall outside the 0 to 255 range
    s = np.clip(s, 0, 255)
    # merge the planes again and convert the image back to BGR color space
    img_copy = cv2.merge([h, s, v])
    img_copy = cv2.cvtColor(img_copy.astype("uint8"), cv2.COLOR_HSV2BGR)
    # update the blurred image and the tilt shift
    blur_intensity_slider = cv2.getTrackbarPos('Blur intensity', 'Tiltshift')
    on_blur_change(blur_intensity_slider)
    update_tiltshift()


def on_focus_change(slider_pos):
    global center_line
    center_line = int(slider_pos * (img_height - 1) / 100)

    # update the window height with the new position of the focus center
    focus_height_slider = cv2.getTrackbarPos('Focus window height', 'Tiltshift')
    on_window_height_change(focus_height_slider)

    # update the image
    update_tiltshift()


def on_window_height_change(slider_pos):
    global l1, l2, mask

    window_height = int(slider_pos * (img_height - 1) / 100)
    l1 = center_line - int(window_height/2)
    l2 = center_line + int(window_height/2)

    # if l1 < 0:
    #     l1 = 0
    # if l2 >= height:
    #     l2 = height -1

    update_tiltshift()


def on_gradient_change(slider_pos):
    global decaimento
    decaimento = slider_pos
    update_tiltshift()


def on_blur_change(slider_pos):
    global blurred_img
    blurred_img = cv2.blur(img_copy, (5, 5))
    for i in range(0, slider_pos):
        blurred_img = cv2.blur(blurred_img, (5, 5))
    update_tiltshift()


# ====================================================================

# load the image, create a copy, a blurred copy and another copy to store the final result
img = cv2.imread('media/rome3.jpg')
img_copy = img.copy()
tiltshifted_img = img.copy()
blurred_img = cv2.blur(img, (5, 5))

img_height, img_width = img.shape[:2]

# create the mask and the negative mask with the same size as the image
mask = np.zeros(img.shape, dtype=np.float32)
mask_negative = mask.copy()

center_line = int(img_height / 2)
l1 = center_line - int(img_height / 4)
l2 = center_line + int(img_height / 4)
decaimento = 50

cv2.namedWindow('Tiltshift')
cv2.createTrackbar('Focus center line', 'Tiltshift', 50, 100, on_focus_change)
cv2.createTrackbar('Focus window height', 'Tiltshift', 50, 100, on_window_height_change)
cv2.createTrackbar('Blur gradient', 'Tiltshift', 50, 100, on_gradient_change)
cv2.createTrackbar('Blur intensity', 'Tiltshift', 0, 10, on_blur_change)
cv2.createTrackbar('Saturation', 'Tiltshift', 0, 5, on_saturation_change)

update_tiltshift()

while True:
    # display the image
    cv2.imshow('Tiltshift', tiltshifted_img)
    cv2.imshow('mask', mask)

    # wait for a key to be pressed
    key = cv2.waitKey(1) & 0xFF
    # if the esc key is pressed, close without saving
    if key == 27:
        break
    # if the space bar is pressed, save and close
    elif key == 32:
        cv2.imwrite('media/Tilt_shifted.jpg', tiltshifted_img)
        break

# destroy all windows
cv2.destroyAllWindows()

