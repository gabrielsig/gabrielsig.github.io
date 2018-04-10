import numpy as np
import cv2

# ======================================================================================================================
# FUNCTIONS

def remove_border_objects(image):
    height, width = image.shape[:2]
    n_removed_objects = 0
    seed = [0, 0]

    for x in [0, height-1]:
        for y in range(0, width):
            if image[x, y] == 255:
                n_removed_objects += 1
                seed = [y, x]
                cv2.floodFill(image, None, tuple(seed), 0)

    for y in [0, width-1]:
        for x in range(0, height):
            if image[x, y] == 255:
                n_removed_objects += 1
                seed = [y, x]
                cv2.floodFill(image, None, tuple(seed), 0)

    # create a copy of the image to write the text and show
    image_copy = image.copy()

    # Write how many objets were removed from the image in the copy of the image
    cv2.putText(image_copy, str(n_removed_objects) + ' Objects removed',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Border removed', image_copy)

    return n_removed_objects


def count_objects(image):
    height, width = image.shape[:2]
    seed = [0, 0]
    n_objects = 0

    for x in range(0, height):
        for y in range(0, width):
            if img[x, y] == 255:
                n_objects += 1
                seed = [y, x]
                cv2.floodFill(img, None, tuple(seed), 100)

    # create a copy of the image
    image_copy = image.copy()

    # Write the object count
    cv2.putText(image_copy, str(n_objects) + ' Objects',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Object Count', image_copy)

    return n_objects

def count_holes(image):
    height, width = image.shape[:2]
    seed = [0, 0]
    n_holes = 0

    # create a copy of the image
    image_copy = image.copy()

    # floodfill the background of the image with 255, the the only pixels with value equal to 0 are inside an object
    cv2.floodFill(img, None, (0, 0), 255)

    for x in range(0, height):
        for y in range(0, width):
            # if we find a pixel with 0, we check if its neighbour is an object or the new 255 background
            if img[x, y] == 0:
                # if its the background, it means that we have already counted the object and that it had multiple holes
                if img[x-1, y] == 255:
                    seed = [y, x]
                    cv2.floodFill(img, None, tuple(seed), 255)
                # if its different then the beackground, it means that we have found a new object with a hole in it
                else:
                    n_holes += 1
                    seed = [y, x]
                    cv2.floodFill(img, None, tuple(seed), 255)
                    # we also fill the object
                    seed = [y, x-1]
                    cv2.floodFill(img, None, tuple(seed), 255)

    # floofill the background back to zero
    cv2.floodFill(img, None, (0, 0), 0)

    # subtract the image with only the objects without holes from the original copy previously created
    image_copy = image_copy - image

    # write the number of objects with holes
    cv2.putText(image_copy, str(n_holes) + ' Objects with holes',
                org=(5, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1)

    # show the copy of the image with the text
    cv2.imshow('Objects with holes', image_copy)

    return n_holes

# ======================================================================================================================
# PROGRAM LOGIC

# load the image
img = cv2.imread('media/bolhas.png', 0)

# show original image
cv2.imshow('Original', img)

# remove objects touching the border of the image
n_removed_objects = remove_border_objects(img)
print("Number of objects removed: ", n_removed_objects)

# label and count the total number of objects
n_objects = count_objects(img)
print("Number of objects: ", n_objects)

#count the number of objects with holes
n_holes = count_holes(img)
print("Number of objects with holes: ", n_holes)

#cv2.imshow("Labled image", img)
cv2.waitKey(0)