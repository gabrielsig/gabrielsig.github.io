import numpy as np
import cv2

def sort_by_hue(color_array, iteration):
    # creante an array to store the hues that we will obtain from the colors (kmenas centers)
    hue = np.zeros(n_clusters)

    # calculate the hue for each member of the color array
    for i in range(n_clusters):
        blue = float(color_array[i,0])
        green = float(color_array[i,1])
        red = float(color_array[i,2])

        # get the min and max color
        min_color = min(min(red, green), blue)
        max_color = max(max(red, green), blue)


        # calculate the hue
        if min == max:
            hue[i] = 0
            continue

        if max_color == red:
            hue[i] = (green - blue)/(max_color - min_color)
        elif max_color == green:
            hue[i] = 2 + (blue - red)/(max_color - min_color)
        else:
            hue[i] = 4 +(red - green)/(max_color - min_color)

        hue[i] = hue[i] * 60
        if hue[i] < 0:
            hue[i] += 360


    # get the indexes in the sequence that would sort the array
    index = np.argsort(hue)
    # reshape the hue array in the correct order based on the above indexes
    hue = hue[index]
    print('Colors {}:'.format(iteration), hue)
    # reshape the array of colors (k means centers) so we can build the color palette
    color_array = color_array[index]

    return color_array

#===============================================================================

# parameters to control the behaviour
n_clusters = 6
n_rounds = 5
n_iterations = 10

img = cv2.imread('flower.jpg')

# reshape the image to get an array to store the 3 color components of each pixel in a row
samples = img.reshape((-1, 3))
# convert to float
samples = np.float32(samples)

# apply the k means 'n_iterations' times to get the different images
for iteration in range(n_iterations):
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.0001)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centers = cv2.kmeans(samples, n_clusters, None, stop_criteria, n_rounds, flags)

    # the 'centers' variable store the colors obtained by the clustering process
    centers = np.uint8(centers)
    # create a new image applying the colors
    labeled_img = centers[labels.flatten()]
    labeled_img = labeled_img.reshape((img.shape))

    # sort the centers (color array) by the hue so we can buil the color palette
    centers = sort_by_hue(centers, iteration)

    # create the image to store  the colors
    colors = np.zeros((100, n_clusters*100, 3), dtype=np.uint8)

    # draw a rectangle of each color
    for i in range(n_clusters):
        top_left = (100*i, 0)
        bottom_right = (100*(i+1), 100)
        curr_color = centers[i].tolist()
        cv2.rectangle(colors, top_left, bottom_right, curr_color, -1)

    cv2.imshow('Color palette {}'.format(iteration),colors)
    cv2.imwrite('iteration{}.jpg'.format(iteration), labeled_img)

cv2.waitKey()
