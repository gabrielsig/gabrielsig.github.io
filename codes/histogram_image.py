import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


def calc_accumulated_histogram(histogram):
    accum_hist = []
    accum_hist.append(int(histogram[0]))
    for i in range(1, len(histogram)):
        accum_hist.append(accum_hist[i - 1] + int(histogram[i]))

    return accum_hist


def equalize(accum_hist, img):

    eq_img = img.copy()

    for x in range(0, height):
        for y in range(0, width):
            # the index in the histogram is the intensity value at (x, y)
            index = img[x, y]
            eq_img[x, y] = np.round(accum_hist[index] * 255 / (height * width))

    return eq_img


img = cv2.imread('media/farol.jpg', 0)

height, width = img.shape[:2]

# number of bins of the histogram
nbins = 256

# calculate the histogram
hist = cv2.calcHist([img], [0], None, [nbins], [0, 256])

# calculate the accumulated histogram
accum_hist = calc_accumulated_histogram(hist)

# equalize the image
eq_img = equalize(accum_hist, img)

# calculate the histogram of the new image
hist2 = cv2.calcHist([eq_img], [0], None, [nbins], [0, 256])

# calculate the accumulated histogram of the equalized image
accum_hist2 = calc_accumulated_histogram(hist2)

# normalize the accumulated histograms to show the graphs
accum_hist_norm = [x / (height*width) for x in accum_hist]
accum_hist_norm2 = [x / (height*width) for x in accum_hist2]


cv2.imshow('Original', img)
cv2.imshow('Equalized', eq_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.title('Accumulated histograms (normalized)')
# plt.plot(accum_hist_norm, 'k')
# plt.plot(accum_hist_norm2, 'b')
# plt.xlim([0, nbins])
# plt.show()
#
# plt.title('Histograms')
# plt.plot(hist, 'k')
# plt.plot(hist2, 'b')
# plt.xlim([0, nbins])
# plt.show()




