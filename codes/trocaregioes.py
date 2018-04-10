import numpy as np
import cv2

img = cv2.imread('media/hermes2.jpeg')

height, width = img.shape[:2]

x_divider = 2
y_divider = 2

x_size = int(height/x_divider)
y_size = int(width/y_divider)

s = []

s.append(img[0:x_size, 0:y_size])
s.append(img[0:x_size, y_size: ])
s.append(img[x_size: , 0:y_size])
s.append(img[x_size: , y_size: ])

#np.random.shuffle(s)

new_img_top = np.concatenate((s[3], s[2]), 1)
new_img_bot = np.concatenate((s[1], s[0]), 1)

new_img = np.concatenate((new_img_top, new_img_bot))

print(np.array(s).shape)

cv2.imshow("Original Image", img)
cv2.imshow("New Image", new_img)
cv2.waitKey(0)

cv2.destroyAllWindows()