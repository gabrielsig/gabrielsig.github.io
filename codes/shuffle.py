import numpy as np
import cv2

img = cv2.imread('media/hermes2.jpeg')

height, width = img.shape[:2]

print('Image size: x = {} | y = {}'.format(height, width))

n_rows = int(input("Type the number of rows: "))
n_cols = int(input("Type the number of columns: "))

if height % n_rows != 0:
    print("[ERROR]: image height is not divisible by the number provided")
    exit()
elif width % n_cols != 0:
    print("[ERROR]: image height is not divisible by the number provided")
    exit()

x_size = int(height / n_rows)
y_size = int(width / n_cols)

s = []

for i in range(0, height, x_size):
    for j in range(0, width, y_size):
        s.append(img[i:(i+x_size), j:(j+y_size)])
        #print('start: {} {} | finish {} {}'.format(x,y,x+x_size,y+y_size))

np.random.shuffle(s)

# create the rows
rows = []

for i in range(0, n_rows):
    print(i * n_cols, (i + 1) * n_cols)
    rows.append(np.concatenate(tuple(s[i * n_cols:(i + 1) * n_cols]), 1))

new_img = np.concatenate(tuple(rows[:]))

cv2.imshow("Original Image", img)
cv2.imshow("Shuffle {} x {}".format(n_rows,n_cols), new_img)
cv2.waitKey(0)

cv2.destroyAllWindows()