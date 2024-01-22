import cv2
import numpy as np
from skimage.filters import sobel
from math import sqrt

def nothing(x):
    pass

# Loading image
image = cv2.imread("sequence_01/frames/0001.bmp")

# Converting BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

height, width = image.shape[:2]

print("Height {}, Width {}".format(height, width))


hsv[:, :, 1] = np.zeros([height, width])
hsv[:, :, 2] = np.zeros([height, width])


for lines in range(height) :
    for columns in range(width) :
        if(12< hsv[lines, columns, 0] and hsv[lines, columns, 0] < 40) :
            hsv[lines, columns, :] = 255
        else :
            hsv[lines, columns, :] = 0


# Apply an opening operation
kernel = np.ones((20, 20), np.uint8)
hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)

# Apply a closing operation
#kernel = np.ones((10, 10), np.uint8)
hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)

hsv = sobel(hsv)

R = 500

vote = np.zeros((int(round(height, -1)/10), int(round(width, -1)/10), R))

for lines in range(0, round(height, -1), 5) :
    for columns in range(0, round(width, -1), 5) :
        if hsv[lines, columns, 0] != 0 :
            for x0 in range(0, round(width, -1), 10) :
                for y0 in range(0, round(height, -1), 10) :
                    r = round(sqrt((columns - x0)**2 + (lines - y0)**2))
                    if r < 500 :
                        vote[int(y0/10), int(x0/10), r-1] = vote[int(y0/10), int(x0/10), r-1] + 1


class BreakLoop(Exception):
    pass

value = np.max(vote)
try:
    for y in range(len(vote[:, 0, 0])) :
        for x in range(len(vote[0, :, 0])) :
            for r in range(len(vote[0, 0, :])) :
                if vote[y, x, r] == value :
                    raise BreakLoop

except BreakLoop:
    print("Sortie de toutes les boucles.")

while True:
# Showing images
    cv2.circle(image, (10*x, 10*y), r, (255, 0, 0), 2)
    cv2.imshow("image", image)

    key = cv2.waitKey(1) & 0xFF

    # Stopping criteria
    if key == ord("q"):
        break