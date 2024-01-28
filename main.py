import cv2
import numpy as np
from skimage.filters import sobel
from math import sqrt

'''# Loading image
image = cv2.imread("sequence_02/frames/0001.bmp")'''

# Charger la vidéo
video_path = "sequence_02/seq2.mkv"
cap = cv2.VideoCapture(video_path)

# Vérifier si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Erreur lors de l'ouverture de la vidéo.")
    exit()

# Lire la première image pour obtenir les dimensions
ret, image = cap.read()
if not ret:
    print("Erreur lors de la lecture de la première image de la vidéo.")
    exit()

# Converting BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

height, width = image.shape[:2]

print("Height {}, Width {}".format(height, width))


hsv[:, :, 1] = np.zeros([height, width])
hsv[:, :, 2] = np.zeros([height, width])


for lines in range(height) :
    for columns in range(width) :
        if(12 < hsv[lines, columns, 0] and hsv[lines, columns, 0] < 40) :
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

R = 300

vote = np.zeros((int(round(height, -1)/10), int(round(width, -1)/10), R))

for lines in range(0, round(height), 5) :
    for columns in range(0, round(width), 5) :
        if hsv[lines, columns, 0] != 0 :
            for x0 in range(0, round(width, -1), 10) :
                for y0 in range(0, round(height, -1), 10) :
                    r = round(sqrt((columns - x0)**2 + (lines - y0)**2))
                    if r < R :
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

    # Lire une nouvelle image
    ret, image = cap.read()

    # Vérifier si la lecture de l'image a réussi
    if not ret:
        print("Fin de la vidéo.")
        break

    #Define the new area to analyze
    if(10*x - (r+40) > 0) :
        bornx_inf = 10*x - (r+40)
    else :
        bornx_inf = 0

    if(10*x + (r+40) < width) :
        bornx_supp = 10*x + (r+40)
    else :
        bornx_supp = width

    if(10*y - (r+40) > 0) :
        borny_inf = 10*y - (r+40)
    else :
        borny_inf = 0

    if(10*y + (r+40) < height) :
        borny_supp = 10*y + (r+40)
    else :
        borny_supp = height

    #-----------------------------------#

    if(10*x - 40 > 0) :
        centerx_inf = 10*x - 40
    else :
        centerx_inf = 0

    if(10*x + 40 < width) :
        centerx_supp = 10*x + 40
    else :
        centerx_supp = width

    if(10*y - 40 > 0) :
        centery_inf = 10*y - 40
    else :
        centery_inf = 0

    if(10*y + 40 < height) :
        centery_supp = 10*y + 40
    else :
        centery_supp = height

    if r - 40 > 0 :
        ray_inf = r - 40
    else :
        ray_inf = r 
    if r + 40 < R :
        ray_supp = r + 40
    else :
        ray_supp = R

    #hsv = image[bornx_inf : bornx_supp, borny_inf : borny_supp, :]
    # Converting BGR to HSV
    hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :] = cv2.cvtColor(image[borny_inf : borny_supp, bornx_inf : bornx_supp, :], cv2.COLOR_BGR2HSV)

    #height, width = image.shape[:2]

    #print("Height {}, Width {}".format(height, width))

    hsv[:, :, 1] = np.zeros([height, width])
    hsv[:, :, 2] = np.zeros([height, width])


    
    for lines in range(borny_inf, borny_supp) :
        for columns in range(bornx_inf, bornx_supp) :
            if(15 < hsv[lines, columns, 0] and hsv[lines, columns, 0] < 30) :
                hsv[lines, columns, :] = 255
            else :
                hsv[lines, columns, :] = 0


    
    # Apply an opening operation
    kernel = np.ones((10, 10), np.uint8)
    hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :] = cv2.morphologyEx(hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :], cv2.MORPH_OPEN, kernel)

    # Apply a closing operation
    #kernel = np.ones((10, 10), np.uint8)
    hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :] = cv2.morphologyEx(hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :], cv2.MORPH_CLOSE, kernel)

    hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :] = sobel(hsv[borny_inf : borny_supp, bornx_inf : bornx_supp, :])

    vote = vote * 0#np.zeros((int(round(height, -1)/10), int(round(width, -1)/10), R))

    ray = 0
    for lines in range(borny_inf, borny_supp, 10) :
        for columns in range(bornx_inf, bornx_supp, 10) :
            if hsv[lines, columns, 0] != 0 :
                for x0 in range(centerx_inf, centerx_supp) :
                    for y0 in range(centery_inf, centery_supp) :
                        ray = round(sqrt((columns - x0)**2 + (lines - y0)**2))
                        if ray_inf < ray < ray_supp :
                            vote[int(y0/10), int(x0/10), ray-1] = vote[int(y0/10), int(x0/10), ray-1] + 1

    value = np.max(vote)
    try:
        for y in range(int(round(centery_inf, -1)/10), int(round(centery_supp, -1)/10)) :
            for x in range(int(round(centerx_inf, -1)/10), int(round(centerx_supp, -1)/10)) :
                for r in range(ray_inf, ray_supp) :
                    if vote[y, x, r] == value :
                        raise BreakLoop

    except BreakLoop:
        print("Sortie de toutes les boucles.")

    key = cv2.waitKey(1) & 0xFF

    # Stopping criteria
    if key == ord("q"):
        break