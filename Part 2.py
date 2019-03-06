import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max

#img = cv2.imread('cactus2.jpeg')
#img3=img


#cv2.imshow('', img)
#cv2.imshow('', img3)
#cv2.waitKey(0)

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    #frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(100, 3, .1, 30, 1.6)

    (kps, descs) = sift.detectAndCompute(gray, None)

    img = cv2.drawKeypoints(img, kps, None, color=(0, 225, 0), flags=cv2.DrawMatchesFlags_DEFAULT)

    cv2.imshow('SIFT', img)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img_name, edge)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()