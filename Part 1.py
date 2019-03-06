import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max

# Capturing video

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

    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 25, 0.04)
    dst = cv2.dilate(dst, None)
    # threshold for optimal value
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    #edge = cv2.Canny(frame, 25, 105)

    # cv2.imshow('Canny Edge', edge)
    cv2.imshow('Harris Corner', img)

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