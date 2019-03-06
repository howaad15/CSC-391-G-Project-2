import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max

img = cv2.imread('G1.png')
img2 = cv2.imread('G2.png')
img3 = cv2.imread('G1.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
dst2=np.array(dst)
# threshold for optimal value
img[dst > 0.01 * dst.max()] = [0, 0, 255]

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
dst2 = cv2.cornerHarris(gray2, 2, 3, 0.04)
dst2 = cv2.dilate(dst2, None)
dst2=np.array(dst2)
# threshold for optimal value
img2[dst2 > 0.01 * dst2.max()] = [0, 0, 255]

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(dst, dst2)

img3 = cv2.imread('G1.jpeg')
img3 = cv2.drawMatches(img, dst, img2, dst2, matches, None, flags=2)

cv2.imshow('', img3)
cv2.waitKey(0)