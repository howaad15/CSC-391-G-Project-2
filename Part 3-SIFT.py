import cv2
import numpy as np
import skimage as ski

img = cv2.imread('G1.png')

img2 = cv2.imread('G2.png')

img3 = cv2.imread('G1.png')


sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

img3 = cv2.drawMatchesKnn(img, kp1, img2, kp2, matches[:15], None, flags=2)

cv2.imshow('', img3)
cv2.imwrite('GSIFT.jpeg', img3)
cv2.waitKey(0)



