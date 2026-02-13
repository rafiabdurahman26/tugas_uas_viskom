import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
print("Current working directory:", os.getcwd())


img1 = cv2.imread('data/img1.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/img2.jpeg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Gambar tidak ditemukan!")
    exit()


sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

img_kp1 = cv2.drawKeypoints(img1, kp1, None)
img_kp2 = cv2.drawKeypoints(img2, kp2, None)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Keypoints Image 1")
plt.imshow(img_kp1, cmap='gray')

plt.subplot(1,2,2)
plt.title("Keypoints Image 2")
plt.imshow(img_kp2, cmap='gray')
plt.show()

# === FEATURE MATCHING (AMAN) ===
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# MATCH SELALU: des1 -> des2
matches = bf.match(des1, des2)

# SORT
matches = sorted(matches, key=lambda x: x.distance)

# === VISUALISASI (URUTAN KONSISTEN) ===
img_match = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:30],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12,6))
plt.title("Hasil Feature Matching SIFT")
plt.imshow(img_match)
plt.axis('off')
plt.show()
