
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2ip import AlphaBlend, BaseIP

img = cv2.imread("girl.png")
img1 = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray = cv2.equalizeHist(gray)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(gray)
plt.subplot(2, 1, 2)

plt.hist(gray.ravel(), 256, [0, 256]) #revel 將所有的像素資料轉為一維的陣列
plt.show()