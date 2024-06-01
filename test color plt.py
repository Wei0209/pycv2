
from cv2ip import AlphaBlend, BaseIP
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('girl.png')
img1 = cv2.cvtColor(img , cv2.COLOR_BGRA2RGB)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# cv2.imshow("color", img)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(img1)
plt.subplot(2, 1, 2)
# 畫出 RGB 三種顏色的分佈圖
color = ('b','g','r')
for i, col in enumerate(color):
  histr = cv2.calcHist([img1],[i],None,[256],[0, 256])
  plt.plot(histr, color = col)
  plt.xlim([0, 256])
plt.show()
# cv2.waitKey(0)

#https://blog.gtwang.org/programming/python-opencv-matplotlib-plot-histogram-tutorial/
#https://www.gushiciku.cn/pl/gSfX/zh-tw
