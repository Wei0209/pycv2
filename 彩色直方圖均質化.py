
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv2ip import AlphaBlend, HistIP

ip = AlphaBlend
op = HistIP
img = ip.imread('foreground.png')
plt.figure(1)
plt.subplot(2, 1, 1)
plt.imshow(img)
plt.subplot(2, 1, 2)

color = ('b','g','r')
for i, col in enumerate(color):
  histr = cv2.calcHist([img],[i],None,[256],[0, 256])
  plt.plot(histr, color = col)
  plt.xlim([0, 256])
plt.show()

