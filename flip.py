

from cv2ip import AlphaBlend, BaseIP
import numpy as np
from PIL import Image
import cv2

ip = AlphaBlend

image_flip = cv2.imread("girl.png")

cv2.imshow("before", image_flip)

image_LRflip = cv2.flip(image_flip, 1) # 左右翻轉

cv2.imshow("after", image_LRflip)
cv2.waitKey(0)