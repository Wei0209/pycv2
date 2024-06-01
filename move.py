
from cv2ip import AlphaBlend, BaseIP
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

ip=AlphaBlend

img = ip.imread("girl.png")
list = [-1,1]
cv2.imshow("before",img)
while(1):
    for i in list:
        # img = cv2.flip(img, 1)
        for j in range(-250*i,250*i,2*i):
            M = np.float32([[1,0,j],[0,1,0]])
            rows,cols = img.shape[:2]
            outt = cv2.warpAffine(img,M,(rows,cols))    
            cv2.imshow("Translated",outt)
            cv2.waitKey(1)
        img = cv2.flip(img, 1)

