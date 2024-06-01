import numpy as np
import cv2 
from matplotlib import pyplot as plt

img = cv2.imread('girl.png')

kernel = np.full((5,5),-np.inf)/30
dst = cv2.filter2D(img,cv2.CV_16SC1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()