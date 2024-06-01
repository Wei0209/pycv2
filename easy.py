
from cv2ip import AlphaBlend, BaseIP
import numpy as np
from PIL import Image
import cv2

ip = AlphaBlend
img1 = ip.imread("background.jpg") #(1248, 2220, 3)
img2 = ip.imread("foreground.png") #(1172, 1500, 4)


dst_planes = ip.SplitAlpha(img2)#[fore,alpha]
foreground = np.float32(dst_planes[0])
alpha = np.float32(dst_planes[1])/255

Dim = alpha.shape
if Dim[1] != img1.shape[1] or Dim[0] != alpha.shape[0]:
    img1 = cv2.resize(img1,(Dim[1],Dim[0]))
    
back = np.float32(img1) #转换为32位元
      
out = ip.DoBlending(foreground, back, alpha)
out = np.uint8(out)  
cv2.imshow("alpha",alpha)
cv2.imshow("Translated",out)
cv2.waitKey(0)