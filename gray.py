
import numpy as np
import cv2
import cv2ip
import matplotlib as plt 



ip = cv2ip.HistIP()
srcImg = ip.imread("girl.png")
F_BRG = ip.ImBGRA2BGR(srcImg) #有包含Alpha通道時才需做ImBGRA2BGR的動作
F_Gray = ip.ImBGR2GRAY(F_BRG)
F_Gray = cv2.equalizeHist(F_Gray)
GrayHist = ip.CalcGrayHist(F_Gray)
forceout = ip.ShowGrayHist(GrayHist)

ip.imshow("增強灰階img", F_Gray)
ip.imshow("灰階直方圖", forceout)

cv2.waitKey(0)
del ip
