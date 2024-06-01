from cv2ip import HistIP
import cv2

ip = HistIP

src = ip.imread("girl.png")
ref = ip.imread("background.jpg")
SrcImg = cv2.resize(src, (720,480))
RefImg = cv2.resize(ref, (720,480))
out = ip.HistMatching(SrcImg, RefImg, type=1)

ip.imshow('src', SrcImg)
ip.imshow('ref', RefImg)
ip.imshow('result', out)

cv2.waitKey(0)