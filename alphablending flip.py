
import cv2
import numpy as np
import cv2ip
import matplotlib.pyplot as plt

def Example_AlphaBlend(Fore_img,Back_img):
    list = [-1,1]
    ip = cv2ip.AlphaBlend()
    img = ip.imread(Fore_img)
    bool = 1

    while(bool):
        for i in list:
            img = cv2.flip(img,1) #翻转
            print("flip")

            for j in range(-100*i,100*i,10*i):

                H = np.float32([[1,0,j],[0,1,0]])
                img1 = cv2.warpAffine(img,H,(img.shape[1] ,img.shape[0])) #平移

                fore_alpha = ip.SplitAlpha(img1) #得到前景与alpha通道
                fore = np.float32(fore_alpha[0]) #前景
                alpha = np.float32(fore_alpha[1])/255 #alpha通道

                ImDim = alpha.shape #获取alpah通道的图片大小
                back = ip.imread(Back_img) #读取背景
                if (ImDim[0] != back.shape[0]) or (ImDim[1] != back.shape[1]):
                    back = cv2.resize(back,(ImDim[1], ImDim[0])) #使背景与前景大小相同

                back = np.float32(back) #转换为32位元
                out = ip.DoBlending(fore, back, alpha) #结合
                out = np.uint8(out) #输出8位元的图像
                ip.imshow("Out",out)
                ip.imshow("alpha", alpha)
                if cv2.waitKey(1) == ord('q'):
                    bool = 0

def main():
    Example_AlphaBlend("foreground.png","background.jpg")
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()