import enum
from turtle import shape
from cv2 import WINDOW_AUTOSIZE, Scharr, accumulate, imshow
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from scipy.fft import dst
from skimage import exposure
from skimage.exposure import match_histograms
import cv2
import numpy as np
from enum import Enum, IntEnum


class BaseIP:
    ###kl
    # 影像讀取、儲存與顯示函數
    ###
    @staticmethod
    def imread(filename):
        return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #cv2.IMREAD_UNCHANGED 讀入格式不變，包含透明度channel (Alpha channel)
        #cv2.IMREAD_GRAYSCALE 以灰階讀入 
        # cv2.IMREAD_COLOR 以BGR格式讀入，此為預設值，忽略透明度channel 
        # 影像讀取

    @staticmethod
    def imwrite(filename,img):
        cv2.imwrite(filename,img)
        # 影像儲存
    
    @staticmethod
    def namedwindow(winname,flags=cv2.WINDOW_AUTOSIZE):
        cv2.namedWindow(winname,flags)
        #顯示函數
    #窗口大小不可變 cv2.namedWindow("img",cv2.WINDOW_AUTOSIZE) 
    #窗口大小自適應比例 cv2.namedWindow("img",cv2.WINDOW_FREERATIO)
    #窗口大小保持比例 cv2.namedWindow("img",cv2.WINDOW_KEEPRATIO)

    @staticmethod
    def imshow(winname,img):    
        cv2.imshow(winname,img)
    #當np.array為無符號8位元時可直接顯示出
    #當np.array為float格式時，須將數值正規化為0~1後才能顯示
    #delay<=0時，程式靜止
    #當delay>0時，函式會等待參數時間(ms)後，返回按鍵的ASCII碼，如果這段時間沒有按鍵按下，會返回-1 
   
class AlphaBlend(BaseIP):
    @staticmethod
    def SplitAlpha(SrcImg):
        alpha = cv2.split(SrcImg)
        foregroundChannels = cv2.merge([alpha[0],alpha[1],alpha[2]])
        alphaChannels = cv2.merge([alpha[3],alpha[3],alpha[3]])
        return foregroundChannels,alphaChannels

    @staticmethod
    def DoBlending(fore, back,alpha):
        foreground = cv2.multiply(alpha,fore)
        background = cv2.multiply(1.0 - alpha, back)
        out = cv2.add(foreground,background)
        return out 

class Ctype(enum.IntEnum): #channel type
    USE_RGB = 1
    USE_HSV = 2
    USE_YUV = 3
    
class HistIP(AlphaBlend, BaseIP):

    def _ImBGRA2BGR(self, SrcImg): 
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGRA2BGR)


    def _ImBGR2GRAY(seif, SrcImg): 
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)


    @staticmethod
    def CalcGrayHist(SrcGray): #灰度影像直方圖計算
        return cv2.calcHist([SrcGray], [0], None, [256], [0.0, 255.0])

    
    def ShowGrayHist(self, winname, GrayHist):
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(hist_w / histSize)
        histImg = np.zeros((hist_w, hist_h, 3), np.uint8)
        cv2. normalize(GrayHist, GrayHist, 0, hist_w, cv2.NORM_MINMAX)
        for col in range(1, 256):
            cv2.line(histImg, (bin_w * (col-1), hist_h - int(GrayHist[col-1])), (bin_w * col, hist_h -int(GrayHist[col])),[255, 255, 255],2, 8, 0)
        cv2.namedWindow(winname, cv2,WINDOW_AUTOSIZE)
        self.imshow(winname, histImg)


    @staticmethod
    def MonoEqualize(SrcGray):
        return cv2.equalizeHist(SrcGray)

    def _ImBGR2HSV(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2HSV)

    def _ImBGR2YUV(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_BGR2YUV)

    def _ImHSV2BGR(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_HSV2BGR)

    def _ImYUV2BGR(self, SrcImg):
        return cv2.cvtColor(SrcImg, cv2.COLOR_YUV2BGR)        
    

    @staticmethod
    def CalcColorHist(SrcColor):
        b_hist = cv2.calcHist([SrcColor], [0], None, [256], [0.0, 256.0])
        g_hist = cv2.calcHist([SrcColor], [1], None, [256], [0.0, 256.0])
        r_hist = cv2.calcHist([SrcColor], [2], None, [256], [0.0, 256.0])
        return b_hist, g_hist, r_hist

    def _calculateLUT(self, ref, src):
        LUT = np.zeros(256)
        val = 0
        for srcval in range(len(src)):
            val
            for refval in range(len(ref)):
                if ref[refval] >= src[srcval]:
                    val = refval
                    break 
            LUT[srcval] = val
        return LUT

    def calculatePDF(SrcGray):
        return 0

    
    def ShowColorHist(self, winname , ColorHist):
        b = ColorHist[0]
        g = ColorHist[1]
        r = ColorHist[2]
        histSize = 256
        hist_w = 512
        hist_h = 400
        bin_w = int(hist_w / histSize)
        histImg = np.zeros((hist_w, hist_h, 3), np.uint8)
        cv2.normalize(ColorHist[0], b, 0, hist_w, cv2.NORM_MINMAX)
        cv2.normalize(ColorHist[1], g, 0, hist_w, cv2.NORM_MINMAX)
        cv2.normalize(ColorHist[2], r, 0, hist_w, cv2.NORM_MINMAX)
        for col in range(1, 256):
            cv2.line(histImg, (bin_w * (col-1), hist_h - int(b[col-1])), (bin_w * col, hist_h -int(b[col])),[255, 255, 255],2, 8, 0)
            cv2.line(histImg, (bin_w * (col-1), hist_h - int(g[col-1])), (bin_w * col, hist_h -int(g[col])),[255, 255, 255],2, 8, 0)
            cv2.line(histImg, (bin_w * (col-1), hist_h - int(r[col-1])), (bin_w * col, hist_h -int(r[col])),[255, 255, 255],2, 8, 0)
        self.namedwindow(winname, cv2.WINDOW_AUTOSIZE)
        self.imshow(winname, histImg)

    def _CalculateCDF(self, src):
        PDF = cv2.calcHist([src], [0], None, [256], [0, 256], accumulate=False)
        CDF = PDF.cumsum()
        normalize_CDF = CDF/float(CDF.max())
        return normalize_CDF

    @staticmethod
    def ColorEqualize(SrcImg, type = Ctype):
        if type == 1:
            channel = cv2.split(SrcImg)
            channel1 = cv2.equalizeHist(channel[0])
            channel2 = cv2.equalizeHist(channel[1])
            channel3 = cv2.equalizeHist(channel[2])
            DstImg = cv2.merge((channel1,channel2,channel3))
            return DstImg
        if type == 2:
            SrcImg = cv2.COLOR_BGR2HSV(SrcImg)
            channel = cv2.split(SrcImg)
            channel_HSV = cv2.equalizeHist(channel[2])
            DstImg = cv2.merge((channel[0],channel[1],channel_HSV))
            DstImg = cv2.COLOR_HSV2BGR(DstImg)
            return DstImg
        if type == 3:
            SrcImg = cv2.COLOR_BGR2YUV
            channel = cv2.split(SrcImg)
            channel_YUV = cv2.equalizeHist(channel[0])
            DstImg = cv2.merge((channel_YUV,channel[1],channel[2]))
            DstImg = cv2.COLOR_YUV2BGR
            return DstImg

    def HistMatching(self, SrcImg, RefImg, type=Ctype):
        if type == 1:
            Src_Color = cv2.cvtColor(SrcImg,cv2.COLOR_BGRA2BGR)
            channelsrc = cv2.split(Src_Color)
            channelsrcB = channelsrc[0]
            channelsrcG = channelsrc[1]
            channelsrcR = channelsrc[2]
            CDF_SRC_B = self._CalculateCDF(channelsrcB)
            CDF_SRC_G = self._CalculateCDF(channelsrcG)
            CDF_SRC_R = self._CalculateCDF(channelsrcR)

            Ref_Color = cv2.cvtColor(SrcImg,cv2.COLOR_BGRA2BGR)
            channel_REF = cv2.split(Ref_Color)
            channel_REF_B = channel_REF[0]
            channel_REF_G = channel_REF[1]
            channel_REF_R = channel_REF[2]
            CDF_REF_B = self._CalculateCDF(channel_REF_B)
            CDF_REF_G = self._CalculateCDF(channel_REF_G)
            CDF_REF_R = self._CalculateCDF(channel_REF_R)

            LUT_B = self._calculateLUT(CDF_REF_B, CDF_SRC_B)
            LUT_G = self._calculateLUT(CDF_REF_G, CDF_SRC_G)
            LUT_R = self._calculateLUT(CDF_REF_R, CDF_SRC_R)
            B_mix = cv2.LUT(channelsrcB, LUT_B)
            G_mix = cv2.LUT(channelsrcG, LUT_G)
            R_mix = cv2.LUT(channelsrcR, LUT_R)

            DstImg = cv2.merge((B_mix, G_mix, R_mix)).astype(np.uint8)
            return DstImg

        if type == 2:
            Src_HSV = self._ImBGR2HSV(SrcImg)
            Ref_HSV = self._ImBGR2HSV(RefImg)

            channel_Src_HSV = cv2.split(Src_HSV)
            channel_Ref_HSV = cv2.split(Ref_HSV)

            Src_HSV_H = channel_Src_HSV[0].astype(float)
            Src_HSV_S = channel_Src_HSV[1].astype(float)
            Src_HSV_V = channel_Src_HSV[2]

            Ref_HSV_V = channel_Ref_HSV[2]

            cdf_Src_HSV_V = self._CalculateCDF(Src_HSV_V)
            cdf_Ref_HSV_V = self._CalculateCDF(Ref_HSV_V)

            V_LUT = self._calculateLUT(cdf_Ref_HSV_V, cdf_Src_HSV_V)
            result_HSV_LUT = cv2.LUT(Src_HSV_V, V_LUT)

            result = result_HSV_LUT.astype(np.uint8)
            result = self._CalculateCDF(result)

            DstImg = cv2.merge((Src_HSV_H,Src_HSV_S,result_HSV_LUT))
            DstImg = self._ImHSV2BGR(DstImg.astype(np.uint8))
            return DstImg

        if type == 3:
            Src_YUV = self._ImBGR2YUV(SrcImg)
            Ref_YUV = self._ImBGR2YUV(RefImg)

            channel_Src_YUV = cv2.split(Src_YUV)
            channel_Ref_YUV = cv2.split(Ref_YUV)

            channel_Src_YUV_Y = channel_Src_YUV[0]
            channel_Src_YUV_U = channel_Src_YUV[1].astype(float)
            channel_Src_YUV_V = channel_Src_YUV[2].astype(float)

            channel_Ref_YUV_Y = channel_Ref_YUV[0]
            channel_Ref_YUV_U = channel_Ref_YUV[1]
            channel_Ref_YUV_V = channel_Ref_YUV[2]

            cdf_channel_Src_YUV_Y = self._CalculateCDF(channel_Src_YUV_Y)
            cdf_channel_Ref_YUV_Y = self._CalculateCDF(channel_Ref_YUV_Y)

            YUV_LUT = self._calculateLUT(cdf_channel_Ref_YUV_Y, cdf_channel_Src_YUV_Y)
            result_YUV_LUT = cv2.LUT(channel_Src_YUV_Y, YUV_LUT)

            result = result_YUV_LUT.astype(np.uint8)
            result = self._CalculateCDF(result)

            DstImg = cv2.merge((result_YUV_LUT,channel_Src_YUV_U,channel_Src_YUV_V))
            DstImg = self._ImYUV2BGR(DstImg.astype(np.uint8))
            return DstImg


class SMType(enum.IntEnum): #SmoothType.
    BLUR = 1
    BOX = 2
    GAUSSIAN = 3
    MEDIAN = 4
    BILATERAL = 5

    #Weight = 1.0/(MASK_WH*MASK_WH)
    #kernel = np.full((MASK_WH, MASK_WH), fWeight, np.float)

class EdgeType(enum.IntEnum): #邊緣偵測
    SOBEL = 1
    CANNY = 2
    SCHARR = 3
    LAPLACE = 4
    COLOR_SOBEL = 5

class CV2_SHARPENING_TYPE(enum.IntEnum):
     LAPLACE_TYPE1 = 1
     LAPLACE_TYPE2 = 2
     SECOND_ORDER_LOG = 3
     UNSHARP_MASK = 4


class ConvIP(HistIP): 
    #平均濾波 Averaging：使用 opencv 的 cv2.blur 或 cv2.boxFilter
    #高斯濾波 Gaussian Filtering：使用 opencv 的 cv2.GaussianBlur
    #中值濾波 Median Filtering：使用 opencv 的 cv2.medianBlur
    #雙邊濾波 Bilateral Filtering：使用 opencv 的 cv2.bilateralFilter

    def Smooth(self, SrcImg, ksize, type = SMType):
        if type == 1:
            return cv2.blur(SrcImg, (ksize, ksize))
        if type == 2:
            return cv2.boxFilter(SrcImg, -1, (ksize, ksize))
        if type == 3:
            return cv2.GaussianBlur(SrcImg, (ksize, ksize), 0)
        if type == 4:
            return cv2.medianBlur(SrcImg, ksize)
        if type == 5:
            return cv2.bilateralFilter(SrcImg, ksize, 85, 10) 
        #sigmaColor越大，则图片中的噪点会越少，但是图片也会越均一模糊
        #sigmaSpace一般越小越好，但是也要根据实际考虑


    def _Sobel(self, SrcImg):
        grayImg = self._ImBGR2GRAY(SrcImg)
        srcX = cv2.Sobel(grayImg,cv2.CV_16S, 1, 0, 3, 1) #深度可變看看
        srcX = cv2.convertScaleAbs(srcX, 1)
        srcY = cv2.Sobel(SrcImg, cv2.CV_16S, 0, 1, 3, 1)
        srcY = cv2.convertScaleAbs(srcY, 1)
        return cv2.addWeighted(srcX, 0.5, srcY, 0.5)
    
    def _colorSobel(self, SrcImg):
        x = cv2.sobel(SrcImg, cv2.CV_64F, 1, 0, ksize = 3 )
        y = cv2.sobel(SrcImg, cv2.CV_64F, 0, 1, ksize = 3)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        return result

    def _Laplace(self, SrcImg):
        grayImg = self._ImBGR2GRAY(SrcImg)
        gray16s = cv2.Laplacian(grayImg, cv2.CV_16S, 3)
        return cv2.convertScaleAbs(gray16s, 1)

    def _Canny(self, SrcImg):        
        grayImg = self._ImBGR2GRAY(SrcImg)
        return cv2.Canny(grayImg, 50, 150)
    
    def _Scharr(self, SrcImg):
        grayImg = self._ImBGR2GRAY(SrcImg)
        resultX = cv2.Scharr(grayImg, cv2.CV_16S, 1, 0)
        resultX = cv2.convertScaleAbs(resultX, 1.0)

        resultY = cv2.Scharr(grayImg, cv2.CV_16S, 0, 1)
        resultY = cv2.convertScaleAbs(resultY, 1.0)
        return cv2.addWeighted(resultX, 0.5, resultY, 0.5, 0)

    def Detect(self ,SrcImg, Type = EdgeType):
        if Type == 1:
            return self._Sobel(SrcImg)

        if Type == 2:
            return self._Canny(SrcImg)

        if Type == 3:
            return self._Scharr(SrcImg)

        if Type == 4:
            return self._Laplace(SrcImg)

        if Type == 5:
            grayImg = self._ImBGR2GRAY(SrcImg)
            x = cv2.Sobel(grayImg, cv2.CV_64F, 1, 0, 3, 1)
            y = cv2.Sobel(grayImg, cv2.CV_64F, 0, 1, 3, 1)
            absx = cv2.convertScaleAbs(x)
            absy = cv2.convertScaleAbs(y)
            Out = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
            return Out

    def Conv2D(self, SrcImg, kernal):
        return cv2.filter2D(SrcImg, -1 ,kernal) #ddepth 深度
    
    def _SharpLaplaceType1(self, SrcImg, gainLanda):
        kernal = np.zeros([3, 3])
        kernal[0][1] = -1*gainLanda
        kernal[1][0] = -1*gainLanda
        kernal[1][1] = 4*gainLanda + 1
        kernal[1][2] = -1*gainLanda
        kernal[2][1] = -1*gainLanda
        hsvImg = self._ImBGR2HSV(SrcImg)
        dst_planes = cv2.split(hsvImg)
        h = dst_planes[0]
        s = dst_planes[1]
        v = dst_planes[2]
        v = self.Conv2D(v, kernal)
        hsvImg = cv2.merge((h, s, v)).astype(np.uint8)
        result = self._ImHSV2BGR(hsvImg)
        return result
        
    def ImSharpening(self, SrcImg, gainLanda, SpType=CV2_SHARPENING_TYPE.LAPLACE_TYPE1, SmType=SMType.BILATERAL):
        if SpType == 1:
            kernal = np.zeros([3, 3])
            kernal[0][1] = -1*gainLanda
            kernal[1][0] = -1*gainLanda
            kernal[1][1] = 4*gainLanda + 1
            kernal[1][2] = -1*gainLanda
            kernal[2][1] = -1*gainLanda
            hsvImg = self._ImBGR2HSV(SrcImg)
            dst_planes = cv2.split(hsvImg)
            h = dst_planes[0]
            s = dst_planes[1]
            v = dst_planes[2]
            v = self.Conv2D(v, kernal)
            hsvImg = cv2.merge((h, s, v)).astype(np.uint8)
            result = self._ImHSV2BGR(hsvImg)
            return result
        if SpType == 2:
            kernal = np.zeros([3, 3])
            for i in range(3):
                for j in range(3):
                    kernal[i][j] = -gainLanda
                    if i == j and j == 1:
                        kernal[i][j] = 8*gainLanda + 1
            hsvImg = self._ImBGR2HSV(SrcImg)
            dst_planes = cv2.split(hsvImg)
            h = dst_planes[0].astype(float)
            s = dst_planes[1].astype(float)
            v = dst_planes[2]
            v = self.Conv2D(v, kernal).astype(float)
            hsvImg = cv2.merge((h, s, v)).astype(np.uint8)
            result2 = self._ImHSV2BGR(SrcImg)
            return result2
        if SpType == 3:
            matrix_elements = np.zeros([5, 5])
            matrix_elements[0][2] = 1
            matrix_elements[1][1] = 1
            matrix_elements[1][2] = 2
            matrix_elements[1][3] = 1
            matrix_elements[2][0] = 1
            matrix_elements[2][1] = 2
            matrix_elements[2][2] = -17
            matrix_elements[2][3] = 2
            matrix_elements[2][4] = 1
            matrix_elements[3][1] = 1
            matrix_elements[3][2] = 2
            matrix_elements[3][3] = 1
            matrix_elements[4][2] = 1
            kernal = gainLanda * matrix_elements
            hsvImg = self._ImBGR2HSV(SrcImg)
            dst_planes = cv2.split(hsvImg)
            h = dst_planes[0].astype(float)
            s = dst_planes[1].astype(float)
            v = dst_planes[2]
            v = self.Conv2D(v, kernal).astype(float)
            hsvImg = cv2.merge((h, s, v)).astype(np.uint8)
            result3 = self._ImHSV2BGR(SrcImg)
            return result3
        
        if SpType == 4:
            hsvImg = self._ImBGR2HSV(SrcImg)
            dst_planes = cv2.split(hsvImg)
            h = dst_planes[0].astype(float)
            s = dst_planes[1].astype(float)
            v = dst_planes[2]
            Coarse = self.Smooth(v, 3, SmType.BILATERAL)
            Fine = gainLanda*(v - Coarse)
            v += Fine
            v = v.astype(float)
            hsvImg = cv2.merge((h,s,v)).astype(np.uint8)
            result4 = self._ImHSV2BGR(hsvImg)
            return result4



    

