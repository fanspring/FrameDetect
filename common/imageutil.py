# -*- coding: utf-8 -*-  

'''
图片读取、存储通用类
'''


from numpy import *
import numpy as np
from PIL import Image,ImageStat
import os
import copy
import util
#import cv2

class ImageTool():
    '''图片工具通用类
    '''
    ''' 定义类常量 '''
    SAVE_TEMP_IMG = False #是否保存每一块图
    IMAGE_LEFT_CENTER = 'img_left_center' #左边中间块
    IMAGE_RIGHT_CENTER = 'img_right_center' #右边中间块
    IMAGE_UPPER_CENTER = 'img_upper_center' #顶部中间块
    IMAGE_LOWER_CENTER = 'img_lower_center' #底部中间块
    IMAGE_CENTER = 'img_center' #中间块
    IMAGE_ORIGINAL = 'img_original' #原始图

    SHARPNESS_METHOD = 'lapalace' #用哪种清晰度算法
    
    '''关闭
    '''
    @staticmethod
    def close(img):
        img.close()
        
    '''打开图片
    '''
    @staticmethod
    def open(filePath):
        img = Image.open(filePath)
        return img
        
    '''convert
    '''
    @staticmethod
    def convert(img, type='L'):
        return img.convert(type)
        
    '''resize图片
    '''
    @staticmethod
    def resize(img, width=360):
        ratio = float(width)/img.size[0]
        height = int(ratio*img.size[1])
        resizedImg = img.resize((width, height))
        #resized.save("tmppic.jpg", format="jpeg")
        return resizedImg

    @staticmethod
    def resize1(img, Width=320,Height=480):
        width =Width
        imgwidth = img.size[0]
        imgheight = img.size[1]
        ratio = float(width)/imgwidth
        height = int(ratio*imgheight)
        if height >= Height:
            height = Height
            width = int(float(height)/imgheight*imgwidth)
        resizedImg = img.resize((width, height))
        #resized.save("tmppic.jpg", format="jpeg")
        return resizedImg
    
    @staticmethod
    def getSharpness(img, name='noname'):
        if ImageTool.SHARPNESS_METHOD == 'fmax':
            return ImageTool.getSharpnessByFmax(img, name)
        elif ImageTool.SHARPNESS_METHOD == 'brightness_var':
            return ImageTool.getBrightnessVar(img),ImageTool.getInformationEntropy(img)
        elif ImageTool.SHARPNESS_METHOD == 'brightness_mean':
            return ImageTool.getBrightnessMean(img),ImageTool.getInformationEntropy(img)
        elif ImageTool.SHARPNESS_METHOD == 'laplace':
            return ImageTool.getLaplaceSharpness(img), ImageTool.getInformationEntropy(img)
        else:
            return ImageTool.getSharpnessByGray(img, name)

    '''计算图像亮度方差
    '''
    @staticmethod
    def getBrightnessVar(img):
        stat = ImageStat.Stat(img)
        return stat.var[0]

    '''计算图像亮度均值
    '''
    @staticmethod
    def getBrightnessMean(img):
        stat = ImageStat.Stat(img)
        return stat.mean[0]


    '''计算清晰度，灰度变化率方式
    '''
    @staticmethod
    def getSharpnessByGray(img, name='noname'):
        gray = ImageGray()
        return gray.getSharpness(img, name)
        
    '''计算清晰度，最大灰度差方式
    '''
    @staticmethod
    def getSharpnessByFmax(img, name='noname'):
        fmax = ImageFmax()
        return fmax.getSharpness(img, name), ImageTool.getInformationEntropy(img)


    '''计算清晰度，拉普拉斯方式
    '''
    @staticmethod
    def getLaplaceSharpness(img):
        laplace = ImageLaplace()
        return laplace.evaluateSharpness(img)


    '''计算图像熵
    '''
    @staticmethod
    def getInformationEntropy(img):
        YY = img.load()
        width = img.width
        height = img.height
        yDict = {}
        for i in range(width):
            for j in range(height):
                if yDict.has_key(YY[i, j]):
                    yDict[YY[i, j]] += 1
                else:
                    yDict[YY[i, j]] = 1

        totalPoint = sum(yDict.values())
        if totalPoint <= 0:
            return -1
        entropy = 0.0
        for key in yDict.keys():
            p = float(yDict[key]) / totalPoint
            if p <= 0:
                print "p取值错误 "
                continue
            entropy += -1 * p * math.log(p, 2)
        return entropy

    '''裁剪图片：去掉顶部和底部1/5，竖向划分成三块
    '''
    @staticmethod
    def cropImage2(img):
        if ImageTool.SAVE_TEMP_IMG: img.save("tmpImg.jpg", format="jpeg")
        
        imgsize = img.size
        width = int(imgsize[0])
        height = int(imgsize[1])
        ''' 下面开始裁剪图片 '''
        ''' 左边中间图片： '''
        leftCenter = (0, height/5, width/3, 4*height/5)  #(left, upper, right, lower)
        leftCenterImg = img.crop(leftCenter)
        if ImageTool.SAVE_TEMP_IMG: leftCenterImg.save("tmpLeftCenter.jpg", format="jpeg")
        ''' 右边中间图片： '''
        rightCenter = (2*width/3, height/5, width, 4*height/5)  #(left, upper, right, lower)
        rightCenterImg = img.crop(rightCenter)
        if ImageTool.SAVE_TEMP_IMG: rightCenterImg.save("tmpRightCenter.jpg", format="jpeg")
        ''' 中间图片： '''
        center = (width/3, height/5, 2*width/3, 4*height/5)  #(left, upper, right, lower)
        centerImg = img.crop(center)
        if ImageTool.SAVE_TEMP_IMG: centerImg.save("tmpCenter.jpg", format="jpeg")
        
        return {ImageTool.IMAGE_ORIGINAL: img, ImageTool.IMAGE_LEFT_CENTER: leftCenterImg, 
                ImageTool.IMAGE_RIGHT_CENTER: rightCenterImg, ImageTool.IMAGE_CENTER: centerImg }
    
    '''裁剪图片：九宫格划分，中间五块
    '''
    @staticmethod
    def cropImage1(img):
        if ImageTool.SAVE_TEMP_IMG: img.save("tmpImg.jpg", format="jpeg")
        
        imgsize = img.size
        width = int(imgsize[0])
        height = int(imgsize[1])
        ''' 下面开始裁剪图片 '''
        ''' 左边中间图片： '''
        leftCenter = (0, height/3, width/3, 2*height/3)  #(left, upper, right, lower)
        leftCenterImg = img.crop(leftCenter)
        if ImageTool.SAVE_TEMP_IMG: leftCenterImg.save("tmpLeftCenter.jpg", format="jpeg")
        ''' 右边中间图片： '''
        rightCenter = (2*width/3, height/3, width, 2*height/3)  #(left, upper, right, lower)
        rightCenterImg = img.crop(rightCenter)
        if ImageTool.SAVE_TEMP_IMG: rightCenterImg.save("tmpRightCenter.jpg", format="jpeg")
        ''' 上边中间图片： '''
        upperCenter = (width/3, 0, 2*width/3, height/3)  #(left, upper, right, lower)
        upperCenterImg = img.crop(upperCenter)
        if ImageTool.SAVE_TEMP_IMG: upperCenterImg.save("tmpUpperCenter.jpg", format="jpeg")
        ''' 下边中间图片： '''
        lowerCenter = (width/3, 2*height/3, 2*width/3, height)  #(left, upper, right, lower)
        lowerCenterImg = img.crop(lowerCenter)
        if ImageTool.SAVE_TEMP_IMG: lowerCenterImg.save("tmpLowerCenter.jpg", format="jpeg")
        ''' 中间图片： '''
        center = (width/3, height/3, 2*width/3, 2*height/3)  #(left, upper, right, lower)
        centerImg = img.crop(center)
        if ImageTool.SAVE_TEMP_IMG: centerImg.save("tmpCenter.jpg", format="jpeg")
        
        return {ImageTool.IMAGE_ORIGINAL: img, ImageTool.IMAGE_LEFT_CENTER: leftCenterImg, 
                ImageTool.IMAGE_RIGHT_CENTER: rightCenterImg, ImageTool.IMAGE_UPPER_CENTER: upperCenterImg,
                ImageTool.IMAGE_LOWER_CENTER: lowerCenterImg, ImageTool.IMAGE_CENTER: centerImg }

    '''图片切块儿
    '''
    @staticmethod
    def cropImagePatches(img,pathname):
        cropsize = 128
        patchSavePath = "/Users/fanchun/Desktop/视频文件分析/机器学习/result"
        patchSavePath1 = "/Users/fanchun/Desktop/视频文件分析/机器学习/result_nonormal"
        videoname = pathname.split('/')[-2]
        name = pathname.split('/')[-1]
        path = os.path.join(patchSavePath,videoname)
        path1 = os.path.join(patchSavePath1,videoname)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path1):
            os.makedirs(path1)
        imgsize = img.size
        width = int(imgsize[0])
        height = int(imgsize[1])
        index = 0
        for i in range(width/cropsize):
            for j in range(height/cropsize):
                rect = (i*cropsize,j*cropsize,(i+1)*cropsize,(j+1)*cropsize)
                tmpimg = img.crop(rect)
                if ImageTool.getInformationEntropy(img=tmpimg.convert('L')) < 6.5:
                    continue
                savedpath = os.path.join(path,"%s_%s_%d.bmp"%(videoname,name,index))
                tmpimg.save(os.path.join(path1,"%s_%s_%d.bmp"%(videoname,name,index)))
                tmpimg = ImageTool.normalize(tmpimg.convert('L'))
                tmpimg.save(savedpath)
                index += 1


class ImageLaplace():
    def __init__(self):
        pass


    def variance_of_laplacian(self,image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def cropMostSharpPatch(self,image):
        crop_size = 400
        image_array = np.array(image)
        height = image_array.shape[0]
        width = image_array.shape[1]
        if height <= 2 * crop_size or width <= 2 * crop_size:
            crop_size = min(width, height) / 2
        _fm, crop_w, crop_h = self.get_best_clear_in_blur_image(image_array, crop_size)
        if crop_w == -1:
            return
        img_crop_array = image_array[crop_h: crop_h + crop_size, crop_w:crop_w + crop_size]
        img_crop = Image.fromarray(img_crop_array)

        return img_crop

    def get_best_clear_in_blur_image(self,image_array, crop_size):
        try:
            height, width, channels = image_array.shape
        except Exception, e:
            return -1, -1, -1
        img_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        if height < crop_size or width < crop_size:
            return -1, -1, -1

        step_size = 10
        x_range = (width - crop_size) / step_size
        y_range = (height - crop_size) / step_size

        pre_fm = 0.0
        pre_crop_w = 0
        pre_crop_h = 0
        for x in range(x_range):
            for y in range(y_range):
                start_w = x * step_size
                start_h = y * step_size
                img_crop = img_gray[start_h: start_h + crop_size, start_w:start_w + crop_size]
                # cv2.imshow("cropped", img_crop)
                # cv2.waitKey(1)
                fm = self.variance_of_laplacian(img_crop)
                if fm > pre_fm:
                    pre_fm = fm
                    pre_crop_h = start_h
                    pre_crop_w = start_w

        # print pre_fm, pre_crop_w, pre_crop_h
        return pre_fm, pre_crop_w, pre_crop_h

    def evaluateSharpness(self,img):

        # handleSaturation(np.array(src))
        # src.convert('L').show()

        gray_img = img.convert('L')
        gray = np.array(gray_img)

        # test(imagePath)

        fm = self.variance_of_laplacian(gray)


        return fm





class ImageGray():
    '''计算图片灰度变化
    '''
    
    '''构造函数
    '''
    def __init__(self):
        pass
    
    '''计算图像的清晰度 和 熵
    '''
    def getSharpness(self, img, name='noname'):
        YYArr = np.array(img.convert('L'))
        grayRate, entropy = self.getSharpnessByPixelData(name, YYArr)
        return grayRate, entropy
        
    '''计算图像像素矩阵的清晰度
    '''
    def getSharpnessByPixelData(self, name, YYArr):
        grayRate, entropy = self.getImageGrayAndEntropy(YYArr)
        util.log("图像块: %s 清晰度: %f  图像熵：%f" % (name, grayRate, entropy))
        return grayRate, entropy
    
    '''计算图片的灰度变化率 和 熵
    '''
    def getImageGrayAndEntropy(self, Y):
        width = Y.shape[1]
        height = Y.shape[0]
        ratio = 0.7071
        P = zeros((height,width))
        YY = copy.copy(Y)
        YY = YY.astype(float)
        # print "计算中。。。"
        yDict = {}
        for i in range(height):
            for j in range(width):
                t = 0
                if i > 0:
                    P[i,j] += abs(YY[i,j]-YY[i-1,j])
                    t += 1
                if i < height-1:
                    P[i,j] += abs(YY[i,j]-YY[i+1,j])
                    t += 1
                if j > 0:
                    P[i,j] += abs(YY[i,j]-YY[i,j-1])
                    t += 1
                if j < width-1:
                    P[i,j] += abs(YY[i,j]-YY[i,j+1])
                    t += 1
                if i > 0 and j > 0:
                    P[i,j] += ratio*abs(YY[i,j]-YY[i-1,j-1])
                    t += 1
                if i > 0 and j < width-1:
                    P[i,j] += ratio*abs(YY[i,j]-YY[i-1,j+1])
                    t += 1
                if i < height-1 and j > 0:
                    P[i,j] += ratio*abs(YY[i,j]-YY[i+1,j-1])
                    t += 1
                if i < height-1 and j < width-1:
                    P[i,j] += ratio*abs(YY[i,j]-YY[i+1,j+1])
                    t += 1
                P[i,j] = P[i,j]/t
                # if t != 8:
                #     print "t error"
                if yDict.has_key(YY[i,j]):
                    yDict[YY[i,j]] += 1
                else:
                    yDict[YY[i,j]] = 1
    
    
        rst = sum(P)/((width)*(height))
        entropy = self.getInformationEntropy(yDict)
        return (rst,entropy)
    
    '''计算图片的信息熵
    '''
    def getInformationEntropy(self, yDict):
        totalPoint = sum(yDict.values())
        entropy = 0.0
        for key in yDict.keys():
            p = float(yDict[key])/totalPoint
            if p <= 0:
                print "p取值错误"
            entropy += -1 * p * math.log(p, 2)
        return entropy


class ImageFmax():
    '''计算图片最大灰度差（微信读书算法）
    '''
    
    '''计算图像的清晰度
    '''
    def getSharpness(self, img, name='noname'):
        img_array = img.load()
        
        width = img.size[0]
        heigh = img.size[1]
        maxP = 0
        maxDirection = 0
        maxP_x = 0
        maxP_y = 0
        d1=d2=d3=d4=d5=d6=0
        
        for w in range(4,width-4):
            for h in range(4,heigh-4):
                S0 = self.gradient(img_array,w-2,h,0)+self.gradient(img_array,w-1,h,0)+self.gradient(img_array,w,h,0)\
                +self.gradient(img_array,w+1,h,0)+self.gradient(img_array,w+2,h,0)
                if S0 > maxP:
                    maxP = S0; maxP_x = w; maxP_y = h; maxDirection = 0
                S90 = self.gradient(img_array,w,h-2,90)+self.gradient(img_array,w,h-1,90)+self.gradient(img_array,w,h,90)\
                +self.gradient(img_array,w,h+1,90)+self.gradient(img_array,w,h+2,90)
                if S90 > maxP:
                    maxP = S90; maxP_x = w; maxP_y = h; maxDirection = 90
                S45 = self.gradient(img_array,w-2,h+2,45)+self.gradient(img_array,w-1,h+1,45)+self.gradient(img_array,w,h,45)\
                +self.gradient(img_array,w+1,h-1,45)+self.gradient(img_array,w+2,h-2,45)
                if S45 > maxP:
                    maxP = S45; maxP_x = w; maxP_y = h; maxDirection = 45
                S135 = self.gradient(img_array,w-2,h-2,135)+self.gradient(img_array,w-1,h-1,135)+self.gradient(img_array,w,h,135)\
                +self.gradient(img_array,w+1,h+1,135)+self.gradient(img_array,w+2,h+2,135)
                if S135 > maxP:
                    maxP = S135; maxP_x = w; maxP_y = h; maxDirection = 135
        
        # 1)
        if maxDirection == 0:
            d1 = abs(img_array[maxP_x,maxP_y-3]-img_array[maxP_x,maxP_y-2])
            d2 = abs(img_array[maxP_x,maxP_y-2]-img_array[maxP_x,maxP_y-1])
            d3 = abs(img_array[maxP_x,maxP_y-1]-img_array[maxP_x,maxP_y])
            d4 = abs(img_array[maxP_x,maxP_y+1]-img_array[maxP_x,maxP_y])
            d5 = abs(img_array[maxP_x,maxP_y+2]-img_array[maxP_x,maxP_y+1])
            d6 = abs(img_array[maxP_x,maxP_y+3]-img_array[maxP_x,maxP_y+2])
        # 2)
        elif maxDirection == 90:
            d1 = abs(img_array[maxP_x-3,maxP_y]-img_array[maxP_x-2,maxP_y])
            d2 = abs(img_array[maxP_x-2,maxP_y]-img_array[maxP_x-1,maxP_y])
            d3 = abs(img_array[maxP_x-1,maxP_y]-img_array[maxP_x,maxP_y])
            d4 = abs(img_array[maxP_x+1,maxP_y]-img_array[maxP_x,maxP_y])
            d5 = abs(img_array[maxP_x+2,maxP_y]-img_array[maxP_x+1,maxP_y])
            d6 = abs(img_array[maxP_x+3,maxP_y]-img_array[maxP_x+2,maxP_y])
        # 3)
        elif maxDirection == 45:
            d1 = abs(img_array[maxP_x-3,maxP_y-3]-img_array[maxP_x-2,maxP_y-2])
            d2 = abs(img_array[maxP_x-2,maxP_y-2]-img_array[maxP_x-1,maxP_y-1])
            d3 = abs(img_array[maxP_x-1,maxP_y-1]-img_array[maxP_x,maxP_y])
            d4 = abs(img_array[maxP_x+1,maxP_y+1]-img_array[maxP_x,maxP_y])
            d5 = abs(img_array[maxP_x+2,maxP_y+2]-img_array[maxP_x+1,maxP_y+1])
            d6 = abs(img_array[maxP_x+3,maxP_y+3]-img_array[maxP_x+2,maxP_y+2])
        # 4)
        elif maxDirection == 135:
            d1 = abs(img_array[maxP_x-3,maxP_y+3]-img_array[maxP_x-2,maxP_y+2])
            d2 = abs(img_array[maxP_x-2,maxP_y+2]-img_array[maxP_x-1,maxP_y+1])
            d3 = abs(img_array[maxP_x-1,maxP_y+1]-img_array[maxP_x,maxP_y])
            d4 = abs(img_array[maxP_x+1,maxP_y-1]-img_array[maxP_x,maxP_y])
            d5 = abs(img_array[maxP_x+2,maxP_y-2]-img_array[maxP_x+1,maxP_y-1])
            d6 = abs(img_array[maxP_x+3,maxP_y-3]-img_array[maxP_x+2,maxP_y-2])
            
        dmax = max(d1,d2,d3,d4,d5,d6); dmin = min(d1,d2,d3,d4,d5,d6)
        Fmax = dmax - dmin
        
        tmp1 = d1+d2+d3;tmp2=d2+d3+d4;tmp3=d3+d4+d5;tmp4=d4+d5+d6;
        d3max = max(tmp1,tmp2,tmp3,tmp4)
        m1 = float(dmax) / (2*Fmax)
        m3 = float(d3max) / (2*Fmax)
        
        row = "%s\tFmax=%d\tmax_x=%d\tmax_y=%d\tdirection=%d\tm1=%f\tm3=%f" %(name,Fmax,maxP_x,maxP_y,maxDirection,m1,m3)
        util.log(row)
        return Fmax
    
    '''计算梯度
    '''
    def gradient(self, img_array, x, y, direction):
        if direction == 0:
            res = math.pow(img_array[x,y]-img_array[x,y-2],2) + math.pow(img_array[x,y]-img_array[x,y+2],2)
            return math.sqrt(res)
        elif direction == 90:
            res = math.pow(img_array[x,y]-img_array[x+2,y],2) + math.pow(img_array[x,y]-img_array[x-2,y],2)
            return math.sqrt(res)
        elif direction == 45:
            res = math.pow(img_array[x,y]-img_array[x-2,y-2],2) + math.pow(img_array[x,y]-img_array[x+2,y+2],2)
            return math.sqrt(res)
        else:
            res = math.pow(img_array[x,y]-img_array[x+2,y-2],2) + math.pow(img_array[x,y]-img_array[x-2,y+2],2)
            return math.sqrt(res)
        

if __name__ == '__main__': 
    image = ImageTool.open('/Users/fanchun/Desktop/视频文件分析/机器学习/抽帧/--rKknOGdQeMYxrwTUSpVw__.mp4/frame001.jpg')
    ''' 按比例缩放图片到统一大小 '''
    rsz = ImageTool.resize1(image)
    rsz.show()

    # ''' 裁剪图片，得到多个图片块 '''
    # images = ImageTool.cropImagePatches(image,'/Users/fanchun/Desktop/视频文件分析/机器学习/抽帧/--rKknOGdQeMYxrwTUSpVw__.mp4/frame001.jpg')




