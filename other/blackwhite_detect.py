#coding:utf-8


import sys
from PIL import Image
import numpy as np
import time
import cv2
import logging


PIC_STEP = 66           # the step for crop to MODEL_PIC_SIZE
RATIO_THR = 0.7         # algorithm thresh for decide split
MODEL_PIC_SIZE = 227    # pic size


USE_ALGTHM = 1         # used algorithm to decide split or not

RESIZE_TYPE = 1         # resize to (227,227) if 1, else resize to (227,?) or (?,227)
BLACK_BAR_THR = 1
AVR_VAL_THR = 5    # use to filter blocking effect


def judge(ori_image):
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F)

    img = Image.open(ori_image).convert('L')
    black = np.array(img)
    new = variance_of_laplacian(black)
    (m,n) = new.shape
    for i in range(m):
        for j in range(n):
            new[i,j] = new[i,j] if new[i,j] >=0 else 0

    maxsum = 0
    secondmaxsum = 0
    index = 0
    nextindex = 0
    for i in range(n):
        if sum(new[:,i]) > maxsum:
            nextindex = index
            secondmaxsum = maxsum
            index = i
            maxsum = sum(new[:,i])
        elif sum(new[:,i]) > secondmaxsum:
            nextindex = i
            secondmaxsum = sum(new[:,i])


    if index < BLACK_BAR_THR or n - index < BLACK_BAR_THR:
        # 5 pixel split screen can't be sensed , handle the case that copy screen
        return 0

    # Image.fromarray(new).show()
    # Image.fromarray(new[:,index:]).show()


    num = 0.0
    for i in range(m):
        num += 1.0 if new[i,index] > 0 and new[i,index]*new[i,index-1]*new[i,index+1]==0 else 0
    ratio1 = num/m

    print ('use algorithm index:%d: ratio:%f  averagevalue: %f '%(index,ratio1,maxsum/(float(n))))

    if maxsum/(float(m)) < AVR_VAL_THR:
        # handle with blocking effect
        return 0

    if abs(index - nextindex) == 1:
        return 0

    if ratio1 > RATIO_THR:
        return 1
    else:
        return 0



if __name__ == '__main__':
    img_path = sys.argv[1]
    #img_path = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/huabian.jpg'
    print judge(img_path)