#coding:utf-8


import sys
from PIL import Image
import numpy as np
import math
import util,time
import cv2


kThreshold = 0
kRangeOfGray = 256
PURE_DETECT_THRESHOLD = 0.5
BLACK_WHITE_DETECT_THRESHOLD = 100
DETECT_TYPE = 1  # if 1 use lapalace; else use entropy




# judege color
COLOR_S_THRESHOLD = 230
BLACK_S_THRESHOLD = 1
BLACK_V_THRESHOLD = 46



BLACK = 1
GREEN = 2
RED = 3
BLUE = 4
OTHER = 10
BLACK_WHITE = 11


NO_COLOR = -1


color_dict = {
    GREEN: [55,65],
    RED:   [118,122],
    BLUE:  [9,11]
}




def get_entropy(img):
    img_width = img.size[0]
    img_height = img.size[1]
    size_pixel = img_height * img_width

    gray_img = img.convert('L')
    img_array = np.array(gray_img)


    gray_count_array = [0 for x in range(kRangeOfGray)]

    for h in range(img_height):
        for w in range(img_width):
            gray = img_array[h, w]
            if gray >= kRangeOfGray:
                continue
            gray_count_array[gray] += 1

    # print gray_count_array

    entropy = 0.0
    for x in range(kRangeOfGray):
        if gray_count_array[x] == 0:
            continue

        value = gray_count_array[x] * 1.0 / size_pixel
        entropy -= value * math.log(value, 2)

    return entropy 

def getColor(imagearray):
    HSV = cv2.cvtColor(imagearray, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    h = H.mean()
    s = S.mean()
    v = V.mean()
    if s > COLOR_S_THRESHOLD and v > BLACK_V_THRESHOLD:
        for color in color_dict.keys():
            if h >= color_dict[color][0] and h <= color_dict[color][1]:
                return color
    if s < BLACK_S_THRESHOLD and v < BLACK_V_THRESHOLD:
        return BLACK
    if s < BLACK_S_THRESHOLD and h < 5:
        return NO_COLOR
    return OTHER

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    img_ar = np.array(image)
    return cv2.Laplacian(img_ar, cv2.CV_64F).var()

def start_detect(img):
    if DETECT_TYPE ==1:
        return variance_of_laplacian(img)
    else:
        return get_entropy(img)

def detect_result(img_path):
    util.log('start detect pure %s'%img_path)
    t1 = time.time()
    img = Image.open(img_path)
    detect_value = start_detect(img)
    t2 = time.time()
    util.log('end detect pure time:%.3fs ' % (t2-t1))
    if detect_value <= PURE_DETECT_THRESHOLD:
        #是纯色
        return getColor(np.array(img))
    elif detect_value > BLACK_WHITE_DETECT_THRESHOLD:
        if getColor(np.array(img)) == NO_COLOR:
            return BLACK_WHITE
        else:
            return 0
    else:
        #不是纯色
        return 0
    
def detect(img_paths):
    return [detect_result(img_path) for img_path in img_paths]



if __name__ == '__main__':
    img_path = sys.argv[1]
    util.log(detect_result(img_path))