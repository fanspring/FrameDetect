# -*- coding: utf-8 -*-  

'''适配图片大小大于227的问题
    
'''

from PIL import Image
import numpy as np
import math,os
import urllib2
import cStringIO
import hashlib
import cv2
from FDConfig import *

"""config value
"""

PIC_STEP = 66           # the step for crop to MODEL_PIC_SIZE
RATIO_THR = 0.85         # algorithm thresh for decide split
MODEL_PIC_SIZE = 227    # pic size


USE_ALGTHM = 1         # used algorithm to decide split or not

RESIZE_TYPE = 1         # resize to (227,227) if 1, else resize to (227,?) or (?,227)
BLACK_BAR_THR = 20
AVR_VAL_THR = 5    # use to filter blocking effect


def adaptPicSize(img):
    resized = resizePicSize(img)
    width = resized.size[0]
    height = resized.size[1]
    pic_ar = [img]
    if height > 227 and width == 227:
        num = int(math.ceil((height - 227) / float(PIC_STEP))) + 1
        util.log('crop tmp pics num:%d'%num)
        for i in range(num):
            if (i * PIC_STEP + 227) <= height:
                box = (0, i * PIC_STEP, 227, i * PIC_STEP + 227)
            else:
                box = (0, height - 227 if height - 227 >= 0 else 0, 227, height)
            pic_ar += [resized.crop(box)]
    elif height == 227 and width > 227:
        num = int(math.ceil((width - 227) / float(PIC_STEP))) + 1
        util.log('crop tmp pics num:%d'%num)
        for i in range(num):
            if (i * PIC_STEP + 227) <= width:
                box = (i * PIC_STEP, 0, i * PIC_STEP + 227, 227)
            else:
                box = (width - 227 if width - 227 >= 0 else 0, 0, width, 227)
            pic_ar += [resized.crop(box)]
    else:
        pic_ar += [resized]
    return pic_ar

def resizePicSize(img):
    ori_width = img.size[0]
    ori_height = img.size[1]
    if RESIZE_TYPE == 1:
        resized = img.resize((MODEL_PIC_SIZE,MODEL_PIC_SIZE),resample = Image.ANTIALIAS)
    else:
        if ori_width < ori_height:
            width = MODEL_PIC_SIZE
            ratio = float(width) / ori_width
            height = int(ratio * ori_height)
        else:
            height = MODEL_PIC_SIZE
            ratio = float(height) / ori_height
            width = int(ratio * ori_width)
        resized = img.resize((width,height),resample = Image.ANTIALIAS)
    return resized


def cropImgPathes(img):
    width = img.size[0]
    height = img.size[1]
    pic_ar = [img]
    m = int(math.ceil(float(width)/MODEL_PIC_SIZE))
    n = int(math.ceil(float(height)/MODEL_PIC_SIZE))
    for i in range(m):
        for j in range(n):
            box = [i*MODEL_PIC_SIZE,j*MODEL_PIC_SIZE,(i+1)*MODEL_PIC_SIZE,(j+1)*MODEL_PIC_SIZE]
            if box[2] > width:
                box[0] = width - MODEL_PIC_SIZE if width - MODEL_PIC_SIZE > 0 else 0
                box[2] = width
            if box[3] > height:
                box[1] = height - MODEL_PIC_SIZE if height - MODEL_PIC_SIZE > 0 else 0
                box[3] = height
            pic_ar += [img.crop(box)]
    return pic_ar




def getRemotePic(url):
    try:
        logger.debug('Start download pic from url:%s' % url)
        data = urllib2.urlopen(url).read()
        file = cStringIO.StringIO(data)
        img = Image.open(file)
    except Exception as e:
        logger.info('')
        img = None
        logger.warning('download pic[{}] failed, reason:{}'.format(url,e))
    finally:
        logger.debug('End download')
    return img



def download_if_need(image_paths):
    new_image_paths_ar = []
    for image_path in image_paths:
        if 'http' in image_path:
            img = getRemotePic(image_path)
        else:
            img = Image.open(image_path)

        if not img: continue

        # handle error in reading png
        img = img.convert('RGB')
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))

        # crop or resize if needed
        pic_ar = crop_or_resize(img)

        # save to tmppath and return new paths
        new_image_paths_ar += save_picar_tmpfolder(image_path,pic_ar,img)

        del pic_ar[:]
        del img

    return new_image_paths_ar




def save_picar_tmpfolder(ori_image_path,pic_ar,ori_img):
    new_image_paths = []
    # add pid in the folder name in case of different process handle same picture urls
    foldername = '{}_{}'.format(PROCESS_PID,get_md5_value(ori_image_path))
    folderpath = os.path.join(tmpPath(), foldername)
    util.make_dir(folderpath)
    ori_img.save(os.path.join(folderpath, 'ori.jpg'))
    for i in range(len(pic_ar)):
        tmppicpath = os.path.join(folderpath, 'tmp_%d.jpg' % i)
        pic_ar[i].save(tmppicpath)
        new_image_paths += [tmppicpath]

    return new_image_paths

def get_original_image(image_path):
    dir_path = osp.dirname(image_path)
    ori_path = osp.join(dir_path,'ori.jpg')
    if os.path.exists(ori_path):
        return ori_path
    else:
        return image_path


def clear_tmp_files(image_paths):
    __import__('shutil').rmtree(os.path.dirname(image_paths[0]))




def crop_or_resize(img):

    if FDConfig.RESIZE_TO_PREDICT and FDConfig.CROP_TO_PREDICT:
        pic_ar = adaptPicSize(img)
    elif FDConfig.CROP_TO_PREDICT:
        pic_ar = cropImgPathes(img)
    elif FDConfig.RESIZE_TO_PREDICT:
        pic_ar = [resizePicSize(img)]
    else:
        pic_ar = [img]
    util.log('[crop]: %s  [resize]:%s [resize_type]%d'%(FDConfig.CROP_TO_PREDICT,FDConfig.RESIZE_TO_PREDICT,RESIZE_TYPE))
    return pic_ar






"""Get md5 for path
"""

def get_sha1_value(src):
    mySha1 = hashlib.sha1()
    mySha1.update(src)
    mySha1_Digest = mySha1.hexdigest()
    return mySha1_Digest

def get_md5_value(src):
    myMd5 = hashlib.md5()
    myMd5.update(src)
    myMd5_Digest = myMd5.hexdigest()
    return myMd5_Digest



"""Judge the split or flur result by probs
"""
def judgeRst(image_paths,probs,mode ):
    rst = 0
    for prob in probs:
        if prob[1]>0.999:
            if USE_ALGTHM and mode == V_MODE_SPLIT:
                rst = split_judge_inner(image_paths[0])
            else:
                rst =1
    if mode == V_MODE_FLUR and FDConfig.CROP_TO_PREDICT and not FDConfig.RESIZE_TO_PREDICT:
        rst = flur_judge_inner(probs)

    return rst

def flur_judge_inner(probs):
    rst = 0
    prob_num = 0
    num = len(probs)
    if not probs[0][1] > 0.6:
        return 0
    for prob in probs:
        if prob[1] > 0.6:
            prob_num += 1
    if num < 9:
        if prob_num > 2:
            rst =1
    else:
        if float(prob_num)/num > 0.3:
            rst =1
    return rst



def split_judge_inner(filename):
    def variance_of_laplacian(image):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F)

    ori_image = get_original_image(filename)
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
    for i in range(m):
        if sum(new[i,:]) > maxsum:
            nextindex = index
            secondmaxsum = maxsum
            index = i
            maxsum = sum(new[i,:])
        elif sum(new[i,:]) > secondmaxsum:
            nextindex = i
            secondmaxsum = sum(new[i,:])

    if index < BLACK_BAR_THR or m - index < BLACK_BAR_THR:
        # 5 pixel split screen can't be sensed , handle the case that copy screen
        return 0

    # Image.fromarray(new).show()
    # Image.fromarray(new[index:,:]).show()
    # print index,m-index
    # print '[test]',maxsum/(float(n))
    # print new[index - 2, :]
    # print new[index-1,:]
    # print new[index, :]
    # print new[index + 1, :]
    # print new[index + 2, :]

    num = 0.0
    for i in range(n):
        num += 1.0 if new[index,i] > 0 and new[index,i]*new[index-1,i]*new[index+1,i]==0 else 0
    ratio1 = num/n

    util.log('use algorithm index:%d: ratio:%f  averagevalue: %f '%(index,ratio1,maxsum/(float(n))))

    if maxsum/(float(n)) < AVR_VAL_THR:
        # handle with blocking effect
        return 0

    if abs(index - nextindex) == 1:
        return 0

    if ratio1 > RATIO_THR:
        return 1
    else:
        return 0




    # num = 0.0
    # num1 = 0.0
    # for i in range(n):
    #     num += 1.0 if new[index,i] >0 else 0
    #     num1 += 1.0 if new[nextindex,i] >0 else 0
    #
    # ratio1 = num/n
    # ratio2 = num1/n
    # util.log('use algorithm index:%d sum: %d nextindex:%d  nextsum: %d ratio1: %f  ratio2: %f  total_n: %d'%(index,maxsum,nextindex,secondmaxsum,ratio1,ratio2,n))
    #
    # if abs(index-nextindex) == 1:
    #     return 0
    #
    # if ratio1 > RATIO_THR or ratio2 > RATIO_THR:
    #     return 1
    # else:
    #     return 0


def handle_final_result(image_paths,rst_dict):
    ori_image = get_original_image(image_paths[0])
    img = Image.open(ori_image)

    # correct case that copy screen by width and height {864, 480}, {480, 864}
    width = img.size[0]
    height = img.size[1]
    if (width == 864 and height == 480) or (width == 480 and height == 864):
        if rst_dict[V_MODE_SPLIT] > 0:
            logger.info('picture is a copy of screen, width: %d  height: %d, do not detect split'%(width,height))
            rst_dict[V_MODE_SPLIT] = 0

    return rst_dict


if __name__ == '__main__':
    img = Image.open('/Users/fanchun/Desktop/视频文件分析/分屏花屏/外网检测结果/Mjc1NjczMDk2MQ==_448468653_1400007595_1510231188.jpg')
    ar = cropImgPathes(img)
    for item in ar:
        item.show()

    pass




