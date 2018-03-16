# -*- coding: utf-8 -*-
import argparse
import numpy as np

import os.path as osp
import util
import ConfigParser
import json
import QZHelper as qz
from classify import classify,classify_new
from classifymodule import ClassifyModule,ClassifyModule_Tensorflow
from other import pure_detect as puredetect
import os
from FDConfig import *
import csv
import cv2
from PIL import Image
CONFIG_PATH = "model.conf"


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open(osp.join(qz.runPath(),'imagenet-classes.txt'), 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    util.log('{:50} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    util.log('-' * 100)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        util.log('{:50} {:30} {} %'.format(img_name, class_name, confidence))




def read_config_paras():
    # Parse arguments for weight file
    config = ConfigParser.ConfigParser()
    util.log('read weight file %s'%osp.join(runPath(),CONFIG_PATH))
    config.read(osp.join(runPath(),CONFIG_PATH))

    FDConfig.read()

    # Parse arguments for imagepath
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', nargs='+', help='One or more images to classify')
    parser.add_argument("-m", "--mode", required=False,default='0', help="path to input directory of images")
    args = parser.parse_args()
    image_paths = args.image_paths
    mode = args.mode

    # detect split frame
    if mode == 'split':
        firstkey = V_MODE_SPLIT
        FDConfig._mode = V_MODE_SPLIT
    elif mode == 'flur':
        firstkey = V_MODE_FLUR
        FDConfig._mode = V_MODE_FLUR
    else:
        firstkey = FDConfig._mode

    # FDConfig.CROP_TO_PREDICT = (firstkey == V_MODE_FLUR)  # crop pic before preidcting or not
    # FDConfig.RESIZE_TO_PREDICT = (firstkey == V_MODE_SPLIT)  # resize pic before preidcting or not
    return firstkey,image_paths,config


def main():

    # read detect mode  pic path and  config path
    firstkey, image_paths, config = read_config_paras()



    #test start
    image_paths = getTestArray('/Users/fanchun/Desktop/视频文件分析/分屏花屏/huapin/train/test2')
    # image_paths = ['/Users/fanchun/Desktop/视频文件分析/分屏花屏/testaccuracy']

    #test end




    old_image_paths = image_paths





    # 1. get picture if needed
    url = image_paths[0]
    image_paths = qz.download_if_need(image_paths)




    # dict for result
    rst_dict = { 'url'  : url,
                 qz.V_MODE_SPLIT: -1,
                 qz.V_MODE_FLUR : -1,
                 qz.V_MODE_PURE : -1
                }

    # final result str for printing
    rst_str = '[result]'

    # 2. classiy flur or split
    if firstkey in (qz.V_MODE_SPLIT,qz.V_MODE_FLUR):
        weight_file = config.get(firstkey, 'WEIGHTS_FILE')
        weight_file_path = osp.join(qz.runPath(), weight_file)
        mean_file = config.get(firstkey, 'MEAN_FILE')
        mean_file_path = osp.join(qz.runPath(), mean_file)

        # Classify the image using ML
        #probs = classify_new(model_data_path=weight_file_path, image_paths=image_paths,mean_file_path=mean_file_path)

        # new Classify the image using ML
        # model = ClassifyModule(model_type='SplitAlexNet',model_data_path=weight_file_path,mean_file_path=mean_file_path)
        # probs = model.predict_imagepaths(image_paths)

        # tensorflow mode predict
        model = ClassifyModule_Tensorflow(model_data_path=weight_file_path,input_size=224)
        probs = model.predict_imagepaths(image_paths)

        # probs = []
        # for i in range(len(image_paths)):
        #     probs += [[0,1]]


        #display_results(image_paths, probs)

        csvfile = file('test_result.csv', 'wb')
        writer = csv.writer(csvfile)
        split_num = 0
        for i in range(len(old_image_paths)):
            tmppath = []
            tmpprob = []
            for j in range(len(image_paths)):
                if qz.get_md5_value(old_image_paths[i]) in image_paths[j]:
                    tmppath += [image_paths[j]]
                    tmpprob += [probs[j]]
            # judge the result
            rst = qz.judgeRst(tmppath, tmpprob, firstkey)
            logger.info('[image result]:%s,%f,%f,%d'%(old_image_paths[i],tmpprob[0][0],tmpprob[0][1],rst))

            writer.writerow([osp.basename(old_image_paths[i]),tmpprob[0][0],tmpprob[0][1],rst])


            pure = puredetect.detect([old_image_paths[i]])[0]

            util.log('rst: %d,%d'%(rst,pure))

            if rst:
                cp_path = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/huapin/train/detect_flur'
                util.run_command('cp %s %s'%(old_image_paths[i],cp_path))

        logger.info('split num : %d'%split_num)
        csvfile.close()










        # rst_str += '%d,' % rst + '[%s result][0]: %.3f%% [1]: %.3f%%'%(firstkey,probs[0][0]*100.0,probs[0][1]*100.0)
        #
        # rst_dict[firstkey] = rst






    #
    # rst_str += ',[url]:%s' % url
    #
    # json_str = json.dumps(rst_dict)
    #
    #
    # util.log('[json result]'+json_str)
    # util.log(rst_str)

    # 4. delete pic
    #os.remove(image_paths[0])


def main1():
    firstkey, image_paths, config = read_config_paras()
    #test start
    image_paths = getTestArray('/Users/fanchun/Desktop/视频文件分析/分屏花屏/外网检测结果/外网作为训练集309-312/qzone')

    old_image_paths = image_paths

    image_paths = qz.download_if_need(image_paths)

    # 2. classiy flur or split
    if firstkey in (qz.V_MODE_SPLIT,qz.V_MODE_FLUR):
        weight_file = config.get(firstkey, 'WEIGHTS_FILE')
        weight_file_path = osp.join(qz.runPath(), weight_file)
        mean_file = config.get(firstkey, 'MEAN_FILE')
        mean_file_path = osp.join(qz.runPath(), mean_file)

        # Classify the image using ML
        #probs = classify_new(model_data_path=weight_file_path, image_paths=image_paths,mean_file_path=mean_file_path)

        # new Classify the image using ML
        # model = ClassifyModule(model_type='SplitAlexNet',model_data_path=weight_file_path,mean_file_path=mean_file_path)
        # probs = model.predict_imagepaths(image_paths)

        # tensorflow mode predict
        model = ClassifyModule_Tensorflow(model_data_path=weight_file_path,input_size=224)
        probs = model.predict_imagepaths(image_paths)


        to_path_root = '/Users/fanchun/Desktop/视频文件分析/分屏花屏/外网检测结果/外网作为训练集309-312/qzone_new'
        for i in range(len(old_image_paths)):
            score = probs[i][1]
            old = old_image_paths[i]
            name = '{:.3f}_{:.1f}_'.format(score,variance_of_laplacian(old))+osp.basename(old)
            to_path = osp.join(to_path_root,name)
            util.run_command('cp {} {}'.format(old,to_path))


def variance_of_laplacian(imagePath):
    src = Image.open(imagePath)
    img = src.convert('L')
    gray = np.array(img)
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian

    return cv2.Laplacian(gray, cv2.CV_64F).var()


def getTestArray(path):
    files = os.listdir(path)
    dst = []
    for filename in files:
        if '.jpg' in filename:
            dst += [os.path.join(path,filename)]
    return dst



if __name__ == '__main__':
    main1()
    #print variance_of_laplacian('/Users/fanchun/Desktop/视频文件分析/分屏花屏/外网检测结果/flur_qzone_0306-0308得分/1.000_MzQ0MDM1NTA=_517027101_1400007595_1520298768.jpg')


