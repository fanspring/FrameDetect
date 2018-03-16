# -*- coding: utf-8 -*-
import argparse
import numpy as np


import json
import QZHelper as qz
from classify import classify,classify_new
from other import pure_detect as puredetect
from FDConfig import *
from classifymodule import ClassifyModule,ClassifyModule_Tensorflow

CONFIG_PATH = "model.conf"


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image
    '''
    # Get a list of ImageNet class labels
    with open(osp.join(runPath(),'imagenet-classes.txt'), 'rb') as infile:
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


    # 1. get picture if needed
    url = image_paths[0]
    image_paths = qz.download_if_need(image_paths)

    if len(image_paths) == 0:
        return

    # dict for result
    rst_dict = { 'url'  : url,
                 V_MODE_SPLIT: -1,
                 V_MODE_FLUR : -1,
                 V_MODE_PURE : -1
                }

    # final result str for printing
    rst_str = '[result]'

    # 2. classiy flur or split
    if firstkey in (V_MODE_SPLIT,V_MODE_FLUR):
        weight_file = config.get(firstkey, 'WEIGHTS_FILE')
        weight_file_path = osp.join(runPath(), weight_file)
        mean_file = config.get(firstkey, 'MEAN_FILE')
        mean_file_path = osp.join(runPath(), mean_file)
        logger.info('read weiht file:%s,%s'%(weight_file_path,mean_file_path))

        # Classify the image using ML
        #probs = classify_new(model_data_path=weight_file_path, image_paths=image_paths,mean_file_path=mean_file_path)  #old method
        #model = ClassifyModule(model_type='SplitAlexNet',model_data_path=weight_file_path, mean_file_path=mean_file_path, oversample= 3, oversample_num= 1)  #new method
        model = ClassifyModule_Tensorflow(model_data_path=weight_file_path,input_size=224)
        probs = model.predict_imagepaths(image_paths)

        display_results(image_paths, probs)

        # judge the result
        rst = qz.judgeRst(image_paths, probs, firstkey)


        rst_str += '%d,' % rst + '[%s result][0]: %.3f%% [1]: %.3f%%'%(firstkey,probs[0][0]*100.0,probs[0][1]*100.0)

        rst_dict[firstkey] = rst


    # 3. detect pure frame
    if config.get('service', 'PURE') == 'ON':
        pure = puredetect.detect(image_paths[0:1])
        rst_str += ' [pure result] %d'%pure[0]
        rst_dict[V_MODE_PURE] = pure[0]

    rst_str += ',[url]:%s' % url

    json_str = json.dumps(rst_dict)


    util.log('[json result]'+json_str)
    util.log(rst_str)



    # 4. delete pic
    #qz.clear_tmp_files(image_paths)


def main1():
    # read detect mode  pic path and  config path
    firstkey, image_paths, config = read_config_paras()

    model = ClassifyModule(model_type='SplitAlexNet',model_data_path=FDConfig._weight_file, mean_file_path=FDConfig._mean_file)
    __import__('service').predict(model, image_paths)


if __name__ == '__main__':
    main()



