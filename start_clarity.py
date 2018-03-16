# -*- coding: utf-8 -*-
import argparse
import numpy as np


import json
import QZHelper as qz
from classify import classify,classify_new
from other import pure_detect as puredetect
from FDConfig import *
from classifymodule import ClassifyModule

CONFIG_PATH = "model.conf"
WEIGHT_FILE = osp.join(runPath(), 'weights/FT_live.npy')




def read_config_paras():
    # Parse arguments for weight file
    config = ConfigParser.ConfigParser()
    config.read(osp.join(runPath(),CONFIG_PATH))

    # Parse arguments for imagepath
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', nargs='+', help='One or more images to classify')
    args = parser.parse_args()
    image_paths = args.image_paths
    return image_paths


def main():

    # read detect mode  pic path and  config path
    image_paths = read_config_paras()



    model = ClassifyModule(model_type='FT_VGG',model_data_path=WEIGHT_FILE,oversample= 3,oversample_num= 20)  #new method
    probs = model.predict_imagepaths(image_paths)

    logger.info('[result]{}'.format(probs))







if __name__ == '__main__':
    main()



