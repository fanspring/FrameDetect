# -*- coding: utf-8 -*-
from util import ut
from os import path as osp, getpid
import util

import ConfigParser
import logging
from logging.handlers import TimedRotatingFileHandler
import re



PID_PATH = '/Users/fanchun/Documents/QZoneCode/avtest_proj/branches/AVTest_Video/FrameDetect2/pid'
LOG_PATH = "/Users/fanchun/Documents/QZoneCode/avtest_proj/branches/AVTest_Video/FrameDetect2/logs/"

# PID_PATH = '/data/pvtest/pythonlog'
# LOG_PATH = "/data/pvtest/pythonlog/"


CONFIG_PATH = "model.conf"
TMP_PATH = 'tmp/'

V_MODE_FIRST_KEY = 'service'
V_MODE_SPLIT_SECOND_KEY = 'SPLIT'
V_MODE_FLUR_SECOND_KEY = 'FLUR'
V_MODE_PURE_SECOND_KEY = 'PURE'
V_WEIGHT_FILE_SECOND_KEY = 'WEIGHTS_FILE'
V_MEAN_FILE_SECOND_KEY = 'MEAN_FILE'


V_MODE_SPLIT = 'split'
V_MODE_FLUR = 'flur'
V_MODE_PURE = 'pure'

appid_dict = {
    'QZONE': 1251983456,
    'NOW': 1251953721
}

PROCESS_PID = getpid()

def log_init():
    logger = logging.getLogger("FDService")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s [%(filename)s][%(process)d] %(message)s")
    log_file_handler = TimedRotatingFileHandler(filename=LOG_PATH+"fdlog", when="midnight", interval=1, backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)
    return logger,log_file_handler

'''log config
'''
logger,handler = log_init()



def runPath():
    __filepath = osp.split(osp.realpath(__file__))[0]
    return __filepath

def tmpPath():
    path = osp.join(runPath(), TMP_PATH)
    util.make_dir(path)  # 创建临时文件夹
    logger.debug('tmp path folder:%s' % path)
    return path

class FDConfig():
    '''
    read project's config value
    '''
    _mode = ''   # detect mode
    _weight_file = ''
    _mean_file = ''
    _if_detect_pure = True
    _appid = '0'

    @classmethod
    def read(cls):
        # Parse arguments for weight file
        config = ConfigParser.ConfigParser()
        config.read(osp.join(runPath(), CONFIG_PATH))

        # read appid
        app = config.get('app', 'NAME')
        if appid_dict.has_key(app):
            FDConfig._appid = appid_dict[app]

        # read mode
        if config.get(V_MODE_FIRST_KEY, V_MODE_SPLIT_SECOND_KEY) == 'ON':
            FDConfig._mode = V_MODE_SPLIT
        elif config.get(V_MODE_FIRST_KEY, V_MODE_FLUR_SECOND_KEY) == 'ON':
            FDConfig._mode = V_MODE_FLUR
        else:
            FDConfig._mode = V_MODE_SPLIT

        #read weight file and mean file
        firstkey = FDConfig._mode
        if FDConfig._mode in (V_MODE_SPLIT, V_MODE_FLUR):
            weight_file = config.get(firstkey, 'WEIGHTS_FILE')
            FDConfig._weight_file = osp.join(runPath(), weight_file)
            mean_file = config.get(firstkey, 'MEAN_FILE')
            FDConfig._mean_file = osp.join(runPath(), mean_file)
            logger.debug('[weight_file]:%s [mean_file]:%s'%(FDConfig._weight_file,FDConfig._mean_file))

        #read if detect pure
        FDConfig._if_detect_pure = (config.get(V_MODE_FIRST_KEY, V_MODE_PURE_SECOND_KEY) == 'ON')

        logger.info('You choose [%s] mode to run and read weight file: %s pure need detect: %s' % (FDConfig._mode, FDConfig._weight_file, FDConfig._if_detect_pure))

        FDConfig.CROP_TO_PREDICT = (FDConfig._mode == V_MODE_FLUR)  # crop pic before preidcting or not
        FDConfig.RESIZE_TO_PREDICT = (FDConfig._mode == V_MODE_SPLIT)  # resize pic before preidcting or not

        FDConfig.CROP_TO_PREDICT = 0
        FDConfig.RESIZE_TO_PREDICT = 0

        logger.info('[crop]:%d  [resize]:%d' % (FDConfig.CROP_TO_PREDICT, FDConfig.RESIZE_TO_PREDICT))


if __name__ == '__main__':
    FDConfig.read()
    print FDConfig._mean_file








