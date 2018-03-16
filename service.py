# -*- coding: utf-8 -*-
# To kick off the script, run the following from the python directory:
#   PYTHONPATH=`pwd` python testdaemon.py start

# standard python libs
import time

# third party libs
from daemon import runner
import numpy as np
import QZHelper as qz
from classifymodule import ClassifyModule,ClassifyModule_Tensorflow
from other import pure_detect as puredetect
import json
from FDConfig import *
from DBHelper import get_pic_infos,report_result




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





def predict(model,image_paths):

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

    # 2. classiy flur or split
    probs = model.predict_imagepaths(image_paths)
    # display_results(image_paths, probs)

    # judge the result
    rst = qz.judgeRst(image_paths, probs, FDConfig._mode)
    rst_dict[FDConfig._mode] = rst

    # 3. detect pure frame
    if FDConfig._if_detect_pure:
        pure = puredetect.detect(image_paths[0:1])
        rst_dict[V_MODE_PURE] = pure[0]

    # 4. correct result
    rst_dict = qz.handle_final_result(image_paths,rst_dict)

    json_str = json.dumps(rst_dict)
    logger.info('[json result]'+json_str)

    # 5. delete pic
    qz.clear_tmp_files(image_paths)

    return rst_dict

class App():
    def __init__(self, name):
        self.stdin_path = '/dev/null'
        self.stdout_path = '/dev/null'
        self.stderr_path = '/dev/null'
        self.pidfile_path = osp.join(PID_PATH,name+'.pid')
        self.pidfile_timeout = 5
        self._model = None

    def run(self):
        while True:
            # Main code goes here ...
            if not isinstance(self._model,ClassifyModule_Tensorflow):
                #self._model = ClassifyModule(model_type='SplitAlexNet',model_data_path=FDConfig._weight_file, mean_file_path=FDConfig._mean_file)
                self._model = ClassifyModule_Tensorflow(model_data_path=FDConfig._weight_file,input_size=224)
                logger.warning('**************LOADING THE ML MODEL******************')

            try:
                main_task(self._model)
            except Exception as e:
                logger.error("[FrameDetect] get error:{}".format(e))
                exit(-1)



def main_task(model):
    image_infos = before_to_do()

    report_list = []
    for image_info in image_infos:
        if not image_info.has_key('url'):
            continue
        image_path = image_info['url']
        rst_dict = predict(model, [image_path])

        if not need_report(rst_dict):
            continue

        # update image info
        image_info = update_info(image_info, rst_dict)
        report_list.append(image_info)

    after_to_do(report_list)

    # control sleep time by requested image infos
    if not len(image_infos) > 5:
        time.sleep(10 - len(image_infos))

    # free memory
    del image_infos[:]
    del report_list[:]


def update_info(image_info,rst_dict):
    image_info['ret1'] = str(rst_dict[V_MODE_SPLIT])
    image_info['ret2'] = str(rst_dict[V_MODE_FLUR])
    image_info['ret3'] = str(rst_dict[V_MODE_PURE])
    # image_info['ret4'] = str(-1)
    # image_info['ret5'] = str(-1)
    return image_info

def need_report(rst_dict):
    if not isinstance(rst_dict,dict):
        return False
    for key in (V_MODE_SPLIT, V_MODE_FLUR, V_MODE_PURE):
        if rst_dict.has_key(key):
            if rst_dict[key] > 0:
                return True
    return False

def before_to_do():
    return get_pic_infos()


def after_to_do(report_list):
    report_result(report_list)



if __name__ == '__main__':

    # set logger
    ut.set_logger(logger)

    # read config
    FDConfig.read()

    argv = __import__('sys').argv
    name = 'default'
    if len(argv) >=3 :
        name = argv[2]

    #init app
    app = App(name)

    daemon_runner = runner.DaemonRunner(app)
    # This ensures that the logger file handle does not get closed during daemonization
    daemon_runner.daemon_context. files_preserve =[handler.stream]
    daemon_runner.do_action()

