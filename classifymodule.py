# -*- coding: utf-8 -*-

import tensorflow as tf
import models
import math
import time
from caffehelper import imloader,io
import numpy as np
from FDConfig import logger
import os.path as osp
import cv2





PREDICT_ITER_STEP = 4



def model_class(model_name):
    module = __import__('models')
    model_class = getattr(module, model_name)
    return model_class

class ClassifyModule_Tensorflow():
    """
    :param 
        model_type---model name, defined in MODELS = (AlexNet, CaffeNet, GoogleNet, NiN, ResNet50, ResNet101, ResNet152, VGG16, SplitAlexNet, FT_VGG)
        model_data_path----weight file path of a trained model
        mean_file_path----the path of mean file which handle before loading image
    """

    def __init__(self, model_data_path, input_size):
        self._model_data_path = model_data_path

        self._input_size = input_size

        # tf.reset_default_graph()
        self._sess = tf.Session(graph=self.load_model_graph())

        self._input_tensor = self._sess.graph.get_tensor_by_name("input:0")  # get input tensor
        self._output_tensor = self._sess.graph.get_tensor_by_name("MobilenetV1/Predictions/Reshape_1:0")  # get output tensor

    def __del__(self):
        self._sess.close()

    def load_model_graph(self):
        logger.debug('Loading the model start')
        t1 = time.time()

        # set to reuse the variables
        tf.get_variable_scope().reuse_variables()

        with tf.gfile.GFile(self._model_data_path) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name='')

        t2 = time.time()
        logger.debug('Loading the model end %.3fs' % (t2 - t1))
        return graph

    def predict_imagepaths(self, image_paths):
        logger.debug('predict image start')
        t1 = time.time()
        rst = []
        iter_num = int(math.ceil(len(image_paths)/float(PREDICT_ITER_STEP)))
        for i in range(iter_num):
            image_path = image_paths[i*PREDICT_ITER_STEP:(i+1)*PREDICT_ITER_STEP]
            input_image = self.load_image(image_path)
            probs = self._sess.run(self._output_tensor, feed_dict={self._input_tensor: input_image})
            for image_i in range(len(image_path)):
                logger.info('[predicting]:{},{}'.format(image_path[image_i], probs[image_i, :]))
            if len(rst) == 0:
                rst = probs
            else:
                rst = np.concatenate((rst, probs), axis=0)
            del input_image

        t2 = time.time()
        logger.debug('predict image end 耗时 %.3fs' % (t2 - t1))
        return rst

    def load_image(self, image_paths):
        input_images = []
        for image_path in image_paths:
            image_np = cv2.imread(image_path)
            image_np = 2 * (image_np / 255.) - 1
            image_np = cv2.resize(image_np, (self._input_size, self._input_size))
            input_images.append(image_np)
        return input_images

class ClassifyModule():
    """
    :param 
        model_type---model name, defined in MODELS = (AlexNet, CaffeNet, GoogleNet, NiN, ResNet50, ResNet101, ResNet152, VGG16, SplitAlexNet, FT_VGG)
        model_data_path----weight file path of a trained model
        mean_file_path----the path of mean file which handle before loading image
    """

    def __init__(self,model_type, model_data_path, mean_file_path=None, oversample = 2, oversample_num = 10):
        self._model_data_path = model_data_path

        self._oversample = oversample
        self.config_oversample_num(oversample,oversample_num)

        self._mean_file_path = mean_file_path if mean_file_path and len(mean_file_path) > 0 and osp.exists(mean_file_path) else None
        # tf.reset_default_graph()
        self._sesh = tf.Session()
        # Get the data specifications for the SplitAlexNet model
        self._spec = models.get_data_spec(model_class=model_class(model_type))
        # Create a placeholder for the input image
        self._input_node = tf.placeholder(tf.float32,
                                    shape=(None, self._spec.crop_size, self._spec.crop_size, self._spec.channels))
        # Get the net structure
        self._net = model_class(model_type)({'data': self._input_node})

        # Get the io to load pic after handle with mean file
        self.config_caffe_io()

        self.load_model_sesh()

    def __del__(self):
        self._sesh.close()

    def config_oversample_num(self,oversample, oversample_num):
        if oversample == 1:
            self._oversample_num = 10
        elif oversample == 2:
            self._oversample_num = 2
        elif oversample == 3:
            self._oversample_num = oversample_num

    def config_caffe_io(self):
        swap = (2, 1, 0) if self._spec.channels == 3 else None
        scale = self._spec.scale_size - 1
        dims = (self._spec.crop_size, self._spec.crop_size)
        self._io = imloader.CaffeIO(mean_file=self._mean_file_path,
                                    channel_swap=swap,  # RGB通道与BGR
                                    raw_scale=scale,  # 把图片归一化到0~1之间
                                    image_dims=dims)  # 设置输入图片的大小

    def load_model_sesh(self):
        logger.debug('Loading the model start')
        t1 = time.time()

        # set to reuse the variables
        tf.get_variable_scope().reuse_variables()

        self._net.load(self._model_data_path, self._sesh)

        t2 = time.time()
        logger.debug('Loading the model end %.3fs' % (t2 - t1))

    def predict_imagepaths(self, image_paths):
        logger.debug('predict image start')
        t1 = time.time()
        rst = 0
        iter_num = int(math.ceil(len(image_paths)/float(PREDICT_ITER_STEP)))
        for i in range(iter_num):
            image_path = image_paths[i*PREDICT_ITER_STEP:(i+1)*PREDICT_ITER_STEP]
            input_image = self.load_image(image_path)
            probs = self._sesh.run(self._net.get_output(), feed_dict={self._input_node: input_image})
            if self._oversample:
                probs = probs.reshape((len(probs) / self._oversample_num, self._oversample_num, -1))
                probs = probs.mean(1)
            for image_i in range(len(image_path)):
                logger.info('[predicting]:{},{}'.format(image_path[image_i], probs[image_i, :]))
            if isinstance(rst, int):
                rst = probs
            else:
                rst = np.concatenate((rst, probs), axis=0)
            del input_image

        t2 = time.time()
        logger.debug('predict image end 耗时 %.3fs' % (t2 - t1))
        return rst

    def load_image(self, image_paths):
        img_ar = []
        for image_path in image_paths:
            img_ar.append(io.load_image(image_path))
        input_images = self._io.get_image(inputs=img_ar, oversample=self._oversample, oversample_num= self._oversample_num)
        del img_ar[:]
        return input_images

if __name__ == '__main__':
    print model_class()



