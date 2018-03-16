# -*- coding: utf-8 -*-

import tensorflow as tf
import models
import dataset
import util
import time
import QZHelper as qz
from caffehelper import imloader,io
import numpy as np


OVER_SAMPLE = True
IMAGE_LOAD_TYPE = 'caffe'
OVER_SAMPLE_NUM = 2


def load_image_tensorflow(image_paths,spec):
    with tf.Session() as sesh:
        # Create an image producer (loads and processes images in parallel)
        image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        indices, input_images = image_producer.get(sesh)

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
    return input_images


def load_image_caffe(image_paths,mean_file_path,spec):
    img_ar = []
    for image_path in image_paths:
        img_ar.append(io.load_image(image_path))
    swap = (2, 1, 0) if spec.channels == 3 else None
    scale = spec.scale_size - 1
    dims = (spec.crop_size,spec.crop_size)

    my_io = imloader.CaffeIO(mean_file=mean_file_path,
                                channel_swap=swap,  # RGB通道与BGR
                                raw_scale=scale,  # 把图片归一化到0~1之间
                                image_dims=dims)  # 设置输入图片的大小
    input_images = my_io.get_image(inputs=img_ar, oversample=OVER_SAMPLE)
    return input_images


def load_image(image_paths,mean_file_path,spec):
    if IMAGE_LOAD_TYPE == 'caffe':
        return load_image_caffe(image_paths,mean_file_path,spec)
    else:
        return load_image_tensorflow(image_paths,spec)


'''Separate steps for loading picture and predicting
'''
def classify(model_data_path, mean_file_path, image_paths):
    '''Classify the given images using SplitAlexNet.'''

    util.log('--------start:%s-------------------------' % image_paths)
    t1 = time.time()

    # Get the data specifications for the SplitAlexNet model
    spec = models.get_data_spec(model_class=models.SplitAlexNet)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = models.SplitAlexNet({'data': input_node})
    #tf.reset_default_graph()
    with tf.Session() as sesh:
        # set to reuse the variables
        tf.get_variable_scope().reuse_variables()

        util.log('Loading the model start')
        net.load(model_data_path, sesh)

        t2 = time.time()

        util.log('Loading the model end %.3fs'% (t2-t1))

        util.log('Loading the images start')
        input_images = load_image(image_paths,mean_file_path,spec)
        t3 = time.time()
        util.log('Loading the images end %.3fs'% (t3-t2))

        # Perform a forward pass through the network to get the class probabilities
        util.log('judging image:%s'%image_paths)
        probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
        if IMAGE_LOAD_TYPE == 'caffe' and OVER_SAMPLE:
            probs = probs.reshape((len(probs) / OVER_SAMPLE_NUM, OVER_SAMPLE_NUM, -1))
            probs = probs.mean(1)
        t4 = time.time()
        del input_images  # free memory

    util.log('--------end 总耗时 %.3fs  加载模型耗时 %.3fs 加载图片耗时 %.3fs 预测耗时 %.3fs-----------------' % ((t4 - t1), (t2 - t1), (t3 - t2),(t4-t3)))
    return probs

""" Separate steps for loading picture and predicting
"""
def classify_new(model_data_path, mean_file_path, image_paths):
    '''Classify the given images using SplitAlexNet.'''

    util.log('--------start:%s-------------------------' % image_paths)
    t1 = time.time()

    # Get the data specifications for the SplitAlexNet model
    spec = models.get_data_spec(model_class=models.SplitAlexNet)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = models.SplitAlexNet({'data': input_node})
    # tf.reset_default_graph()
    with tf.Session() as sesh:
        # set to reuse the variables
        tf.get_variable_scope().reuse_variables()

        util.log('Loading the model start')
        net.load(model_data_path, sesh)

        t2 = time.time()

        util.log('Loading the model end %.3fs' % (t2 - t1))

        rst = 0
        for image_path in image_paths:
            input_image = load_image([image_path], mean_file_path, spec)

            probs = sesh.run(net.get_output(), feed_dict={input_node: input_image})

            if IMAGE_LOAD_TYPE == 'caffe' and OVER_SAMPLE:
                probs = probs.reshape((len(probs) / OVER_SAMPLE_NUM, OVER_SAMPLE_NUM, -1))
                probs = probs.mean(1)
            util.log('[predicting]:%s,%f,%f'%(image_path,probs[0,0],probs[0,1]))
            if isinstance(rst, int): rst = probs
            else:
                rst = np.concatenate((rst,probs),axis=0)
            del input_image
    t3 = time.time()
    util.log('--------end 总耗时 %.3fs  加载模型耗时 %.3fs 预测耗时 %.3fs-----------------' % (
    (t3 - t1), (t2 - t1), (t3 - t2)))

    return rst


def test():

    weight_file_path = ''
    mean_file_path = ''
    image_paths = ''

    # get picture if needed
    if 'http' in image_paths[0]:
        image_paths = [qz.getRemotePic(url=image_paths[0])]


    # Classify the image
    probs = classify(model_data_path=weight_file_path, image_paths=image_paths,mean_file_path=mean_file_path)
    rst = qz.judgeRst([],probs)
    util.log('[split result] %d,[0]: %.3f%% [1]: %.3f%%'%(rst,probs[0][0]*100.0,probs[0][1]*100.0))


if __name__ == '__main__':
    test()



