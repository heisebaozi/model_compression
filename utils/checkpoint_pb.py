#!/usr/bin/env python
#coding=utf-8

"""
this files function is: change checkpoint file to pb file. 
checkpoint file is temp save in the train process.
pb file is used to predict or get feature.  
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt

import argparse
import logging

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
import shutil
import cv2  

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append("/network")
sys.path.append("/metrics")
sys.path.append("/utils")

# resnet networks from slim
import face_net as face_net
from test_verification import test_verification
from mail_report import report

pb_model_dir = "weibiao/mimic/exp/exp04/pbfiles"
pb_model_name = "feature.pb"
if not tf.gfile.Exists(pb_model_dir): #创建目录
    tf.gfile.MakeDirs(pb_model_dir)


def freeze_graph(model_folder):
    print("start to freeze the model.")
    checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    output_graph = os.path.join(pb_model_dir, pb_model_name) #PB模型保存路径

    output_node_names = "squeezenet/feature/BiasAdd" #原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph() #获得默认的图
    #input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图
    
#    for op in graph.get_operations():
#        print(op.name)
    with tf.Session() as sess:
        gd = sess.graph.as_graph_def()
        for node in gd.node:            
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in xrange(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        #print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            gd,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())
def load_graph(frozen_graph_filename):  
    # We parse the graph_def file  
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
  
    # We load the graph_def in the default graph  
    with tf.Graph().as_default() as graph:  
        tf.import_graph_def(  
            graph_def,   
            input_map=None,   
            return_elements=None,   
            #name="prefix",   
            op_dict=None,   
            producer_op_list=None  
        )  
    return graph  

def consin_distance(vectors1, vectors2):  
     
    vec1 = vectors1  
    vec2 = vectors2  
    vec0 = np.dot(vec1, vec2)  
    vec1 = np.power(vec1, 2)  
    vec2 = np.power(vec2, 2)  
    vec1 = np.sum(vec1)  
    vec2 = np.sum(vec2)  
    vec1 = np.sqrt(vec1)  
    vec2 = np.sqrt(vec2)  
    sim  = (vec0/(vec1 * vec2))  
    return sim

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_decoded.set_shape([640,512,3])
    image_cropped = tf.image.crop_to_bounding_box(image_decoded,160,96,320,320)
    image_resized = tf.image.resize_images(image_cropped, (224, 224), 0)
    image_processed = tf.cast(image_resized, tf.float32) * (1.0 / 255) - 0.5
    return image_processed, filename

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    checkpoint_dir = "weibiao/mimic/exp/exp04/checkpoint"
    freeze_graph(checkpoint_dir)
    


