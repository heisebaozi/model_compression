#!/usr/bin/env python
"""
utils for compute network params counts
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

def count_training_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1 
        for dim in shape:
            variable_parameters *=dim.value
        total_parameters += variable_parameters
        return total_parameters

def get_model_size():
    total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    #model size is measure by MB.
    model_size = round((total_parameters * 4)/1024/1024,2)
    return total_parameters,model_size
