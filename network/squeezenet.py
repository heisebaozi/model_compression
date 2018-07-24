from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d,flatten,dropout,fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope


@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs):
    return conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_classes = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(is_training,
                                      self._weight_decay,
                                      self._batch_norm_decay)):
                return self._squeezenet(x, self._num_classes)

    @staticmethod
    def _squeezenet(images, num_classes=1000):
        net = conv2d(images,32, [3, 3], stride=2, scope='conv1')
        print(net.shape)
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = fire_module(net, 32, 128, scope='fire4')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9')

        net = flatten(net, scope='flatten')
        net = dropout(net, keep_prob=0.7, scope='dropout1')
        net = fully_connected(net, 128, activation_fn=None, scope='feature')
        feature = dropout(net, keep_prob=0.5, scope='dropout2')
        logit = fully_connected(feature, num_classes, activation_fn=None, scope='logits')

        return logit, feature

def _arg_scope(is_training, weight_decay, bn_decay):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay),
                   normalizer_fn=batch_norm,
                   normalizer_params={'is_training': is_training,
                                      'fused': True,
                                      'decay': bn_decay}):
        with arg_scope([fully_connected], weights_regularizer=l2_regularizer(weight_decay)):
            with arg_scope([dropout], is_training=is_training) as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
