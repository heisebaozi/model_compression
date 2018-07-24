#!/usr/bin/env python
#coding=utf-8

"""
train_val face classifier from TFRecords
This is a train framework including train code and test code. 
finetune use cloud data

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

reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append("/weibiao/mimic/network")
sys.path.append("/weibiao/mimic/metrics")
sys.path.append("/weibiao/mimic/utils")

# resnet networks from slim
from squeezenet import Squeezenet
from test_verification import test_verification
from mail_report import report

# Define argumentments
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("gpu", "0,1", "gpu")
tf.app.flags.DEFINE_integer("dim_features", 128, "dimension of feature")
tf.app.flags.DEFINE_integer("num_classes", 129896, "number of classes")
tf.app.flags.DEFINE_string("directory", "//weibiao/mimic/exp/exp04", "work directory")
tf.app.flags.DEFINE_string("dataset_dir", "/data/face", "dataset directory")
tf.app.flags.DEFINE_integer("num_epochs", 10000, "number of epoch")
tf.app.flags.DEFINE_integer("batch_size", 256, "training batch size")
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('weight_decay', 0.001, 'Weights of L2 loss.')
tf.app.flags.DEFINE_float('batch_norm_decay',0.9, 'bn norm decay')
tf.app.flags.DEFINE_integer('steps_to_log', 100, 'Steps to log and print loss')
tf.app.flags.DEFINE_integer('steps_to_save', 10000, 'Steps to save model')
tf.app.flags.DEFINE_integer('steps_to_test', 10000, 'Steps to test')
tf.app.flags.DEFINE_string("train_lists", "cloud_train_min_10_1_feature_wb.tfrecords,cloud_train_min_10_2_feature_wb.tfrecords", "train dataset list")
tf.app.flags.DEFINE_string("test_lists", "jdb_test.tfrecords", "test dataset list")
tf.app.flags.DEFINE_integer('num_test_features', 200000, 'number of features of the test dataset')

# ---------------------------------------------------------------------------
def _parse_function(example_proto):
    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'channel': tf.FixedLenFeature([], tf.int64),
               'train/label': tf.FixedLenFeature([], tf.int64),
               'train/image': tf.FixedLenFeature([], tf.string),
               'train/feature': tf.FixedLenFeature([],tf.string)}

    parsed_features = tf.parse_single_example(example_proto, features=feature)
    # decode the string to image
    rows = tf.cast(parsed_features['height'], tf.int32)
    cols = tf.cast(parsed_features['width'], tf.int32)
    chans = tf.cast(parsed_features['channel'], tf.int32)
    image = tf.decode_raw(parsed_features['train/image'], tf.float32)
    image = tf.reshape(image, [64, 64, 3])
    label = tf.cast(parsed_features['train/label'], tf.int32)
    big_net_feature = tf.decode_raw(parsed_features['train/feature'],tf.float32)
    big_net_feature = tf.reshape(big_net_feature,[128])
    return image, label, big_net_feature

def _parse_function_test(example_proto):
    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'channel': tf.FixedLenFeature([], tf.int64),
               'train/label': tf.FixedLenFeature([], tf.int64),
               'train/image': tf.FixedLenFeature([], tf.string)}

    parsed_features = tf.parse_single_example(example_proto, features=feature)
    # decode the string to image
    rows = tf.cast(parsed_features['height'], tf.int32)
    cols = tf.cast(parsed_features['width'], tf.int32)
    chans = tf.cast(parsed_features['channel'], tf.int32)
    image = tf.decode_raw(parsed_features['train/image'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(parsed_features['train/label'], tf.int32)
    fake_feature = tf.zeros([128])
    return image, label,fake_feature
# data enhancement
def _augment_function(image,label,feature):
    image_augmented = tf.image.random_flip_left_right(image)
    return image_augmented, label,feature

def _preprocess_function(image,label,feature):
    # resize the image to 64*64
    #image_resized =  tf.image.resize_images(image, (128, 128), 0)
    # image = tf.image.resize_images(image, (64, 64),method=0)
    image_processed = tf.cast(image, tf.float32)
    # image_processed = tf.cast(image, tf.float32) * (1.0 / 127.5) - 1
    return image_processed, label,feature
def _preprocess_function_test(image,label,feature):
    # resize the image to 64*64
    image = tf.image.resize_images(image, (64, 64),method=0)
    image_processed = tf.cast(image, tf.float32) * (1.0 / 127.5) - 1
    return image_processed, label,feature

# ---------------------------------------------------------------------------
#arc loss define

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(FLAGS.dim_features, FLAGS.num_classes), # (embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')
        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output

def main(unused_args):
    """Train face classifier for a number of epochs."""
    print('>>>>>>>>>>tensorflow version: %s<<<<<<<<<<<<<<<' % (tf.__version__))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    # logging config
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # datefmt='%a, %d %b %Y %H:%M:%S',
                        # filename='train.log',
                        filemode='w')

    with tf.Session() as sess:
        graph = tf.get_default_graph()
	# node_names = [node.name for node in tf.get_default_graph.as_graph_def().node]
	# for i in node_names:
	    
        # data process part.

        # global_step = tf.Variable(1, name='global_step', trainable=False)
        # parse dataset
        train_tfrecords_list = FLAGS.train_lists.split(",")
        test_tfrecords_list = FLAGS.test_lists.split(",")
        # join path
        train_dataset_path = [os.path.join('/data/face/dataset/', tfr) for tfr in train_tfrecords_list]
        test_dataset_path = [os.path.join('/data/face/dataset/', tfr) for tfr in test_tfrecords_list]
        # train dataset
        train_dataset = tf.data.TFRecordDataset(train_dataset_path)
        train_dataset = train_dataset.map(_parse_function)
        train_dataset = train_dataset.map(_augment_function)
        train_dataset = train_dataset.map(_preprocess_function)
        train_dataset = train_dataset.shuffle(buffer_size=50000) # random shuffle
        train_dataset = train_dataset.repeat(FLAGS.num_epochs)  # epoch
        train_dataset = train_dataset.batch(FLAGS.batch_size)  # batch size
        # test dataset
        test_dataset = tf.data.TFRecordDataset(test_dataset_path)
        test_dataset = test_dataset.map(_parse_function_test)
        test_dataset = test_dataset.map(_preprocess_function_test)
        test_dataset = test_dataset.repeat() # infinite epoch,terminated manully
        test_dataset = test_dataset.batch(FLAGS.batch_size) # batch size
        # handle, iterator, next_element

        #handle = tf.placeholder(tf.string,shape=[])
        #iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        
        train_iterator = train_dataset.make_one_shot_iterator() # one_shot
        batch_images, sparse_labels,big_net_features = train_iterator.get_next()
        batch_labels = tf.one_hot(tf.squeeze(sparse_labels), FLAGS.num_classes, dtype=tf.int32)
        test_iterator = test_dataset.make_initializable_iterator() # initializable
        test_batch_images, test_sparse_labels,_ = test_iterator.get_next()
        # train_handle = sess.run(train_iterator.string_handle())
        # test_handle = sess.run(test_iterator.string_handle()) 
      
        is_training_holder = tf.placeholder(tf.bool,name="is_training")
        
        NUM_CLASSES = FLAGS.num_classes
        NUM_FEATURE = FLAGS.dim_features
        batch_labels_holder = tf.placeholder(tf.int32, name="labels_holder",shape=[FLAGS.batch_size,FLAGS.num_classes])
        big_net_features_holder = tf.placeholder(tf.float32,shape=[FLAGS.batch_size,NUM_FEATURE])
        input_op = tf.placeholder(tf.float32, name='input_op', shape=[None,64,64,3])
	#--------------------net--------------------------------
        net = Squeezenet(FLAGS)
        _, mimic_feature = net.build(input_op, is_training=is_training_holder)

	#print (mimic_feature)
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=True)
        logit = arcface_loss(embedding=mimic_feature, labels=sparse_labels, w_init=w_init_method, out_num=FLAGS.num_classes)
        # print(logit)
        logit.set_shape([FLAGS.batch_size, FLAGS.num_classes])
        
        softmax_loss_arc = slim.losses.softmax_cross_entropy(logit, batch_labels_holder)
        softmax_loss_arc_reg = slim.losses.get_total_loss(add_regularization_losses=True)

        l2_loss_00 = tf.nn.l2_loss(mimic_feature - big_net_features_holder)/FLAGS.batch_size
        
        cost = 1 * softmax_loss_arc_reg + 0.1 * l2_loss_00

        lr = FLAGS.learning_rate

        optimizer = tf.train.GradientDescentOptimizer(lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(cost)
        
        correct_pred = tf.equal(tf.argmax(logit, 1), tf.argmax(batch_labels_holder,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # print weights info
        trainable_ws = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        regularization_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print('--------------------------------------------------------------------------------')
        print("trainable_variables:")
        for w in trainable_ws:
            shp = w.get_shape().as_list()
            print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
        print('--------------------------------------------------------------------------------')
        print("regularization_losses:")
        for w in regularization_ws:
            shp = w.get_shape().as_list()
            print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
        print('--------------------------------------------------------------------------------')
        
        # initializing
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # tensorboard
        tf.summary.scalar('accuracy', accuracy)
        # add the cost scalar to tensorboard.
        # only to monitor the loss function.
        tf.summary.scalar('loss',cost)
        tf.summary.scalar('l2_loss',l2_loss_00)
        tf.summary.scalar('arc_loss',softmax_loss_arc_reg)
        merged = tf.summary.merge_all()
        dir_path = FLAGS.directory
        
        summary_dir = os.path.join(dir_path, 'summary')
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        checkpoint_dir = os.path.join(FLAGS.directory, 'checkpoint')
        pbfile_dir = os.path.join(FLAGS.directory, "pbfiles")
        input_op = graph.get_tensor_by_name("input_op:0")

        """
#======================================================================================================
        # for finetune
        var_list={}
        all_variables = tf.all_variables()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
           # print("tensor_name: ", key)
           # print(reader.get_tensor(key))
            for v in all_variables:
                if (key == v.op.name) and (reader.get_tensor(key).shape == v.shape):
                    var_list[key]=v

        saver = tf.train.Saver(var_list)
#======================================================================================================
        """
        
        # save checkpoint
        saver = tf.train.Saver(max_to_keep=10)
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        step = 0
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        # start training networks
        duration = 0
        first_call = 1;
        while True:
            try:
                start_time = time.time()
                # print(sess.run([batch_images,batch_labels],feed_dict={handle: train_handle}))
                # run optimizer
                #cc ,dd = sess.run([batch_labels,logits],feed_dict={handle: train_handle, is_training_holder:True})
                #print(cc)
                #print(dd)
                image_array,labels_array, big_net_features_array = sess.run([batch_images,batch_labels, big_net_features])
                #print(batch_images)
                _, arc_loss_value, l2_loss_value, total_loss, train_acc,feat_val = sess.run([train_op, softmax_loss_arc_reg, l2_loss_00, cost, accuracy, mimic_feature],
                                                                                feed_dict={input_op: image_array,batch_labels_holder:labels_array,big_net_features_holder: big_net_features_array,is_training_holder:True})
                #_= sess.run([train_op], feed_dict={input_op: image_array,batch_labels_holder:labels_array,big_net_features_holder: big_net_features_array,is_training_holder:True})

                duration += time.time() - start_time
                # log
                if step % FLAGS.steps_to_log == 0:
                    logging.info('step %d: total loss = %.4f arc_loss = %.4f l2 loss = %.4f accuracy = %.4f (%.3f sec)' \
                                  % (step, total_loss, arc_loss_value, l2_loss_value, train_acc, duration))
                   # print(feat_val[0])
                    # add summary
                    result = sess.run(merged, feed_dict={input_op: image_array,batch_labels_holder:labels_array,big_net_features_holder: big_net_features_array,is_training_holder:True})
                    writer.add_summary(result, step)
                    duration = 0
                # save model
                if (step % FLAGS.steps_to_save == 0) and (step != 0) and (first_call == 0):
                    #saver = tf.train.Saver()
                    logging.info("Saving parameters to %s/model.ckpt-%d " % (checkpoint_dir, step))
                    saver.save(sess, checkpoint_dir + '/model.ckpt', global_step=step)
                    # save the pb file.
                    # graph_def = tf.get_default_graph().as_graph_def()
                    # output_graph_def = graph_util.convert_variables_to_constants(   
                    #             sess,
                    #             graph_def,
                    #             ["fc7"] #需要保存节点的名字
                    # )
                    # output_graph = pbfile_dir + "feature.pb"
                    # with tf.gfile.GFile(output_graph, "wb") as f:
                    #     f.write(output_graph_def.SerializeToString())
                    # print("%d ops in the final graph." % len(output_graph_def.node))

                # test
                
                if (step % FLAGS.steps_to_test == 0) and (step != 0) and (first_call == 0) :
                    batch = 0
                    feature_dic = {} # feature dictionary
                    feature_array = np.empty(shape=[0, FLAGS.dim_features], dtype=float)
                    logging.info('Start extracting features from jdb dataset...')
                    # initializing, start extracting features
                    sess.run(test_iterator.initializer)
                    while True:
                        batch += 1
                        start_time = time.time()
                       # mimic_feature_array = sess.run(mimic_feature, feed_dict={handle:train_handle})
                        test_image_array = sess.run(test_batch_images)
                        lfw_features = sess.run(mimic_feature, feed_dict={input_op: test_image_array, is_training_holder:False})
                        lfw_features = np.array(lfw_features)
                        duration = time.time() - start_time
                        # print(np.shape(lfw_features))
                        feature_array = np.concatenate((feature_array, np.squeeze(lfw_features)), axis=0)
                        num_iter = math.ceil(FLAGS.num_test_features / FLAGS.batch_size)
                        if batch % 10 == 0:
                            logging.info('Extracting batch %d(total batch %d): (%.3f sec/batch)' % (batch, num_iter, duration))
                        if batch > num_iter:
                            feature_array = feature_array[0:FLAGS.num_test_features, :]
                            if batch % 10 != 0:
                                logging.info('Extracting batch %d(total batch %d): (%.3f sec/batch)' % (batch, num_iter, duration)) 
                            logging.info('Successfully extract jdb features from jdb dataset!')        
                            break

                    feature_dic['feature'] = feature_array
                    feature_dir = os.path.join(FLAGS.directory, 'feature')
                    if not os.path.exists(feature_dir):
                        os.makedirs(feature_dir)
                    feature_path = os.path.join(FLAGS.directory, 'feature', 'jdb_features.mat')
                    # run this script in 103 machine.
                    pairs_path = os.path.join('/data/face', 'dataset', 'jdb_test_pairs.mat')
                    if not os.path.exists(pairs_path):
                        logging.info("error: %s not exists!" % (pairs_path))
                        break
                    logging.info('Features saved to %s!' % (feature_path))
                    sio.savemat(feature_path, feature_dic)
                    test_acc, th = test_verification(feature_path, pairs_path, 100000)
                    logging.info("step = %d, th = %f, acc = %f" % (step, th, test_acc))
                    
                   # mail report
                    date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    subject = 'mimic_squeezenet_wb: l2 + arc loss (' + date_time + '),'
                    content_map = {"network": "squeezenet", "train_dataset": "cloud dataset", "learning_rate": lr,
                                   "batch_size": FLAGS.batch_size,
                                   "step": step, "jdb_verification_accuracy": test_acc, "threshold": th,
                                   "l2_loss": l2_loss_value, "total_loss": total_loss,
                                   "arc_loss": arc_loss_value, "train_acc":train_acc}
                    receivers = ['']
                    report(content_map=content_map, subject=subject, receivers=receivers)
                    
                step += 1
		first_call = 0
            except tf.errors.OutOfRangeError:
                print('Done Training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
           # finally:
            #    break
  
# .............................................................................
if __name__ == "__main__":
    tf.app.run()


