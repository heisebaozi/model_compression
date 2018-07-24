#!/usr/bin/env python
#encoding:utf-8

import os
from tensorflow.python import pywrap_tensorflow

checkpoint_dir = "/"

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

print ckpt
exit()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
checkpoint_path = os.path.join(model_dir, "model.ckpt")
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
