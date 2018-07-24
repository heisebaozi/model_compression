#!/usr/bin/env python
#encoding:utf-8
"""
freezing model from checkpoint file to pb file. 

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

reload(sys)
sys.setdefaultencoding('utf-8')

from tensorflow.python.framework.graph_util import convert_variables_to_constants


