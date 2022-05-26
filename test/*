#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : dataset_transformation_debug.py
#   Author      : ronen halevy 
#   Created date:  5/9/22
#   Description :
#
# ================================================================
import tensorflow as tf
import preprocess_dataset
from parse_tfrecords import parse_tfrecords
from matplotlib import pyplot as plt

import train
import preprocess_dataset
import utils

import numpy as np

tfrecords_dir = '/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords'
image_size = 416
max_bboxes = 100
class_file = None

dataset, dataset_size, batch_size, image_size, anchors, max_bboxes, grid_sizes = train.config_train()

image_batch, y_train = next(dataset.as_numpy_iterator())
y_train = tf.expand_dims(y_train, axis=0)
downsize_stride = 32
output_shape = [32,13,13,3,5]
dataset = preprocess_dataset.arrange_in_grid(y_train, anchors[0], downsize_stride, output_shape, max_bboxes)

pass

# def arrange_in_grid(y_train, anchors, downsize_stride, output_shape, max_bboxes)
