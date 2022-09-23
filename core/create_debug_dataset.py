#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_debug_dataset.py
#   Author      : ronen halevy 
#   Created date:  9/20/22
#   Description :
#
# ================================================================
import tensorflow as tf

def load_debug_dataset():
    x_train = tf.image.decode_jpeg(
        open('datasets/coco2012/images/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train/255, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 1, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 1, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 1, 67]
    ] + [[0, 0, 0, 0, 0, 0]] * 97
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))