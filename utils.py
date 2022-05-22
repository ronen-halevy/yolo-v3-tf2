#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : utils.py
#   Author      : ronen halevy 
#   Created date:  5/3/22
#   Description :
#
# ================================================================

import tensorflow as tf

def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./girl.png', 'rb').read(), channels=3)
    x_train = tf.cast(tf.expand_dims(x_train, axis=0) , tf.float32) / 255

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 97
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

def generate_random_dataset(dataset_size=300, image_h=416, image_w=416, max_boxes=100, classes=80):
    '''

    :param dataset_size:
    :type dataset_size:
    :param image_h:
    :type image_h:
    :param image_w:
    :type image_w:
    :param max_boxes:
    :type max_boxes:
    :param classes:
    :type classes:
    :return:
    :rtype:
    '''
    images = tf.zeros([dataset_size, image_h, image_w, 3])
    xy_min = tf.random.uniform(shape=[dataset_size, max_boxes, 2], minval=0., maxval=0.8, dtype=tf.float32, seed=42)
    x_max = tf.random.uniform(shape=[dataset_size, max_boxes, 1], minval=xy_min[:, :, 0:1], maxval=1, dtype=tf.float32,
                              seed=42)
    y_max = tf.random.uniform(shape=[dataset_size, max_boxes, 1], minval=xy_min[:, :, 1:2], maxval=1, dtype=tf.float32,
                              seed=42)
    objectivenss = tf.ones(shape=[dataset_size, max_boxes, 1],dtype=tf.float32)
    boxes = tf.concat([xy_min, x_max, y_max, objectivenss], axis=-1)
    classes = tf.random.uniform(shape=[dataset_size, max_boxes, 1], minval=0, maxval=classes-1, dtype=tf.int32, seed=42)
    classes = tf.cast(classes, tf.float32)
    labels = tf.concat([boxes, classes], -1)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def render_bboxes(image, y):
    '''

    :param image:
    :type image:
    :param y:
    :type y:
    :return:
    :rtype:
    '''

    boxes = y[..., 0:4].astype(float)  # / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    indices = [1, 0, 3, 2]
    boxes = boxes[..., indices]
    boxes = boxes[0:10]
    boxes = tf.expand_dims(boxes, axis=0)
    boxes = tf.cast(boxes, tf.float32) / 416
    colors = [[1, 255, 0]]

    image1 = tf.image.draw_bounding_boxes(
        image * 255, boxes, colors, name=None
    )

    return image1[0] / 255
