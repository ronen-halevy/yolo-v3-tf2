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
