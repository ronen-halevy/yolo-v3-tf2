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
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os



def load_fake_dataset(render_dataset_example=False):
    x_train = tf.image.decode_jpeg(
        open('./girl.png', 'rb').read(), channels=3)
    x_train = tf.cast(tf.expand_dims(x_train, axis=0), tf.float32) / 255

    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
             ] + [[0, 0, 0, 0, 0]] * 97
    if render_dataset_example:
        render_bboxes(x_train, labels)

    y_train = tf.convert_to_tensor(labels, tf.float32)

    render_bboxes(x_train, y_train.numpy())

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


def generate_random_dataset(dataset_size=300, image_h=416, image_w=416, max_bboxes=100, classes=80):
    '''

    :param dataset_size:
    :type dataset_size:
    :param image_h:
    :type image_h:
    :param image_w:
    :type image_w:
    :param max_bboxes:
    :type max_bboxes:
    :param classes:
    :type classes:
    :return:
    :rtype:
    '''
    images = tf.zeros([dataset_size, image_h, image_w, 3])
    xy_min = tf.random.uniform(shape=[dataset_size, max_bboxes, 2], minval=0., maxval=0.8, dtype=tf.float32, seed=42)
    x_max = tf.random.uniform(shape=[dataset_size, max_bboxes, 1], minval=xy_min[:, :, 0:1], maxval=1, dtype=tf.float32,
                              seed=42)
    y_max = tf.random.uniform(shape=[dataset_size, max_bboxes, 1], minval=xy_min[:, :, 1:2], maxval=1, dtype=tf.float32,
                              seed=42)
    objectivenss = tf.ones(shape=[dataset_size, max_bboxes, 1], dtype=tf.float32)
    boxes = tf.concat([xy_min, x_max, y_max, objectivenss], axis=-1)
    classes = tf.random.uniform(shape=[dataset_size, max_bboxes, 1], minval=0, maxval=classes - 1, dtype=tf.int32,
                                seed=42)
    classes = tf.cast(classes, tf.float32)
    labels = tf.concat([boxes, classes], -1)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset


def render_bboxes(image, bboxes):
    bboxes = tf.cast(bboxes, tf.float32)[tf.newaxis, :,:]
    bboxes, _ = tf.split(bboxes, [4, 1], axis=-1 )
    indices = [1, 0, 3, 2]
    bboxes = tf.gather(bboxes, indices, axis=-1)
    colors = [[1, 255, 0]]
    image = tf.image.draw_bounding_boxes(
        image, bboxes, colors, name=None
    )
    plt.imshow(image[0])
    plt.show()

# def render_bboxes1(image, y):
#     '''
#
#     :param image:
#     :type image:
#     :param y:
#     :type y:
#     :return:
#     :rtype:
#     '''
#
#     bboxes = y[..., 0:4].astype(float)  # / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
#     indices = [1, 0, 3, 2]
#     bboxes = bboxes[..., indices]
#     bboxes = bboxes[0:10]
#     bboxes = tf.expand_dims(bboxes, axis=0)
#     bboxes = tf.cast(bboxes, tf.float32)
#     colors = [[1, 255, 0]]
#
#     image = tf.image.draw_bounding_boxes(
#         image, bboxes, colors, name=None
#     )
#     plt.imshow(image[0])
#     plt.show()
#     return image[0] / 255


def load_sample_dataset(annotations_path, class_names_path, max_bboxes, render_dataset_example):

    with open(annotations_path) as f:
        annotations = json.load(f)

    image_file = annotations['image_filename']
    head, tail = os.path.split(annotations_path)
    image_path = f'{head}/{image_file}'
    x_train = tf.image.decode_jpeg(tf.io.read_file(image_path))
    x_train = tf.cast(tf.expand_dims(x_train, axis=0), tf.float32) / 255

    bboxes = annotations['bboxes']
    labels = [annotation['label'] for annotation in annotations['objects']]


    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        filename=class_names_path, key_dtype=tf.string, key_index=0, value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), default_value=-1)

    labels = tf.convert_to_tensor(labels, tf.string)
    labels = tf.cast(class_table.lookup(labels), tf.float32)[:, tf.newaxis]
    bboxes = tf.convert_to_tensor(bboxes, tf.float32)
    bboxes = tf.concat([bboxes, labels], axis=-1)

    nbboxes = bboxes.shape[0]
    npad_boxes = max(max_bboxes - nbboxes, 0)
    pad_boxes = tf.convert_to_tensor([[0., 0., 0., 0., 0.]] * npad_boxes)
    bboxes = tf.concat([bboxes, pad_boxes], axis = 0)
    # bboxes = bboxes + [[0, 0, 0, 0, 0]] * npad_boxes

    # y_train = tf.convert_to_tensor(bboxes, tf.float32)
    y_train = tf.expand_dims(bboxes, axis=0)

    if render_dataset_example:
        render_bboxes(x_train, bboxes)


    return tf.data.Dataset.from_tensor_slices((x_train, y_train))
