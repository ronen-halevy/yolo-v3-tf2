#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_dataset.py
#   Author      : ronen halevy 
#   Created date:  10/19/22
#   Description :
#
# ================================================================
import tensorflow as tf
from core.load_tfrecords import parse_tfrecords
from core.create_dataset_from_files import create_dataset_from_files
from core.load_tfrecords import parse_tfrecords


def load_debug_dataset(image_size):
    x_train = tf.image.decode_jpeg(
        open('datasets/coco2012/images/girl.png', 'rb').read(), channels=3)
    x_train = tf.image.resize(x_train / 255, [image_size, image_size])
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 1, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 1, 56],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 1, 67]
             ] + [[0, 0, 0, 0, 0, 0]] * 97
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)
    dataset_size = y_train.shape[0]
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return ds, dataset_size


def create_dataset(dataset_config, image_size, max_bboxes, classes_name_file, max_dataset_examples):
    dataset = [None] * 2
    dataset_size = [-1] * 2
    if dataset_config['input_data_source'] == 'tfrecords':
        for idx, split_name in enumerate(['train', 'valid']):
            tfrecord = dataset_config['tfrecords'][split_name]
            dataset[idx] = parse_tfrecords(
                tfrecord, image_size, max_bboxes, classes_name_file)

    elif dataset_config['input_data_source'] == 'coco_format_files':
        for idx, split_name in enumerate(['train', 'valid']):
            train_images_dir = dataset_config['coco_format_files'][split_name]['images_dir']
            annotations = dataset_config['coco_format_files'][split_name]['annotations']
            dataset[idx], dataset_size[idx] = create_dataset_from_files(train_images_dir,
                                                                                    annotations,
                                                                                    image_size,
                                                                                    max_dataset_examples,
                                                                                    max_bboxes=100)

    else:  # debug_data
        for idx in range(2):
            dataset[id], dataset_size[idx] = load_debug_dataset(image_size)

    return dataset, dataset_size
