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
from core.load_tfrecords import parse_tfrecords
from core.create_dataset_from_coco_files import create_dataset_from_coco_files
from core.load_tfrecords import parse_tfrecords


def load_debug_dataset(image_size, batch_size):
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
    ds = ds.repeat(batch_size)  # repeats permit faster train execution due to batching
    return ds, dataset_size


def get_data_from_tfrecords(train_tfrecords, val_tfrecords, image_size, max_bboxes, classes_name_file):
    dataset = []
    tfrecords_dir_train = f'{train_tfrecords}'
    tfrecords_dir_val = f'{val_tfrecords}'
    for ds_dir in [tfrecords_dir_train, tfrecords_dir_val]:
        dset = parse_tfrecords(
            ds_dir, image_size, max_bboxes, classes_name_file)
        dataset.append(dset)
    return dataset


def create_dataset(dataset_config, image_size, max_dataset_examples, dataset_repeats):
    dataset = []
    if dataset_config['input_data_source'] == 'tfrecords':
        dataset = self.get_data_from_tfrecords(dataset_config['tfrecords']['train'],
                                               dataset_config['tfrecords']['valid'],
                                               image_size, max_bboxes,
                                               classes_name_file)
    elif dataset_config['input_data_source'] == 'coco_format_files':
        dataset_size = []
        for ds_split_config in (dataset_config['coco_format_files']['train'],
                                dataset_config['coco_format_files']['valid']):
            ds_split, ds_split_size = create_dataset_from_coco_files(ds_split_config['images_dir'],
                                                                     ds_split_config['annotations'],
                                                                     image_size,
                                                                     max_dataset_examples,
                                                                     max_bboxes=100
                                                                     )

            dataset.append(ds_split)
            dataset_size.append(ds_split_size)
        # Modify batch size, to avoid an empty dataset after batching with drop_remainder=True:
        batch_size = min((batch_size, min(dataset_size)))
    else:  # debug_data
        train_dataset, dataset_size = load_debug_dataset(image_size, batch_size)
        val_dataset, dataset_size = load_debug_dataset(image_size, batch_size)
        dataset = [train_dataset, val_dataset]

    if dataset_repeats:  # repeat train and val
        dataset[0] = dataset[0].repeat(dataset_repeats)  # train
        dataset[1] = dataset[1].repeat(dataset_repeats)  # val
    return dataset
