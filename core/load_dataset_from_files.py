#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : load_dataset.py
#   Author      : ronen halevy 
#   Created date:  7/6/22
#   Description :
#
# ================================================================

import tensorflow as tf
import yaml

def bytes_feature_list(value):
    value = [x.encode('utf8') for x in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def split_dataset(ds, ds_size, train_split, val_split):
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    if not train_size * val_size:
        print('Dataset too small for splits. Duplicating data for all splits')
        train_ds = val_ds = test_ds = ds
        train_size = val_size = ds_size
    else:
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds, train_size, val_size


def concat_box_obj_with_classes(bboxes_and_obj, classes):
    classes = tf.expand_dims(classes, axis=-1)
    bboxes_and_obj = tf.convert_to_tensor(bboxes_and_obj, tf.float32)
    y_train = tf.concat((bboxes_and_obj, tf.cast(classes, tf.float32)), axis=-1)
    return y_train


def concat_box_and_obj(bboxes, positive_objectiveness, pad_size):
    bboxes_and_obj = tf.concat((tf.convert_to_tensor(bboxes), positive_objectiveness), axis=-1)
    # pad_size = max_bboxes - len(bboxes)
    padded_bboxes_and_obj = tf.pad(bboxes_and_obj, ((0, pad_size), (0, 0)), 'constant')
    return padded_bboxes_and_obj


def decode_and_resize_image(filename, size, y):
    img_st = tf.io.read_file(filename)
    img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
    img = tf.cast(img_dec, tf.float32)
    img = tf.image.resize(img/255, [size, size])
    return img, y


def load_dataset(images_dir, annotations_path, classes_name_file, image_size=416, dataset_cuttof_size=None,
                 max_bboxes=100, train_split=0.7, val_split=0.2):
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        filename=classes_name_file, key_dtype=tf.string, key_index=0, value_dtype=tf.int64,
        value_index=tf.lookup.TextFileIndex.LINE_NUMBER, delimiter="\n"), default_value=0
    )
    with open(annotations_path) as file:
        annotations = yaml.safe_load(file)

    bboxes_and_obj_list = []
    classes_list = []
    image_filenames = []
    for image_entry in annotations:
        positive_objectiveness = tf.repeat(tf.Variable([[1.]]), repeats=len(image_entry['bboxes']), axis=0)
        pad_size = max_bboxes - len(image_entry['bboxes'])
        padded_bboxes_and_obj = concat_box_and_obj(image_entry['bboxes'], positive_objectiveness, pad_size)
        bboxes_and_obj_list.append(padded_bboxes_and_obj)
        entry_classes = image_entry['labels'] + pad_size * [
            'any_default']  # pad classes separately - with any string default e.g. 'any_default'
        classes_list.append(entry_classes)
        image_filenames.append(f'{images_dir}/{image_entry["image_filename"]}')

    classes_tensor = tf.convert_to_tensor(classes_list)
    classes_entries = class_table[classes_tensor]
    y_train = concat_box_obj_with_classes(bboxes_and_obj_list, classes_entries)

    if dataset_cuttof_size:
        y_train = y_train[:dataset_cuttof_size, ...]
        image_filenames = image_filenames[:dataset_cuttof_size]

    ds = tf.data.Dataset.from_tensor_slices((image_filenames, y_train))
    ds = ds.map(lambda x, y: decode_and_resize_image(x, image_size, y))
    ds_size = y_train.shape[0]
    train_ds, val_ds, test_ds, train_size, val_size = split_dataset(ds, ds_size, train_split, val_split)
    return train_ds, val_ds, test_ds, train_size, val_size # tf.data.Dataset.from_tensor_slices((x_train, y_train))


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


if __name__ == '__main__':
    images_dir = '/datasets/shapes/three_circles/input/images_and_annotations_file/images/'
    annotations_path = '/datasets/shapes/three_circles/input/images_and_annotations_file/annotations/annotations.json'

    load_dataset(images_dir, annotations_path)
    pass
