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
import json
import numpy as np

def bytes_feature_list(value):
    value = [x.encode('utf8') for x in value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def decode_and_resize_image(filename, size, y):
    img_st = tf.io.read_file(filename)
    img_dec = tf.image.decode_image(img_st, channels=3, expand_animations=False)
    img = tf.cast(img_dec, tf.float32)
    # resize w/o keeping aspect ratio - no prob for normal sized images
    img = tf.image.resize(img/255, [size, size])
    return img, y


def create_example(image_annotations, sparse_to_dense_category_id, max_bboxes, im_width, im_height):

    bboxes = []
    categories = []
    for annot in image_annotations:
        bboxes.append(annot['bbox'])
        categories.append(sparse_to_dense_category_id[annot['category_id']])
    scale = np.array([im_width, im_height, im_width, im_height])
    bboxes = np.array(bboxes) / scale

    positive_objectiveness = tf.repeat(tf.Variable([[1.]]), repeats=bboxes.shape[0], axis=0)
    pad_size = max_bboxes - bboxes.shape[0]

    bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)

    # rearrange_bbox_xywh_to_xyxy:
    xmin, ymin, w, h = tf.split(bboxes, [1,1,1,1], axis=-1)
    bboxes = tf.concat([xmin, ymin, xmin+w, ymin+h], axis=-1)

    categories = tf.expand_dims(tf.cast(categories, tf.float32), axis=-1)
    example = tf.concat([bboxes, positive_objectiveness, categories], axis=-1)
    padded_example = tf.pad(example, ((0, pad_size), (0, 0)), 'constant')

    return padded_example


def create_dataset_from_coco_files(images_dir, annotations_path, image_size, max_dataset_examples, max_bboxes=100):

    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    annotations_list = annotations['annotations']
    categories_list = annotations['categories']
    sparse_to_dense_category_id = {category['id']: index for index, category in enumerate(categories_list)}
    num_examples = min(len(annotations['images']), max_dataset_examples or float('inf'))
    images_list = annotations['images'][0:num_examples]

    y_train = []
    image_paths = []

    for image_entry in images_list:
        im_height = image_entry['height']
        im_width = image_entry['width']

        image_annotations = [annot for annot in annotations_list if annot['image_id'] == image_entry['id']]
        image_path = images_dir + '/' + image_entry['file_name']
        image_paths.append(image_path)
        example = create_example(image_annotations, sparse_to_dense_category_id, max_bboxes, im_width, im_height) # limit to max or pad to max anyway
        y_train.append(example)
    y_train = tf.convert_to_tensor(y_train)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, y_train))
    ds = ds.map(lambda x, y: decode_and_resize_image(x, image_size, y))
    ds_size = y_train.shape[0]
    return ds, ds_size


if __name__ == '__main__':
    images_dir = 'datasets/Oxford Pets.v1-by-breed.coco/train' # /home/ronen/fiftyone/coco-2017/validation/data # datasets/coco2012/images
    annotations_path: 'datasets/Oxford Pets.v1-by-breed.coco/train/_annotations.coco.json' # /home/ronen/fiftyone/coco-2017/validation/labels.json # datasets/coco2012/annotations.json

    create_dataset_from_coco_files(images_dir, annotations_path)

