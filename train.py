#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : scratch.py
#   Author      : ronen halevy
#   Created date:  5/2/22
#   Description :
#
# ================================================================
import numpy as np
import tensorflow as tf
import argparse

from parse_tfrecords import parse_tfrecords

from preprocess_dataset import preprocess_dataset

from utils import render
from utils import generate_random_dataset


def split_dataset(dataset, dataset_size):
    train_size = int(0.7 * dataset_size)
    val_size = int(0.20 * dataset_size)
    test_size = int(0.10 * dataset_size)

    dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=None)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset
    from preprocess_dataset import preprocess_dataset


def config_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_debug_dataset', default=False, action='store_true')

    parser.add_argument('--render_dataset_example', default=False, action='store_true')

    parser.add_argument("--tfrecords_dir", type=str,
                        default='/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')

    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')

    parser.add_argument("--max_boxes", type=int, default=100,
                        help='max bounding boxes in an example image')

    parser.add_argument("--batch_size", type=int, default=32,
                        help='batch_size')

    parser.add_argument("--image_size", type=int, default=416,
                        help='Algorithm"s image_size . Assumed a square')

    parser.add_argument("--anchors_file", type=str, default=None,
                        help='anchors_file')

    parser.add_argument("--dataset_limit_size", type=str, default=100,
                        help='Train will limit dataset to dataset_size. If None, no limit is set')

    grid_sizes = np.array([13, 26, 52])

    args = parser.parse_args()
    tfrecords_dir = args.tfrecords_dir
    image_size = args.image_size
    batch_size = args.batch_size
    class_file = args.classes
    max_boxes = args.max_boxes
    use_debug_dataset = args.use_debug_dataset
    anchors_file = args.anchors_file
    dataset_limit_size = args.dataset_limit_size

    if use_debug_dataset:
        dataset = generate_random_dataset()
        anchors = np.array(
            [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]],
            np.float32) / image_size
    else:
        dataset = parse_tfrecords(tfrecords_dir, image_size, max_boxes, class_file)

        # anchors =
        # read_anchors(anchors_file) if anchors_file else \
        anchors = np.array(
            [[[20, 23], [10, 13], [16, 30]], [[30, 61], [62, 45], [59, 19]], [[116, 90], [156, 198], [373, 326]]],
            np.float32) / image_size

    dataset_size = dataset_limit_size if dataset_limit_size else dataset.cardinality().numpy()

    if args.render_dataset_example:
        image, y = list(dataset.as_numpy_iterator())[0]
        images = tf.expand_dims(image, axis=0)  # * 255
        render(images, y)

    return dataset, dataset_size, batch_size, image_size, anchors, max_boxes, grid_sizes


def train():
    dataset, dataset_size, batch_size, image_size, anchors, max_boxes, grid_sizes = config_train()
    train_dataset, test_dataset, val_dataset = split_dataset(dataset, dataset_size)
    dataset = preprocess_dataset(dataset, batch_size, image_size, anchors, grid_sizes, max_boxes)


if __name__ == '__main__':
    train()
