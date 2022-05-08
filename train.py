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



def main():
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


    args = parser.parse_args()
    tfrecords_dir = args.tfrecords_dir
    image_size = args.image_size
    batch_size = args.batch_size
    class_file = args.classes
    max_boxes = args.max_boxes
    use_debug_dataset = args.use_debug_dataset
    anchors_file = args.anchors_file

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
            [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]],
            np.float32) / image_size

    # dataset_size = len(list(dataset))
    dataset_size = 100 # ronen debug

    train_size = int(0.7 * dataset_size)
    val_size = int(0.20 * dataset_size)
    test_size = int(0.10 * dataset_size)

    dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=None)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    if args.render_dataset_example:

        render(train_dataset)

    ###############################
    image_size = 416
    # downsize_strides = [32, 16, 8]
    grid_sizes = np.array([13, 26, 52])

    # anchors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    #
    batch_size = 32

    from preprocess_dataset import preprocess_dataset
    dataset = preprocess_dataset(dataset, batch_size, image_size, anchors, grid_sizes, max_boxes)
    pass
    # render(dataset)
    pass

    # scale_dataset = arrange_in_grid(y_train, anchors, image_size)
    # print(scale_dataset.size())
    # tensors = [scale_dataset.read(i) for i in range(scale_dataset.size().numpy())]
    # print(tensors)

    # downsize_strides = [32, 16, 8]
    # grid_sizes = [13, 26, 52]
    # anchors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    #
    # batch_size = 32
    # image_size = 416



    # train_dataset = preprocess_dataset(train_dataset, batch_size, image_size, anchors)
    # test_dataset = preprocess_dataset(test_dataset, batch_size, image_size, anchors)
    # val_dataset = preprocess_dataset(val_dataset, batch_size, image_size, anchors)
    #


if __name__ == '__main__':
    main()
