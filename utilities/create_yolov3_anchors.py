#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : create_yolov3_anchors.py
#   Author      : ronen halevy 
#   Created date:  5/3/22
#   Description :
#
# ================================================================

import numpy as np
from sklearn.cluster import KMeans
import argparse
import pathlib
import os

from core.load_tfrecords import parse_tfrecords


# def plot_scatter_graph(w_h, kmeans):
#     image_width = 2 * max(w_h[..., 0])
#     image_height = 2 * max(w_h[..., 1])
#
#     plt.scatter(w_h[..., 0], w_h[..., 1], c=kmeans.labels_, alpha=0.5)
#     plt.xlabel("width")
#     plt.ylabel("height")
#     plt.title('K-means Clustering of Boxes Widths and Heights')
#     plt.figure(figsize=(10, 10))
#     plt.show()
def sort_anchors(anchors):
    anchors_sorted = anchors[(anchors[:, 0] * anchors[:, 1]).argsort()]
    return anchors_sorted


def arrange_wh_array(labels):
    w_h_tuples = (labels[..., 2] - labels[..., 0], labels[..., 3] - labels[..., 1])
    widths = np.expand_dims(w_h_tuples[0], axis=-1)
    heights = np.expand_dims(w_h_tuples[1], axis=-1)
    w_h = np.concatenate([widths, heights], axis=-1)
    mask = np.all(w_h != [0., 0.], axis=-1)
    w_h = w_h[mask]


import tensorflow as tf


def arrange_wh_array(labels):
    w_h_tuples = (labels[..., 2] - labels[..., 0], labels[..., 3] - labels[..., 1])
    widths = tf.expand_dims(w_h_tuples[0], axis=-1)
    heights = tf.expand_dims(w_h_tuples[1], axis=-1)
    w_h = tf.concat([widths, heights], axis=-1)
    mask = tf.reduce_all(w_h != [0., 0.], axis=-1)
    w_h = w_h[mask]
    return w_h


def creat_yolo_anchors(dataset):
    w_h = dataset.map(lambda _, labels: (
        arrange_wh_array(labels)
    ))
    w_h = w_h.unbatch()
    # Still, though it hearts...switching to numpy to use sklearn KMeans
    w_h = list(w_h.as_numpy_iterator())
    kmeans = KMeans(n_clusters=9)
    kmeans.fit(w_h)
    anchors = kmeans.cluster_centers_
    sorted_anchors = sort_anchors(anchors).astype(np.float32)
    return sorted_anchors


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecords_dir", type=str,
                        default='datasets/shapes/three_circles/input/tfrecords/train',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')
    parser.add_argument("--classes", type=str,
                        default='dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')
    parser.add_argument("--max_bboxes", type=int, default=100,
                        help='max bounding boxes in an example image')

    parser.add_argument("--image_size", type=int, default=416,
                        help='image_size assumed a square')

    parser.add_argument("--anchors_out_file", type=str, default='datasets/shapes/anchors/shapes_yolov3_anchors.txt',
                        help='image_size assumed a square')

    args = parser.parse_args()
    tfrecords_dir = args.tfrecords_dir
    max_bboxes = args.max_bboxes
    image_size = args.image_size
    anchors_out_file = args.anchors_out_file

    dataset = parse_tfrecords(tfrecords_dir, image_size, max_bboxes, class_file=None)
    anchors = creat_yolo_anchors(dataset)
    head, tail = os.path.split(anchors_out_file)
    pathlib.Path(head).mkdir(parents=True, exist_ok=True)

    np.savetxt(anchors_out_file, anchors, delimiter=',', fmt='%10.5f')
    print(f'result anchors:\n{anchors}')


if __name__ == '__main__':
    main()

# print('selected anchoanchors:\n', anchors)
#
#
# image_width = 2 * max(w_h[..., 0])
# image_height = 2 * max(w_h[..., 1])
#
# plt.scatter(w_h[..., 0], w_h[..., 1], c=kmeans.labels_, alpha=0.5)
# plt.xlabel("width")
# plt.ylabel("height")
# plt.title('K-means Clustering of Boxes Widths and Heights')
# plt.figure(figsize=(10, 10))
# plt.show()
#
#
# fig, ax = plt.subplots(figsize=(10, 10))
# c_w, c_h = image_width / 2, image_height / 2
#
# for index, anchor in enumerate(anchors):
#   rectangle = plt.Rectangle((c_w - anchor[0] / 2, c_h - anchor[1] / 2), anchor[0], anchor[1], linewidth=3,
#                                   edgecolor=list(np.random.choice(range(255), size=3) / 255)
#                                   , facecolor='none')
# ax.add_patch(rectangle)
# ax.set_aspect(1.0)
# plt.axis([0, image_width, 0, image_height])
# plt.xlabel("width")
# plt.ylabel("height")
# plt.title('Resultant 9 Anchor Boxes')
# plt.show()
