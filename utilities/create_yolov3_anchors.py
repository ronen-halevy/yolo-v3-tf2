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
# TODO - set in descending order!
import tensorflow as tf
import yaml


import numpy as np
from sklearn.cluster import KMeans
import argparse
import pathlib
import os

from core.load_tfrecords import parse_tfrecords
from core.create_dataset_from_files import create_dataset_from_files


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
    widths = tf.expand_dims(w_h_tuples[0], axis=-1)
    heights = tf.expand_dims(w_h_tuples[1], axis=-1)
    w_h = tf.concat([widths, heights], axis=-1)
    mask = tf.reduce_all(w_h != [0., 0.], axis=-1)
    w_h = w_h[mask]
    return w_h


def creat_yolo_anchors(dataset, n_clusters):
    w_h = dataset.map(lambda _, labels: (
        arrange_wh_array(labels)
    ))
    w_h = w_h.unbatch()
    # Still, though it hearts...switching to numpy to use sklearn KMeans
    w_h = list(w_h.as_numpy_iterator())
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(w_h)
    anchors = kmeans.cluster_centers_
    sorted_anchors = sort_anchors(anchors).astype(np.float32)
    return sorted_anchors


def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='create_anchors_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        config_file = yaml.safe_load(stream)




    n_clusters = config_file['n_clusters']

    input_data_source = config_file['input_data_source']


    # tfrecords_dir = config_file['tfrecords_dir']
    # limit = config_file['limit']

    # classes = config_file['classes']

    max_bboxes = config_file['max_bboxes']
    image_size = config_file['image_size']
    anchors_out_file = config_file['anchors_out_file']


    if input_data_source == 'tfrecords':
        tfrecords_dir = config_file['tfrecords']['tfrecords_dir']
        dataset = parse_tfrecords(tfrecords_dir, image_size, max_bboxes, class_file=None)
    elif input_data_source == 'data_files':
        images_dir = config_file['data_files']['images_dir']
        annotations_file = config_file['data_files']['annotations']
        max_dataset_examples = None
        dataset, _ = create_dataset_from_files(images_dir, annotations_file,
                                                                    image_size,
                                                                    max_dataset_examples,
                                                                    max_bboxes)


    anchors = creat_yolo_anchors(dataset, n_clusters)
    head, tail = os.path.split(anchors_out_file)
    pathlib.Path(head).mkdir(parents=True, exist_ok=True)

    np.savetxt(anchors_out_file, anchors, delimiter=',', fmt='%10.5f')
    print(f'result anchors:\n{anchors}')
    print(f'anchors_out_file: {anchors_out_file}')

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
