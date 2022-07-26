#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : tester.py
#   Author      : ronen halevy 
#   Created date:  5/9/22
#   Description :
#
# ================================================================
from matplotlib import pyplot as plt

import train
from core import preprocess_dataset, utils

import numpy as np


def plot_transformed_dataset(dataset, num_of_grids=3, num_of_anchors_set=3):
    """

    :param dataset:
    :type dataset:
    :return:
    :rtype:
    """
    data = list(dataset.as_numpy_iterator())
    image_batch, grids_boxes_batch = next(dataset.as_numpy_iterator())
    fig, axs = plt.subplots(num_of_grids, num_of_anchors_set, figsize=(15, 15))

    fig.suptitle("Bounding Boxes Per Grid Per Anchors Set")
    for grid_index, boxes_batch in enumerate(grids_boxes_batch):
        for index in range(num_of_anchors_set):
            boxes = boxes_batch[0]
            image = np.expand_dims(image_batch[0], axis=0)
            per_anchor_boxes = boxes[:, :, index, :]
            mask = per_anchor_boxes[:, :, 2] != 0
            per_anchor_boxes = per_anchor_boxes[mask]
            image = utils.render_bboxes(image, per_anchor_boxes)
            axs[grid_index, index].imshow(image)
            axs[grid_index, index].title.set_text(
                f'\n{per_anchor_boxes.shape[0]} boxes in \nGrid: {boxes.shape[0]}x{boxes.shape[0]}. Anchor set: {index}')

    plt.tight_layout()
    plt.show()


def test_preprocess_dataset():
    """

    :return:
    :rtype:
    """
    dataset, dataset_size, batch_size, image_size, anchors, max_bboxes, grid_sizes = train.config_train()
    train_dataset, test_dataset, val_dataset = train.split_dataset(dataset, dataset_size)
    dataset = preprocess_dataset.preprocess_dataset(dataset, batch_size, image_size, anchors, grid_sizes, max_bboxes)
    plot_transformed_dataset(dataset)
    # Create two subplots and unpack the output array immediately


if __name__ == '__main__':
    test_preprocess_dataset()
