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
import preprocess_dataset
import utils

import numpy as np


def test_preprocess_dataset():
    """

    :return:
    :rtype:
    """
    dataset, dataset_size, batch_size, image_size, anchors, max_boxes, grid_sizes = train.config_train()
    train_dataset, test_dataset, val_dataset = train.split_dataset(dataset, dataset_size)
    dataset = preprocess_dataset.preprocess_dataset(dataset, batch_size, image_size, anchors, grid_sizes, max_boxes)
    data = list(dataset.as_numpy_iterator())
    image_batch, grids_boxes_batch = next(dataset.as_numpy_iterator())
    # Create two subplots and unpack the output array immediately
    f, axs = plt.subplots(3, 3)
    # ax1.plot(x, y)
    # ax1.set_title('Sharing Y axis')
    # ax2.scatter(x, y)
    for grid_index, boxes_batch in enumerate(grids_boxes_batch):
        for index in range(anchors.shape[-2]):
            boxes = boxes_batch[0]
            image = np.expand_dims(image_batch[0], axis=0)
            per_anchor_boxes = boxes[:, :, index, :]
            mask = per_anchor_boxes[:, :, 2] != 0
            per_anchor_boxes = per_anchor_boxes[mask]
            image = utils.render(image, per_anchor_boxes)
            axs[grid_index, index].imshow(image)

    plt.tight_layout()
    pass
    plt.show()
    pass
    pass


if __name__ == '__main__':
    test_preprocess_dataset()
