#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : transorm-train-dataset.py
#   Author      : ronen halevy 
#   Created date:  4/27/22
#   Description :
#
# ================================================================

import tensorflow as tf


def resize_image(image, t_w, t_h):
    _, s_h, s_w, _ = image.shape

    scale = min(t_w / s_w, t_h / s_h)

    n_w, n_h = int(scale * s_w), int(scale * s_h)

    image = tf.image.resize(
        image,
        [n_h, n_w],
    )

    scaled_image = tf.image.pad_to_bounding_box(
        image, (t_h - n_h) // 2, (t_w - n_w) // 2, t_h, t_w
    )
    # y_n = tf.squeeze(y_n, axis=0)
    return scaled_image


# @tf.function
def arrange_in_grid(y_train, anchors, downsize_stride, output_shape, max_boxes):
    """

    :param y_train:
    :type y_train:
    :param anchors:
    :type anchors:
    :param downsize_stride:
    :type downsize_stride:
    :param output_shape:
    :type output_shape:
    :param max_boxes:
    :type max_boxes:
    :return:
    :rtype:
    """
    grid_anchors = tf.cast(anchors, tf.float32)
    anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(grid_anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], grid_anchors[..., 0]) * tf.minimum(box_wh[..., 1], grid_anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    iou_max = tf.math.argmax(iou, axis=-1)
    iou_max = tf.expand_dims(iou_max, axis=-1)
    iou_max = tf.cast(iou_max, tf.int32)
    box_xy = (y_train[..., 0:2] + y_train[..., 2:4]) / 2
    box_xy = tf.cast(box_xy, tf.float32)

    grid_xy = tf.cast(box_xy // downsize_stride, tf.int32)

    indices_tmp = tf.concat([grid_xy, iou_max], axis=-1)

    batches = y_train.shape[0]
    # !!!! ronen should be number of boxes 950)
    # boxes = y_train.shape[-1]
    boxes = max_boxes
    batch_box_indices = tf.tile(tf.expand_dims(tf.range(batches), -1), [1, boxes])

    batch_box_indices = tf.expand_dims(batch_box_indices, axis=-1)
    batch_box_indices = tf.cast(batch_box_indices, tf.int32)
    indices = tf.concat([batch_box_indices, indices_tmp], axis=-1)
    scale_dataset = tf.scatter_nd(
        indices, y_train, output_shape
    )
    # tf.print(scale_dataset.shape)

    return scale_dataset


def preprocess_dataset(dataset, batch_size, image_size, anchors, grid_sizes, max_boxes):
    dataset = dataset.batch(batch_size, drop_remainder=True)
    downsize_strides = image_size / grid_sizes
    dataset = dataset.map(lambda x, y: (
        resize_image(x, image_size, image_size),
        tuple([arrange_in_grid(y, tf.convert_to_tensor(anchor), grid_stride,
                               [batch_size, grid_size, grid_size, 3, 5], max_boxes) for anchor, grid_stride, grid_size
               in
               zip(anchors, downsize_strides, grid_sizes)])
    ))
    return dataset
