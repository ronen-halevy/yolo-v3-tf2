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
    return scaled_image


def extract_boxes_indices_on_grid(boxes, grid_shape, best_anchor_indices, max_bboxes):
    box_center_xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2
    box_center_xy_grid_indices = tf.cast(
        box_center_xy * grid_shape[1:3], tf.int32)

    batches = boxes.shape[0]
    batch_indices = tf.tile(tf.range(batches)[:, tf.newaxis], [
                            1, max_bboxes])[:, :, tf.newaxis]
    grid_indices = tf.concat(
        [batch_indices, box_center_xy_grid_indices, best_anchor_indices], axis=-1)
    return grid_indices


def calc_best_anchor_per_box(bboxes, anchors):
    anchors = tf.reshape(anchors, [-1, 2])
    grid_anchors = tf.cast(anchors, tf.float32)
    anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]
    box_wh = bboxes[..., 2:4] - bboxes[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(grid_anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(
        box_wh[..., 0], grid_anchors[..., 0]) * tf.minimum(box_wh[..., 1], grid_anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    best_anchor_indices = tf.expand_dims(tf.math.argmax(
        iou, axis=-1, output_type=tf.int32), axis=-1)
    return best_anchor_indices


def arrange_in_grid(y_train, anchors, grid_index, output_shape, max_bboxes):
    """
    :param y_train:
    :type y_train:
    :param anchors:
    :type anchors:
    :param downsize_stride:
    :type downsize_stride:
    :param output_shape:
    :type output_shape:
    :param max_bboxes:
    :type max_bboxes:
    :return:
    :rtype:
    """

    best_anchor_indices = calc_best_anchor_per_box(y_train, anchors)
    print(best_anchor_indices)

    grid_indices = extract_boxes_indices_on_grid(
        y_train, output_shape, best_anchor_indices % 3, max_bboxes)

    # Ignore zero boxes:
    mask_valid_bbox = y_train[..., 2] != 0
    best_anchor_indices = tf.squeeze(best_anchor_indices, axis=-1)
    mask_selected_anchor_index_above_min = tf.greater(best_anchor_indices, tf.constant((2-grid_index)*anchors.shape[0]))
    mask_selected_anchor_index_below_max = tf.less(best_anchor_indices, tf.constant(((2-grid_index)+1)*anchors.shape[0]))

    mask = tf.math.logical_and(mask_valid_bbox, mask_selected_anchor_index_above_min)
    mask = tf.math.logical_and(mask, mask_selected_anchor_index_below_max)

    y_train = y_train[mask]
    grid_indices = grid_indices[mask]

    dataset_in_grid = tf.zeros(output_shape)
    # to ensure uniqueness in grid entries, taking the max:TODO - prefers bigger if on same grid box

    dataset_in_grid = tf.tensor_scatter_nd_max(
        dataset_in_grid, grid_indices, y_train
    )
    return dataset_in_grid


def preprocess_dataset(dataset, batch_size, image_size, anchors_table, grid_sizes, max_bboxes):
    # TODO check that again!!
    dataset = dataset.batch(batch_size, drop_remainder=True)

    downsize_strides = image_size / grid_sizes
    # dataset = dataset.map(lambda x, y: (
    #     resize_image(x, image_size, image_size),
    #     tuple([arrange_in_grid(y, tf.convert_to_tensor(anchors_table), grid_index # ronen TODO was 3,6 check shape
    #                            # +1 is a patch - todo add the obj in dataset already...
    #                            [batch_size, grid_size, grid_size,
    #                                anchors.shape[0], tf.shape(y)[-1]], max_bboxes
    #                            ) for grid_index, (anchors, grid_stride, grid_size)
    #            in
    #            enumerate(zip(anchors_table, downsize_strides, grid_sizes))]
    #           )
    # ))

    dataset = dataset.map(lambda x, y: (
        resize_image(x, image_size, image_size),
        tuple([arrange_in_grid(y, tf.convert_to_tensor(anchors_table), grid_index,  # ronen TODO was 3,6 check shape
            # +1 is a patch - todo add the obj in dataset already...
        [batch_size, grid_size, grid_size,
         anchors.shape[0], tf.shape(y)[-1]], max_bboxes
                               ) for grid_index, (anchors, grid_stride, grid_size)
               in
               enumerate(zip(anchors_table, downsize_strides, grid_sizes))]
              )
    ))

    # dataset = dataset.prefetch(
    #     buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
