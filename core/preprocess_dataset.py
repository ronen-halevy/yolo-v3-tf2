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
from .utils import resize_image

import tensorflow as tf


class PreprocessDataset:

    # @tf.function
    def _extract_boxes_indices_on_grid(self, boxes, grid_shape, best_anchor_indices, max_bboxes):
        box_center_xy = (boxes[..., 0:2] + boxes[..., 2:4]) / 2
        box_center_xy = tf.reverse(box_center_xy, axis=[-1])
        box_center_xy_grid_indices = tf.cast(
            box_center_xy * grid_shape[1:3], tf.int32)

        batches = boxes.shape[0]
        batch_indices = tf.tile(tf.range(batches)[:, tf.newaxis], [
            1, max_bboxes])[:, :, tf.newaxis]
        grid_indices = tf.concat(
            [batch_indices, box_center_xy_grid_indices, best_anchor_indices], axis=-1)
        return grid_indices

    # @tf.function
    def _calc_best_iou_anchor_per_box(self, bboxes, anchors):
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

    # @tf.function
    def _arrange_in_grid(self, y_train, anchors, grid_index, output_shape, max_bboxes):
        """
        :param y_train:
        :type y_train:
        :param anchors:
        :type anchors:
        :param output_shape:
        :type output_shape:
        :param max_bboxes:
        :type max_bboxes:
        :return:
        :rtype:
        """

        best_anchor_indices = self._calc_best_iou_anchor_per_box(y_train, anchors)
        grid_indices = self._extract_boxes_indices_on_grid(
            y_train, output_shape, tf.cast(best_anchor_indices / anchors.shape[1], tf.int32), max_bboxes)

        # Ignore zero boxes:
        best_anchor_indices = tf.squeeze(best_anchor_indices, axis=-1)
        mask_selected_anchor_index_above_grid_sclae_min = tf.greater_equal(best_anchor_indices,
                                                                           tf.constant(grid_index * anchors.shape[0]))
        mask_selected_anchor_index_below_grid_sclae_max = tf.less(best_anchor_indices,
                                                                  tf.constant((grid_index + 1) * anchors.shape[0]))

        mask_valid_bbox = y_train[..., 4] != 0  # obj val
        mask = tf.math.logical_and(mask_valid_bbox, mask_selected_anchor_index_above_grid_sclae_min)
        mask = tf.math.logical_and(mask, mask_selected_anchor_index_below_grid_sclae_max)

        y_train = y_train[mask]
        grid_indices = grid_indices[mask]
        dataset_in_grid = tf.zeros(output_shape)

        dataset_in_grid = tf.tensor_scatter_nd_update(
            dataset_in_grid, grid_indices, y_train)

        return dataset_in_grid

    def preprocess_dataset_debug(self, dataset, batch_size, image_size, anchors_table, grid_sizes, max_bboxes):
        # TODO check that again!!
        # Important note: Since drop_remainder=True is set, dataset size must be at least batch_size, otherwise, dataset will be empty.
        # Reason for setting drop_remainder=True: This preprocess scatters data on a batch_size*grid_size*grid_size cube.
        # Implementation is vectorized oriented, so the batch dimennsion indices valuess are taken as [0:batch_size], assuming
        # batch_size entries in a batch. If drop_remainder was not true, last batch in an epoch might have less examples. In this
        # case, since number of examples would be less than batc_size indices, the program will fail.

        dataset = dataset.batch(batch_size, drop_remainder=True)
        downsize_strides = image_size / grid_sizes

        for x, y in dataset:
            resize_image(x, image_size, image_size),  # Todo - check it back !!!!
            # tf.image.resize(x, (image_size, image_size))
            tuple([self._arrange_in_grid(y, tf.convert_to_tensor(anchors_table), grid_index,
                                         # ronen TODO was 3,6 check shape
                                         [batch_size, grid_size, grid_size,
                                          anchors.shape[0], tf.shape(y)[-1]], max_bboxes
                                         ) for grid_index, (anchors, grid_stride, grid_size)
                   in
                   enumerate(zip(anchors_table, downsize_strides, grid_sizes))]
                  )
        return dataset

    def __call__(self, dataset, batch_size, image_size, anchors_table, grid_sizes, max_bboxes):
        # Important note: Since drop_remainder=True is set, dataset size must be at least batch_size, otherwise, dataset will be empty.
        # Reason for setting drop_remainder=True: This preprocess scatters data on a batch_size*grid_size*grid_size cube.
        # Implementation is vectorized oriented, so the batch dimennsion indices valuess are taken as [0:batch_size], assuming
        # batch_size entries in a batch. If drop_remainder was not true, last batch in an epoch might have less examples. In this
        # case, since number of examples would be less than batc_size indices, the program will fail.

        dataset = dataset.batch(batch_size, drop_remainder=True)

        downsize_strides = image_size / grid_sizes
        dataset = dataset.map(lambda x, y: (
            resize_image(x, image_size, image_size),  # Todo - check it back
            tuple([self._arrange_in_grid(y, tf.convert_to_tensor(anchors_table), grid_index,
                                         [batch_size, grid_size, grid_size,
                                          anchors.shape[0], tf.shape(y)[-1]], max_bboxes
                                         ) for grid_index, (anchors, grid_stride, grid_size)
                   in
                   enumerate(zip(anchors_table, downsize_strides, grid_sizes))]
                  )
        ))
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
