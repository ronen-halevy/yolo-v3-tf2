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

    def _produce_grid_scale_indices(self, boxes, grid_shape, best_anchor_indices, max_bboxes):
        # box contains: x,y,w,h
        box_center_xy = (boxes[..., 0:2] + boxes[..., 2:4]/2)
        # why reverse - scatter indices constructed here point on (row, col), while before reverse x,y are (col, row)
        box_center_xy = tf.reverse(box_center_xy, axis=[-1])
        box_center_xy_grid_indices = tf.cast(
            box_center_xy * grid_shape[1:3], tf.int32)

        batches = boxes.shape[0]
        batch_indices = tf.tile(tf.range(batches)[:, tf.newaxis], [
            1, max_bboxes])[:, :, tf.newaxis]
        grid_indices = tf.concat(
            [batch_indices, box_center_xy_grid_indices, best_anchor_indices], axis=-1)
        return grid_indices

    def _find_max_iou_anchors(self, bboxes, anchors):
        anchors = tf.reshape(anchors, [-1, 2])
        grid_anchors = tf.cast(anchors, tf.float32)
        anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]
        box_wh = bboxes[..., 2:4]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                         (1, 1, tf.shape(grid_anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(
            box_wh[..., 0], grid_anchors[..., 0]) * tf.minimum(box_wh[..., 1], grid_anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        best_anchor_indices = tf.math.argmax(
            iou, axis=-1, output_type=tf.int32)
        return best_anchor_indices

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
        iou_selected_anchors = self._find_max_iou_anchors(y_train, anchors)
        grid_scaled_boxes_indices = self._produce_grid_scale_indices(
            y_train, output_shape, tf.cast(tf.expand_dims(iou_selected_anchors, axis=-1) / anchors.shape[1], tf.int32),
            max_bboxes)

        # Find best_iou_grid_index - iou_selected_anchors's related grid index:
        best_iou_grid_index = tf.histogram_fixed_width_bins(
            values=tf.cast(iou_selected_anchors, tf.float32),  # tf.cast(iou_selected_anchors, tf.float32),
            value_range=[0., tf.size(anchors, tf.dtypes.float32)],
            nbins=anchors.shape[0],
            dtype=tf.dtypes.float32,
            name=None
        )
        # if best_iou_grid_index is not equal to current loop's pass grid_index- mask box off:
        grid_index_mask = best_iou_grid_index == grid_index

        # Mask off boxes if valid indication a.k.a obj is not set :
        mask_valid_bbox = y_train[..., 4] != 0  # obj val
        # Integrate masks:
        mask = tf.math.logical_and(mask_valid_bbox, grid_index_mask)

        y_train = y_train[mask]
        grid_scaled_boxes_indices = grid_scaled_boxes_indices[mask]
        dataset_in_grid = tf.zeros(output_shape)

        dataset_in_grid = tf.tensor_scatter_nd_update(
            dataset_in_grid, grid_scaled_boxes_indices, y_train)

        return dataset_in_grid

    def preprocess_dataset_debug(self, dataset, batch_size, image_size, anchors_table, grid_sizes, max_bboxes):
        # TODO check that again!!
        # Important note: Since drop_remainder=True is set, dataset size must be at least batch_size, otherwise, dataset
        # will be empty.
        # Also if the reminder of dataset end is less then batch size, it is drop. E.g if batch size is 8, dataset of
        # length 7 or length 33 drops 7 samples
        # Reason for setting drop_remainder=True is the requirement for a predictable batch size.
        # The preprocessed data is scattered to a batch_size*grid_size*grid_size cube. So batch_size must be predicted.
        # If drop_remainder was not true, last batch in an epoch might have less examples. In this
        # case, since number of examples would be less than batc_size indices, the program will fail.
        # Alternative implementation: Set batch size to 1, to avoid any drop.

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
