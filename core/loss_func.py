#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : loss_func.py
#   Author      : ronen halevy
#   Created date:  5/11/22
#   Description :
#
# ================================================================
import tensorflow as tf
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy,
)


def get_loss_func(anchors, nclasses, eager_mode=True):
    def yolo_loss(y_true, y_pred):
        pred_xy, pred_wh, pred_obj, pred_class = tf.split(
            y_pred, (2, 2, 1, nclasses), axis=-1)

        # pred_xy = tf.sigmoid(pred_xy)
        # pred_obj = tf.sigmoid(pred_obj)
        # pred_class = tf.sigmoid(pred_class)

        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)

        # Next 2 lines: assume format xmin, ymin, xmax, ymax
        true_bbox_center_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 1# 2 - true_wh[..., 0] * true_wh[..., 1]
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1] # ronen added here to have exact results

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_bbox_center_xy = true_bbox_center_xy * tf.cast(grid_size, tf.float32) - \
                  tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        obj_mask = tf.squeeze(true_obj, -1)

        xy_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_bbox_center_xy - pred_xy), axis=-1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(0, 1, 2, 3))

        wh_loss = obj_mask * box_loss_scale * \
                  tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        wh_loss = tf.reduce_sum(wh_loss, axis=(0, 1, 2, 3))

        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_sum(obj_loss, axis=(0, 1, 2, 3))

        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)
        class_loss = tf.reduce_sum(class_loss, axis=(0, 1, 2, 3))
        # in eager mode, extra info is returned for display/debug:
        result = tf.cond(eager_mode,  lambda: tf.stack([xy_loss, wh_loss, obj_loss, class_loss]), lambda: xy_loss+wh_loss+obj_loss+class_loss)
        return result


    return yolo_loss
