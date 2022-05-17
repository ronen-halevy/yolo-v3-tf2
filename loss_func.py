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
    CategoricalCrossentropy
)

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area


def bbox_iou_broadcast(boxes1, boxes2):
    boxes1 = tf.expand_dims(boxes1, -2)
    new_shape = tf.broadcast_dynamic_shape(tf.shape(boxes1), tf.shape(boxes2))
    boxes1 = tf.broadcast_to(boxes1, new_shape)
    boxes2 = tf.broadcast_to(boxes2, new_shape)
    iou = bbox_iou(boxes1, boxes2)
    return iou

def calc_class_loss(pred_class, true_class_idx, obj_mask):
    class_loss = obj_mask * sparse_categorical_crossentropy(
        true_class_idx, pred_class)

    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
    return class_loss


def calc_obj_loss(pred_obj, true_obj, obj_mask, ignore_mask):
    obj_loss = binary_crossentropy(true_obj, pred_obj)
    obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))

    return obj_loss


def calc_xy_loss(pred_xy, true_xy, obj_mask, box_loss_scale):
    # 5. calculate all losses
    squared_xy_loss = obj_mask * box_loss_scale * \
              tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
    sum_squared_xy_loss = tf.reduce_sum(squared_xy_loss, axis=(1, 2, 3))

    return sum_squared_xy_loss


def calc_wh_loss(pred_wh, true_wh, obj_mask, box_loss_scale):
    squared_wh_loss = obj_mask * box_loss_scale * \
              tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
    sum_squared_wh_loss = tf.reduce_sum(squared_wh_loss, axis=(1, 2, 3))

    return sum_squared_wh_loss


def get_loss_func(anchors, nclasses=80, ignore_thresh=0.5):
    def loss(y_true, y_pred):
        pred_box, pred_obj, pred_class = y_pred
        true_center_xy, true_wh, true_obj, true_class_idx = tf.split(
            y_true, (2, 2, 1, 1), axis=-1)

        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        true_center_xy = true_center_xy * tf.cast(grid_size, tf.float32)  # - \
        true_box = tf.concat([true_center_xy - true_wh / 2, true_center_xy + true_wh / 2], axis=-1)
        obj_mask = tf.squeeze(true_obj, -1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(bbox_iou_broadcast(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box[..., 0:4], obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        pred_center_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]

        xy_loss = calc_xy_loss(pred_center_xy, true_center_xy, obj_mask, box_loss_scale)

        wh_loss = calc_wh_loss(pred_wh, true_wh, obj_mask, box_loss_scale)
        obj_loss = calc_obj_loss(pred_obj, true_obj, obj_mask, ignore_mask)

        class_loss = calc_class_loss(pred_class, true_class_idx, obj_mask)
        return xy_loss, wh_loss, obj_loss, class_loss

    return loss


