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

f = open("newloss.txt", "w")

@tf.function
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

@tf.function
def broadcast_iou(box_1, box_2):
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)

    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)

@tf.function
def bbox_iou_broadcast(boxes1, boxes2):
    boxes1 = tf.expand_dims(boxes1, -2)

    boxes2 = tf.expand_dims(boxes2, 0)

    new_shape = tf.broadcast_dynamic_shape(tf.shape(boxes1), tf.shape(boxes2))
    boxes1 = tf.broadcast_to(boxes1, new_shape)
    boxes2 = tf.broadcast_to(boxes2, new_shape)
    iou = bbox_iou(boxes1, boxes2)
    return iou

#ronen TODO - cancel exp. using log instead
@tf.function
def decode_predictions2(pred_xy_center, pred_wh, pred_obj, pred_class, anchors):
    pred_xy_center_offset = tf.sigmoid(pred_xy_center)
    pred_wh_decoded = pred_wh

    pred_obj_decoded = tf.sigmoid(pred_obj)
    pred_class_decoded = tf.sigmoid(pred_class)
    return pred_xy_center_offset, pred_wh_decoded, pred_obj_decoded, pred_class_decoded

#ronen todo - old no tf.log
@tf.function
def decode_predictions(pred_xy_center, pred_wh, pred_obj, pred_class, anchors):
    pred_xy_center_offset = tf.sigmoid(pred_xy_center)
    pred_wh_decoded = tf.exp(pred_wh) * anchors

    pred_obj_decoded = tf.sigmoid(pred_obj)
    pred_class_decoded = tf.sigmoid(pred_class)
    return pred_xy_center_offset, pred_wh_decoded, pred_obj_decoded, pred_class_decoded

@tf.function
def arrange_pred_bbox(xy_center_offset, wh, grid_size):
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    xy_center = (tf.cast(grid, tf.float32) + xy_center_offset) / grid_size

    bbox = tf.concat([xy_center - wh / 2, xy_center + wh / 2], axis=-1)

    return bbox


def arrange_true_bbox(true_xy_min, true_xy_max, grid_size):
    bbox = tf.concat([true_xy_min, true_xy_max], axis=-1)
    return bbox

@tf.function
def calc_iou_ignore_mask(true_box, pred_bbox, true_obj, iou_ignore_thresh):
    best_iou = tf.map_fn(
        lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
            x[1], tf.cast(x[2], tf.bool))), axis=-1),
        (pred_bbox, true_box, true_obj),
        tf.float32)

    ignore_mask = tf.cast(best_iou < iou_ignore_thresh, tf.float32)
    return ignore_mask

@tf.function
def calc_xy_loss(pred_xy_center_offset, true_xy_center, grid_size, true_obj, box_loss_scale):
    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    true_xy_center_offset = (true_xy_center * grid_size - grid)

    xy_loss = true_obj * box_loss_scale * \
        tf.reduce_sum(tf.square(true_xy_center_offset -
                      pred_xy_center_offset), axis=-1)

    xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))

    return xy_loss

# ronen todo this is using log instead of exp
# @tf.function
def calc_wh_loss(pred_wh, true_wh, true_obj, box_loss_scale, grid_size, anchors):
    true_wh = tf.math.log(true_wh / anchors)
    true_wh = tf.where(tf.math.is_inf(true_wh),
                       tf.zeros_like(true_wh), true_wh)

    squared_wh_loss = true_obj * box_loss_scale * \
        tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)

    sum_squared_wh_loss = tf.reduce_sum(squared_wh_loss, axis=(1, 2, 3))

    return sum_squared_wh_loss

@tf.function
def calc_class_loss(pred_class, true_class_idx, obj_mask):
    class_loss = obj_mask * sparse_categorical_crossentropy(
        true_class_idx, pred_class)

    class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
    return class_loss

@tf.function
def calc_obj_loss(true_obj, decoded_pred_obj, obj_mask, ignore_mask):
    obj_loss = binary_crossentropy(true_obj, decoded_pred_obj)
    obj_loss = obj_mask * obj_loss + \
        (1 - obj_mask) * ignore_mask * obj_loss
    obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
    return obj_loss


def get_loss_func_keep(anchors, nclasses=80, iou_ignore_thresh=0.5):
    def new_lose(y_true, y_pred):

        true_xy_min, true_xy_max, true_obj, true_class_idx = tf.split(
            y_true, (2, 2, 1, 1), axis=-1)

        pred_xy_center, pred_wh, pred_obj, pred_class = tf.split(
            y_pred, (2, 2, 1, nclasses), axis=-1)

        decoded_pred_xy_center_offset, decoded_pred_wh, decoded_pred_obj, decoded_pred_class = decode_predictions2(
            pred_xy_center, pred_wh, pred_obj, pred_class, anchors)

        grid_size = tf.cast(tf.shape(y_pred)[1], tf.float32)

        pred_bbox = arrange_pred_bbox(
            decoded_pred_xy_center_offset, decoded_pred_wh, grid_size)

        true_bbox = arrange_true_bbox(true_xy_min, true_xy_max, grid_size)
        true_obj_mask = tf.squeeze(true_obj, axis=-1)
        iou_ignore_mask = calc_iou_ignore_mask(
            true_bbox, pred_bbox, true_obj_mask, iou_ignore_thresh)

        obj_loss = calc_obj_loss(
            true_obj, decoded_pred_obj, true_obj_mask, iou_ignore_mask)
        true_wh = true_xy_max - true_xy_min
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
        true_xy_center = (true_xy_min + true_xy_max) / 2

        xy_loss = calc_xy_loss(decoded_pred_xy_center_offset,
                               true_xy_center, grid_size, true_obj_mask, box_loss_scale)
        wh_loss = calc_wh_loss(decoded_pred_wh, true_wh,
                               true_obj_mask, box_loss_scale, grid_size, anchors)
        class_loss = calc_class_loss(
            decoded_pred_class, true_class_idx, true_obj_mask)
        return tf.stack([tf.math.reduce_sum(xy_loss), tf.math.reduce_sum(wh_loss), tf.math.reduce_sum(obj_loss),
                         tf.math.reduce_sum(class_loss)])

    return new_lose

def _meshgrid(n_a, n_b):

    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]
def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    box_wh = tf.exp(box_wh) * anchors
    # pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)


    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]


    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)


    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def get_loss_func(anchors, nclasses=80, iou_ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, nclasses)
        # pred_box, objectness, class_probs, pred_xywh = yolo_boxes(
        #     y_pred, anchors, nclasses)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < iou_ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return tf.stack([tf.math.reduce_sum(xy_loss), tf.math.reduce_sum(wh_loss), tf.math.reduce_sum(obj_loss),
                         tf.math.reduce_sum(class_loss)])

        # return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
