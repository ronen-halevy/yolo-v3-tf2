#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : models.py
#   Author      : ronen halevy 
#   Created date:  5/11/22
#   Description :
#
# ================================================================
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

XY_FEILD = 2
WH_FEILD = 2
OBJ_FIELD = 1


def conv_block(Input, nfilters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        Input = ZeroPadding2D(((1, 0), (1, 0)))(Input)  # top left half-padding
        padding = 'valid'
    conv = Conv2D(filters=nfilters, kernel_size=kernel_size,
                  strides=strides, padding=padding,  # TODO: consider constants assignments
                  use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(Input)
    if batch_norm:
        conv = BatchNormalization()(conv)
        conv = LeakyReLU(alpha=0.1)(conv)  # TODO: consider constants assignments
    return conv


def residual_block(Input, nfilters_1, nfilters_2):
    skip = Input
    conv = conv_block(Input, nfilters_1, kernel_size=1)
    conv = conv_block(conv, nfilters_2, kernel_size=3)
    output = Add()([skip, conv])
    return output


def darknet53(name=None):
    inputs = Input([None, None, 3], name='darknet53 in')

    conv = conv_block(inputs, nfilters=32, kernel_size=3)
    conv = conv_block(conv, nfilters=64, kernel_size=3, strides=2)

    for i in range(1):
        conv = residual_block(conv, nfilters_1=32, nfilters_2=64)

    conv = conv_block(conv, nfilters=128, kernel_size=3, strides=2)

    for i in range(2):
        conv = residual_block(conv, nfilters_1=64, nfilters_2=128)

    conv = conv_block(conv, nfilters=256, kernel_size=3, strides=2)

    for i in range(8):
        conv = residual_block(conv, nfilters_1=128, nfilters_2=256)

    out1 = conv
    conv = conv_block(conv, nfilters=512, kernel_size=3, strides=2)

    for i in range(8):
        conv = residual_block(conv, nfilters_1=256, nfilters_2=512)

    out2 = conv
    conv = conv_block(conv, nfilters=1024, kernel_size=3, strides=2)

    for i in range(4):
        conv = residual_block(conv, nfilters_1=512, nfilters_2=1024)

    out = tf.keras.Model(inputs, (out1, out2, conv), name=name)

    return out


def get_grids_common_blockn(filters, name=None):
    def grids_common_blockn(input_data):
        input = Input(input_data.shape[1:], name=f'{name} input')
        conv = conv_block(input, filters, 1)
        conv = conv_block(conv, filters * 2, 3)
        conv = conv_block(conv, filters, 1)
        conv = conv_block(conv, filters * 2, 3)
        conv = conv_block(conv, filters, 1)
        return Model(input, conv, name=name)(input_data)

    return grids_common_blockn


def get_grids_common_block(nfilters, name=None):
    def grids_common_block(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:], name=f'{name} input1'), Input(x_in[1].shape[1:], name=f'{name} input2')
            input, skip = inputs
            conv = conv_block(input, nfilters, kernel_size=1)
            conv = UpSampling2D(2)(conv)
            conv = Concatenate()([conv, skip])
        else:
            conv = inputs = Input(x_in.shape[1:])

        conv = conv_block(conv, nfilters, 1)
        conv = conv_block(conv, nfilters * 2, 3)
        conv = conv_block(conv, nfilters, 1)
        conv = conv_block(conv, nfilters * 2, 3)
        conv = conv_block(conv, nfilters, 1)
        return Model(inputs, conv, name=name)(x_in)

    return grids_common_block


def get_grid_output(nfilters, nanchors, nclasses, name=None):
    def grid_output(input_data):
        input = Input(input_data.shape[1:], name=f'{name} input')
        conv = conv_block(input, nfilters, kernel_size=3)
        conv = conv_block(conv, nfilters=nanchors * (nclasses + (XY_FEILD + WH_FEILD + OBJ_FIELD)), kernel_size=1,
                          batch_norm=False)

        conv = Lambda(lambda x: tf.reshape(x, (-1, input_data.shape[1], input_data.shape[2],
                                               nanchors, nclasses + (XY_FEILD + WH_FEILD + OBJ_FIELD))))(conv)

        return Model(input, conv, name=name)(input_data)

    return grid_output


def yolo_nms(outputs, classes, yolo_max_boxes, nms_iou_threshold, nms_score_threshold):
    bbox, confidence, class_probs = outputs

    scores = confidence * tf.sqrt(class_probs)

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores, [1])
    bbox = tf.reshape(bbox, (-1, 4))
    classes = tf.argmax(dscores, 1)
    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=yolo_max_boxes,
        iou_threshold=nms_iou_threshold,
        score_threshold=nms_score_threshold,
        soft_nms_sigma=0.
    )

    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    num_of_valid_detections = tf.expand_dims(tf.shape(selected_indices)[0], axis=0)
    selected_boxes = tf.gather(bbox, selected_indices)
    selected_boxes = tf.expand_dims(selected_boxes, axis=0)
    selected_scores = tf.expand_dims(selected_scores, axis=0)
    selected_classes = tf.gather(classes, selected_indices)
    selected_classes = tf.expand_dims(selected_classes, axis=0)

    return selected_boxes, selected_scores, selected_classes, num_of_valid_detections


def arrange_bbox(xy, wh):
    grid_size = xy.shape[1:3]
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    bbox = tf.concat([xy_min, xy_max], axis=-1)
    return bbox


def arrange_output(grid_pred, nclasses):
    pred_xy, pred_wh, pred_obj, class_probs = Lambda(lambda x: tf.split(x, (2, 2, 1, nclasses), axis=-1))(grid_pred)

    # pred_wh = Lambda(lambda x: tf.exp(x[1]) * anchors_table[0])(pred_xw)

    pred_xy = Lambda(lambda x: tf.sigmoid(x))(pred_xy)
    pred_obj = Lambda(lambda x: tf.sigmoid(x))(pred_obj)
    class_probs = Lambda(lambda x: tf.sigmoid(x))(class_probs)
    return pred_xy, pred_wh, pred_obj, class_probs


def yolov3_model(anchors_table, image_size=None, nclasses=80, training=True, yolo_max_boxes=None,
                 nms_iou_threshold=None, nms_score_threshold=None):
    inputs = Input([image_size, image_size, 3], name='darknet53 input')
    nanchors = anchors_table.shape[0]
    out1, out2, darknet_out = darknet53(name='darknet53')(inputs)

    intermediate_out0 = get_grids_common_block(nfilters=512, name='intermediate_out0')(darknet_out)
    intermediate_out1 = get_grids_common_block(nfilters=256, name='intermediate_out1')((intermediate_out0, out2))
    intermediate_out2 = get_grids_common_block(nfilters=128, name='intermediate_out2')((intermediate_out1, out1))

    grid_pred0 = get_grid_output(1024, nanchors=nanchors, nclasses=nclasses, name='grid_pred0')(intermediate_out0)
    grid_pred1 = get_grid_output(512, nanchors=nanchors, nclasses=nclasses, name='grid_pred1')(intermediate_out1)
    grid_pred2 = get_grid_output(256, nanchors=nanchors, nclasses=nclasses, name='grid_pred2')(intermediate_out2)

    pred_xy0, pred_wh0, pred_obj0, class_probs0 = arrange_output(grid_pred0, nclasses)
    # pred_wh0 = Lambda(lambda x: tf.exp(x) * anchors_table[0])(pred_wh0)

    pred_xy1, pred_wh1, pred_obj1, class_probs1 = arrange_output(grid_pred1, nclasses)
    # pred_wh1 = Lambda(lambda x: tf.exp(x) * anchors_table[1])(pred_wh1)

    pred_xy2, pred_wh2, pred_obj2, class_probs2 = arrange_output(grid_pred2, nclasses)
    # pred_wh2 = Lambda(lambda x: tf.exp(x) * anchors_table[2])(pred_wh2)


    out_grid0 = Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]], axis=-1))(
        (pred_xy0, pred_wh0, pred_obj0, class_probs0))
    out_grid1 = Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]], axis=-1))(
        (pred_xy1, pred_wh1, pred_obj1, class_probs1))
    out_grid2 = Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]], axis=-1))(
        (pred_xy2, pred_wh2, pred_obj2, class_probs2))

    if training:
        return Model(inputs, (out_grid0, out_grid1, out_grid2
                              ), name='yolov3')

    bbox0 = Lambda(lambda x: arrange_bbox(x[0], tf.exp(x[1]) * anchors_table[0]))((pred_xy0, pred_wh0))
    bbox1 = Lambda(lambda x: arrange_bbox(x[0], tf.exp(x[1]) * anchors_table[1]))((pred_xy1, pred_wh1))
    bbox2 = Lambda(lambda x: arrange_bbox(x[0], tf.exp(x[1]) * anchors_table[2]))((pred_xy2, pred_wh2))

    merge_grid_outputs_op = Lambda(
        lambda x: tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in x], axis=1))
    bbox = merge_grid_outputs_op((bbox0, bbox1, bbox2))
    confidence = merge_grid_outputs_op((pred_obj0, pred_obj1, pred_obj2))
    class_probs = merge_grid_outputs_op((class_probs0, class_probs1, class_probs2))

    outputs = Lambda(lambda x: yolo_nms(x, nclasses, yolo_max_boxes, nms_iou_threshold, nms_score_threshold),
                     name='yolo_nms')((bbox, confidence, class_probs))

    return Model(inputs, outputs, name='yolov3')
