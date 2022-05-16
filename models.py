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

def conv_block(input, nfilters, kernel_size, strides=1, activation=True, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        input = ZeroPadding2D(((1, 0), (1, 0)))(input)  # top left half-padding
        padding = 'valid'
    conv = Conv2D(filters=nfilters, kernel_size=kernel_size,
                  strides=strides, padding=padding, # TODO: consider constants assignments
                  use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(input)
    if batch_norm:
        conv = BatchNormalization()(conv)
    if activation:
        conv = LeakyReLU(alpha=0.1)(conv) # TODO: consider constants assignments
    return conv


def residual_block(input, nfilters_1, nfilters_2):
    skip = input
    conv = conv_block(input, nfilters_1, kernel_size=1)
    conv = conv_block(conv, nfilters_2, kernel_size=3)
    output = Add()([skip, conv])
    return output


def darknet53(input_data, name=None):
    conv = conv_block(input_data, nfilters=32, kernel_size=3)
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

    out = tf.keras.Model(input_data, (out1, out2, conv), name=name)

    return out


def get_grids_common_block(filters, name=None):
    def grids_common_block(input_data):
        input = Input(input_data.shape[1:], name=f'{name} input')
        conv = conv_block(input, filters, 1)
        conv = conv_block(conv, filters * 2, 3)
        conv = conv_block(conv, filters, 1)
        conv = conv_block(conv, filters * 2, 3)
        conv = conv_block(conv, filters, 1)
        return Model(input, conv, name=name)(input_data)

    return grids_common_block


def get_concat_block(nfilters, name=None):
    def concat_block(prev_grid_intermediate, darknet_intermediate_out):
        inputs = Input(prev_grid_intermediate.shape[1:], name=f'{name} input1'), Input(
            darknet_intermediate_out.shape[1:], name=f'{name} input2')
        input, skip = inputs
        conv = conv_block(input, nfilters=nfilters, kernel_size=1)
        conv = UpSampling2D(2)(conv)
        conv = Concatenate()([conv, skip])
        return Model(inputs, conv, name=name)((prev_grid_intermediate, darknet_intermediate_out))

    return concat_block


def get_grid_output(nfilters, nanchors, nclasses, name=None):
    def grid_output(input_data):
        input = Input(input_data.shape[1:], name=f'{name} input')
        conv = conv_block(input, nfilters, kernel_size=3)
        conv = conv_block(conv, nfilters=nanchors * (nclasses + (XY_FEILD+WH_FEILD+OBJ_FIELD)), kernel_size=1, batch_norm=False)

        conv = Lambda(lambda x: tf.reshape(x, (-1, input_data.shape[1], input_data.shape[2],
                                               nanchors, nclasses + (XY_FEILD+WH_FEILD+OBJ_FIELD))))(conv)

        return Model(input, conv, name=name)(input_data)

    return grid_output


def grid_pred_decode(grid_pred, anchors, nclasses):
    batch_size = tf.shape(grid_pred)[0]
    pred_xy, pred_wh, pred_obj, pred_class = tf.split(grid_pred, [XY_FEILD, WH_FEILD, OBJ_FIELD, nclasses], axis=-1)
    grid_size = tf.shape(grid_pred)[1]
    grid_range = tf.range(grid_size, dtype=tf.int32)
    grid_x = tf.tile(tf.expand_dims(grid_range, axis=0), [grid_size, 1])
    grid_y = tf.tile(tf.expand_dims(grid_range, axis=-1), [1, grid_size])
    grid = tf.concat([tf.expand_dims(grid_x, axis=-1), tf.expand_dims(grid_y, axis=-1)], axis=-1)
    grid = tf.tile(grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchors.shape[0], 1])
    grid = tf.cast(grid, tf.float32)
    box_xy = (tf.sigmoid(pred_xy) + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(pred_wh) * anchors
    box_xy_min = box_xy - box_wh / 2
    box_xy_max = box_xy + box_wh / 2
    bbox = tf.concat([box_xy_min, box_xy_max], axis=-1)
    return bbox, pred_obj, pred_class


def yolov3_model(anchors_table, image_size=None, nclasses=80):
    inputs = Input([image_size, image_size, 3], name='darknet53 input')
    nanchors = anchors_table.shape[0]
    out1, out2, darknet_out = darknet53(inputs, name='darknet53')(inputs)
    coarse_intermediate_out = get_grids_common_block(filters=512, name='coarse_grid_path')(darknet_out)
    coarse_grid_pred = get_grid_output(nfilters=1024, nanchors=nanchors, nclasses=nclasses, name='coarse_output')(
        coarse_intermediate_out)

    med_concat_out = get_concat_block(nfilters=256)(coarse_intermediate_out, out2)
    med_intermediate_out = get_grids_common_block(filters=256, name='med_grid_path')(med_concat_out)
    med_grid_pred = get_grid_output(512, nanchors=nanchors, nclasses=nclasses, name='med_output')(med_intermediate_out)

    fine_concat_out = get_concat_block(nfilters=128)(med_intermediate_out, out1)
    fine_intermediate_out = get_grids_common_block(filters=256, name='fine_grid_path')(fine_concat_out)
    fine_grid_pred = get_grid_output(256, nanchors=nanchors, nclasses=nclasses, name='fine_output')(
        fine_intermediate_out)

    coarse_grid_out = Lambda(lambda x: grid_pred_decode(x, anchors_table[0], nclasses),
                             name='coarse_grid_pred_decode')(coarse_grid_pred)
    med_grid_out = Lambda(lambda x: grid_pred_decode(x, anchors_table[1], nclasses),
                          name='med_grid_pred_decode')(med_grid_pred)
    fine_grid_out = Lambda(lambda x: grid_pred_decode(x, anchors_table[2], nclasses),
                           name='fine_grid_pred_decode')(fine_grid_pred)

    # outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
    #                  name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    # return Model(inputs, (coarse_grid_pred, med_grid_pred, fine_grid_pred), name='yolov3')

    return Model(inputs, (coarse_grid_out, med_grid_out, fine_grid_out), name='yolov3')
