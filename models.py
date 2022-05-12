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

def conv_block(input, nfilters, kernel_size, strides=1, activation=True, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        input = ZeroPadding2D(((1, 0), (1, 0)))(input)  # top left half-padding
        padding = 'valid'
    conv = Conv2D(filters=nfilters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(input)
    if batch_norm:
        conv = BatchNormalization()(conv)
    if activation:
        conv = LeakyReLU(alpha=0.1)(conv)
    return conv

def residual_block(input, nfilters_1, nfilters_2):
    skip = input
    conv = conv_block(input, nfilters_1, kernel_size=1)
    conv = conv_block(conv, nfilters_2, kernel_size=3)
    output = Add()([skip, conv])
    return output


def darknet53(input_data, name=None):

    conv = conv_block(input_data, nfilters=32,  kernel_size=3)
    conv = conv_block(conv, nfilters=64,  kernel_size=3, strides=2)

    for i in range(1):
        conv = residual_block(conv,  nfilters_1=32,  nfilters_2=64)

    conv = conv_block(conv,  nfilters=128,  kernel_size=3, strides=2)

    for i in range(2):
        conv = residual_block(conv, nfilters_1=64,  nfilters_2=128)

    conv = conv_block(conv,  nfilters=256,  kernel_size=3, strides=2)

    for i in range(8):
        conv = residual_block(conv, nfilters_1=128,  nfilters_2=256)

    out1 = conv
    conv = conv_block(conv,  nfilters=512,  kernel_size=3, strides=2)

    for i in range(8):
        conv = residual_block(conv, nfilters_1=256, nfilters_2=512)

    out2 = conv
    conv = conv_block(conv, nfilters=1024,  kernel_size=3, strides=2)

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
        inputs = Input(prev_grid_intermediate.shape[1:], name=f'{name} input1'), Input(darknet_intermediate_out.shape[1:], name=f'{name} input2')
        input, skip = inputs
        conv = conv_block(input, nfilters=nfilters,  kernel_size=1)
        conv = UpSampling2D(2)(conv)
        conv = Concatenate()([conv, skip])
        return Model(inputs, conv, name=name)((prev_grid_intermediate, darknet_intermediate_out))
    return concat_block

def get_grid_output(nfilters, anchors, classes, name=None):
    def grid_output(input_data):
        input = Input(input_data.shape[1:], name=f'{name} input')
        conv = conv_block(input, nfilters, kernel_size=3)
        conv = conv_block(conv, nfilters=anchors*(classes + 5), kernel_size=1, batch_norm=False)
        return Model(input, conv, name=name)(input_data)

    return grid_output





def model(size=None, nanchors=3, nclasses=80):
    input = Input([size, size, 3], name='darknet53 input')
    out1, out2, darknet_out = darknet53(input, name='darknet53')(input)
    coarse_intermediate_out = get_grids_common_block(filters=512, name='coarse_grid_path')(darknet_out)
    coarse_output = get_grid_output(nfilters=1024, anchors=nanchors, classes=nclasses, name='coarse_output')(coarse_intermediate_out)



    med_concat_out = get_concat_block(nfilters=256)(coarse_intermediate_out, out2)
    med_intermediate_out = get_grids_common_block(filters=256, name='med_grid_path')(med_concat_out)
    med_output = get_grid_output(512, anchors=nanchors, classes=nclasses, name='med_output')(med_intermediate_out)
    #
    fine_concat_out = get_concat_block(nfilters=128)(med_intermediate_out, out1)
    fine_intermediate_out = get_grids_common_block(filters=256, name='fine_grid_path')(fine_concat_out)
    fine_output = get_grid_output(256, anchors=nanchors, classes=nclasses, name='fine_output')(fine_intermediate_out)

    return Model(input, (coarse_output, med_output, fine_output), name='yolov3')