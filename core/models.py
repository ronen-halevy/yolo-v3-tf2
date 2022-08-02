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


class YoloV3Model:
    XY_FEILD = 2
    WH_FEILD = 2
    OBJ_FIELD = 1

    @staticmethod
    def _conv_block(in_data, nfilters, kernel_size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            in_data = ZeroPadding2D(((1, 0), (1, 0)))(in_data)  # top left half-padding
            padding = 'valid'
        conv = Conv2D(filters=nfilters, kernel_size=kernel_size,
                      strides=strides, padding=padding,  # TODO: consider constants assignments
                      use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(in_data)
        if batch_norm:
            conv = BatchNormalization()(conv)
            conv = LeakyReLU(alpha=0.1)(conv)  # TODO: consider constants assignments

        return conv

    def _residual_block(self, in_data, nfilters_1, nfilters_2):
        skip = in_data
        conv = self._conv_block(in_data, nfilters_1, kernel_size=1)
        conv = self._conv_block(conv, nfilters_2, kernel_size=3)
        output = Add()([skip, conv])
        return output

    def _darknet53(self, name=None):
        inputs = Input([None, None, 3], name='_darknet53 in')

        conv = self._conv_block(inputs, nfilters=32, kernel_size=3)
        conv = self._conv_block(conv, nfilters=64, kernel_size=3, strides=2)

        for i in range(1):
            conv = self._residual_block(conv, nfilters_1=32, nfilters_2=64)

        conv = self._conv_block(conv, nfilters=128, kernel_size=3, strides=2)

        for i in range(2):
            conv = self._residual_block(conv, nfilters_1=64, nfilters_2=128)

        conv = self._conv_block(conv, nfilters=256, kernel_size=3, strides=2)

        for i in range(8):
            conv = self._residual_block(conv, nfilters_1=128, nfilters_2=256)

        out1 = conv
        conv = self._conv_block(conv, nfilters=512, kernel_size=3, strides=2)

        for i in range(8):
            conv = self._residual_block(conv, nfilters_1=256, nfilters_2=512)

        out2 = conv
        conv = self._conv_block(conv, nfilters=1024, kernel_size=3, strides=2)

        for i in range(4):
            conv = self._residual_block(conv, nfilters_1=512, nfilters_2=1024)

        out = tf.keras.Model(inputs, (out1, out2, conv), name=name)

        return out


    def upsample_and_concat(self, nfilters, name=None):
        def network_head(x_in):
            in_data, skip = x_in
            conv = self._conv_block(in_data, nfilters, kernel_size=1)
            conv = UpSampling2D(2)(conv)
            conv = Concatenate()([conv, skip])
            return conv
        return network_head

    def get_network_neck(self, nfilters, ngrids, nclasses, name=None):
        def network_neck(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:], name=f'{name} input1'), Input(x_in[1].shape[1:],
                                                                                name=f'{name} input2')
                in_data, skip = inputs
                conv = self._conv_block(in_data, nfilters, kernel_size=1)
                conv = UpSampling2D(2)(conv)
                conv = Concatenate()([conv, skip])
            else:
                conv = inputs = Input(x_in.shape[1:])

            conv = self._conv_block(conv, nfilters, 1)
            conv = self._conv_block(conv, nfilters * 2, 3)
            conv = self._conv_block(conv, nfilters, 1)
            conv = self._conv_block(conv, nfilters * 2, 3)
            conv = self._conv_block(conv, nfilters, 1)
            return Model(inputs, conv, name=name)(x_in)
        return network_neck

    def get_network_head(self, nfilters, ngrids, nclasses, name=None):
        def network_head(x_in):
            conv = inputs = Input(x_in.shape[1:])
            conv = self._conv_block(conv, nfilters, 3)
            conv = self._conv_block(conv,
                                    nfilters=ngrids * (nclasses + (self.XY_FEILD + self.WH_FEILD + self.OBJ_FIELD)),
                                    kernel_size=1,
                                    batch_norm=False)

            conv = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                                   ngrids,
                                                   nclasses + (self.XY_FEILD + self.WH_FEILD + self.OBJ_FIELD))))(
                conv)

            return Model(inputs, conv, name=name)(x_in)

        return network_head

    @staticmethod
    def _arrange_output(grid_pred, nclasses):
        pred_xy, pred_wh, pred_obj, class_probs = Lambda(lambda x: tf.split(x, (2, 2, 1, nclasses), axis=-1))(grid_pred)
        pred_xy = Lambda(lambda x: tf.sigmoid(x))(pred_xy)
        pred_obj = Lambda(lambda x: tf.sigmoid(x))(pred_obj)
        class_probs = Lambda(lambda x: tf.sigmoid(x))(class_probs)
        concat_output = Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]], axis=-1))((pred_xy, pred_wh, pred_obj,
                                                                           class_probs))
        return concat_output

    def __call__(self, image_size=None, nclasses=80, nanchors=3):
        inputs = Input([image_size, image_size, 3], name='darknet53 input')
        out1, out2, darknet_out = self._darknet53(name='darknet53')(inputs)

        neck_out0 = self.get_network_neck(512, nanchors, nclasses, name='neck0')(darknet_out)
        head_out0 = self.get_network_head(1024, nanchors, nclasses, name='head0')(neck_out0)

        neck_out1 = self.get_network_neck(256, nanchors, nclasses, name='neck1')((neck_out0, out2))
        head_out1 = self.get_network_head(512, nanchors, nclasses, name='head1')(neck_out1)

        neck_out2 = self.get_network_neck(128, nanchors, nclasses, name='neck2')((neck_out1, out1))
        head_out2 = self.get_network_head(256, nanchors, nclasses, name='head2')(neck_out2)


        concat_output0 = self._arrange_output(head_out0, nclasses)
        concat_output1 = self._arrange_output(head_out1, nclasses)
        concat_output2 = self._arrange_output(head_out2, nclasses)
        return Model(inputs, (concat_output0, concat_output1, concat_output2), name='yolov3')

