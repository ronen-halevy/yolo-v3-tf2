import os

import numpy as np
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras import Input, Model


# TODO: It seems like turning on eager execution always gives different values during
# TODO: inference. Weirdly, it also loads the network very fast compared to non-eager.
# TODO: It could be that in eager mode, the weights are not loaded. Need to verify
# TODO: this.
# tf.enable_eager_execution()



def parse_model_cfg(nclasses, sub_models, **kwargs):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param inputs: Input tensor
    :param include_yolo_head: Includes the YOLO head
    :return: A list of output layers and the network config
    """

    inputs = Input(shape=(None, None, 3))

    x, layers, yolo_layers = inputs, [], []
    ptr = 0
    for model in sub_models:
        for layer_config in model['layers_config']:
            layer_type = layer_config['type']

            if layer_type == 'convolutional':
                ngrids = kwargs['ngrids'] # needed to find filters by eval str - find below
                decay = kwargs['decay']
                if isinstance(layer_config['filters'], str):
                    xy_field = kwargs['xy_field'] # needed to find filters by eval str - find below
                    wh_field= kwargs['wh_field'] # needed to find filters by eval str - find below
                    obj_field= kwargs['obj_field'] # needed to find filters by eval str - find below

                    layer_config['filters'] = eval(layer_config['filters'])
                x, layers, yolo_layers, ptr = _build_conv_layer(x, layer_config, layers, yolo_layers, ptr, decay)

            elif layer_type == 'shortcut':
                x, layers, yolo_layers, ptr = _build_shortcut_layer(x, layer_config, layers, yolo_layers, ptr)

            elif layer_type == 'yolo':
                ngrids = kwargs['ngrids']
                x, layers, yolo_layers, ptr = _build_yolo_layer(x, nclasses, layers, yolo_layers, ptr, ngrids)

            elif layer_type == 'route':
                x, layers, yolo_layers, ptr = _build_route_layer(x, layer_config, layers, yolo_layers, ptr)

            elif layer_type == 'upsample':
                x, layers, yolo_layers, ptr = _build_upsample_layer(x, layer_config, layers, yolo_layers, ptr)

            elif layer_type == 'maxpool':
                x, layers, yolo_layers, ptr = _build_maxpool_layer(x, layer_config, layers, yolo_layers, ptr)

            else:
                raise ValueError('{} not recognized as layer_config type'.format(layer_type))
        if 0:
            _verify_weights_completed_consumed(ptr)

    output_layers = yolo_layers

    return output_layers, layers, inputs


# def _read_net_config(layer_config):
#     decay = float(layer_config['decay'])
#
#     return {
#         'decay': decay
#     }


def _build_conv_layer(x, layer_config, layers, outputs, ptr, decay):
    stride = int(layer_config['stride'])
    filters = int(layer_config['filters'])
    kernel_size = int(layer_config['size'])
    pad = int(layer_config['pad'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    use_batch_normalization = 'batch_normalize' in layer_config

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               use_bias=not use_batch_normalization,
               activation='linear',
               kernel_regularizer=l2(decay))(x)

    if use_batch_normalization:
        x = BatchNormalization()(x)

    assert layer_config['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(
        layer_config['activation'])

    if layer_config['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers, outputs, ptr


def _build_upsample_layer(x, layer_config, layers, outputs, ptr):
    stride = int(layer_config['stride'])
    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers, outputs, ptr


def _build_maxpool_layer(x, layer_config, layers, outputs, ptr):
    stride = int(layer_config['stride'])
    size = int(layer_config['size'])

    x = MaxPooling2D(pool_size=(size, size),
                     strides=(stride, stride),
                     padding='same')(x)
    layers.append(x)

    return x, layers, outputs, ptr


def _build_route_layer(_x, layer_config, layers, outputs, ptr):
    selected_layers = [layers[int(l)] for l in layer_config['layers']]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers, outputs, ptr

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers, outputs, ptr

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _build_shortcut_layer(x, layer_config, layers, outputs, ptr):
    from_layer = layers[int(layer_config['from'])]
    x = Add()([from_layer, x])
    assert layer_config['activation'] == 'linear', 'Invalid activation: {}'.format(layer_config['activation'])
    layers.append(x)

    return x, layers, outputs, ptr


XY_FEILD = 2
WH_FEILD = 2
OBJ_FIELD = 1


def _build_yolo_layer(x, nclasses, layers, outputs, ptr, ngrids):
    x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                        ngrids,
                                        nclasses + (XY_FEILD + WH_FEILD + OBJ_FIELD))))(
        x)
    layers.append(x)
    outputs.append(x)
    return x, layers, outputs, ptr


if __name__ == '__main__':
    import yaml

    inputs = Input(shape=(None, None, 3))
    model_config_file = 'config/yolov3_model.yaml'

    with open(model_config_file, 'r') as stream:
        model_config = yaml.safe_load(stream)
    nclasses = 3

    output_layers, layers, inputs = parse_model_cfg(nclasses, **model_config)

    model = Model(inputs, output_layers)

    model.summary()
