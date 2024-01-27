# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras import Input, Model
import yaml

import argparse
import sys
from copy import deepcopy
from pathlib import Path
import yaml  # for torch hub

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np
import tensorflow as tf

from tensorflow import keras

from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior() # allows running NumPy code, accelerated by TensorFlow
from keras import mixed_precision


# from models.common import (C3, SPP, SPPF, Bottleneck, BottleneckCSP, C3x, Concat, Conv, CrossConv, DWConv,
#                            DWConvTranspose2d, Focus, autopad)
# from models.experimental import  attempt_load
# from models.yolo import Detect, Segment
# from utils.tf_general import LOGGER, make_divisible, print_args

# from utils.tf_plots import feature_visualization
# def _parse_route(layer_conf, inputs_entry, layers):

def _parse_detect(x, layers, nc):
    # x = tf.keras.layers.Reshape((x, x.shape[1],
    #                              3, -1))(x)
    layers.append(x)
    return x, layers


def _parse_decoder(x, layers, decay, nc, grid_size):
    """

    :param x:
    :type x:
    :param nclasses:
    :type nclasses:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """
    activation = False
    bn = False
    kernel_size = 1
    stride = 1
    nanchors, xy_field, wh_field, obj_field = 3, 2, 2, 1
    no = (nc + xy_field + wh_field + obj_field)
    filters = nanchors * no
    x, _ = _parse_convolutional(x, layers, decay, bn, activation, filters, kernel_size, stride, pad=1)
    x = tf.reshape(x, [-1, grid_size, grid_size, nanchors, no])

    # # x = Lambda(lambda xx: tf.reshape(xx, (-1, tf.shape(xx)[1], tf.shape(xx)[2],
    # #                                       ngrids,
    # #                                       nclasses + (xy_field + wh_field + obj_field))))(
    # #     x)
    # x = tf.keras.layers.Reshape((grid_size, grid_size,
    #                              ngrids, nclasses + (xy_field + wh_field + obj_field)))(x)
    #
    # # pred_xy, pred_wh, pred_obj, class_probs = Lambda(lambda xx: tf.split(xx, (2, 2, 1, nclasses), axis=-1))(x)
    # pred_xy, pred_wh, pred_obj, class_probs = tf.split(x, (2, 2, 1, nclasses), axis=-1)
    #
    #
    # # pred_xy = Lambda(lambda xx: tf.sigmoid(xx))(pred_xy)
    # # pred_obj = Lambda(lambda xx: tf.sigmoid(xx))(pred_obj)
    # # class_probs = Lambda(lambda xx: tf.sigmoid(xx))(class_probs)
    #
    # # x = Lambda(lambda xx: tf.concat([xx[0], xx[1], xx[2], xx[3]], axis=-1))((pred_xy, pred_wh, pred_obj,
    # #                                                                          class_probs))
    # # new:
    # pred_xy = tf.keras.activations.sigmoid(pred_xy)
    # pred_obj = tf.keras.activations.sigmoid(pred_obj)
    # class_probs = tf.keras.activations.sigmoid(class_probs)
    #
    # x = tf.keras.layers.concatenate(
    #     [pred_xy, pred_wh, pred_obj,
    #     class_probs], axis=-1
    # )
    #

    # x = tf.keras.layers.Reshape((grid_size, grid_size,
    #                              ngrids, nclasses + (xy_field + wh_field + obj_field)))(x)

    layers.append(x)
    return x, layers


def _parse_maxpool(x, layers, pool_size, stride_xy, pad='same'):
    """

    :param x:
    :type x:
    :param layer_conf:
    :type layer_conf:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """
    padding = pad

    x = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                     strides=stride_xy,
                                     padding=padding)(x)
    layers.append(x)

    return x, layers


def _parse_route(x, layers):
    """

    :param _x:
    :type _x:
    :param layer_conf:
    :type layer_conf:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """
    # selected_layers = []
    # # route source can be 'inputs' and layers. arrange all. Concatenate input sources if there are more than a single source
    # if 'layers' in layer_conf['source']:
    #     selected_layers = [layers[int(layer)] for layer in layer_conf['source']['layers']]
    #
    # selected_inputs = []
    # if 'inputs' in layer_conf['source']:
    #     if isinstance(inputs_entry, list):
    #         selected_inputs = [inputs_entry[idx] for idx in layer_conf['source']['inputs']]
    #     else:
    #         selected_inputs = [inputs_entry]
    #
    # selected_layers.extend(selected_inputs)
    #
    # if len(selected_layers) == 1:
    #     x = selected_layers[0]
    #     layers.append(x)
    #     return x, layers

    # elif len(selected_layers) == 2:
    x = Concatenate(axis=3)(x)
    layers.append(x)

    return x, layers


def _parse_upsample(x, layers, stride):
    """

    :param x:
    :type x:
    :param layer_conf:
    :type layer_conf:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """
    # stride = int(layer_conf['stride'])
    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers


def _parse_shortcut(x, layers):
    """
    :param x:
    :type x:
    :param layer_conf:
    :type layer_conf:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """
    # from_layer = layers[int(layer_conf['from'])]
    x = Add()(x)
    # assert layer_conf['activation'] == 'linear', 'Invalid activation: {}'.format(layer_conf['activation'])
    layers.append(x)

    return x, layers


def _parse_convolutional(x, layers, decay, bn, activation, filters, kernel_size, stride, pad=1
                         ):
    """

    :param x:
    :type x:
    :param layer_conf:
    :type layer_conf:
    :param layers:
    :type layers:
    :param decay:
    :type decay:
    :return:
    :rtype:
    """
    # stride = int(layer_conf['stride'])
    # filters = int(layer_conf['filters'])
    # kernel_size = int(layer_conf['size'])
    # pad = int(layer_conf['pad'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    # 'batch_normalize' in layer_conf

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               use_bias=not bn,
               activation='linear',
               kernel_regularizer=l2(decay))(x)

    if bn:
        x = BatchNormalization()(x)

    # assert layer_conf['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(
    #     layer_conf['activation'])
    #
    # if layer_conf['activation'] == 'leaky':
    if activation:
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers


def parse_model(inputs, na, nc, mlist, ch, imgsz, decay_factor):  # model_dict, input_channels(3)

    """
    Constructs the model by parsing model layers' configuration
    :param anchors: list[nl[na*2] of anchor sets per layer. int
    :param nc: nof classes. Needed to determine no -nof outputs, which is used to check for last stage. int
    :param gd: depth gain. A scaling factor. float
    :param gw: width gain. A scaling factor. float
    :param mlist: model layers list. A layer is a list[4] structured: [from,number dup, module name,args]
    :param ch: list of nof in channels to layers. Initiated as [3], then an entry is appended each layers loop iteration
    :param ref_model_seq: A trained ptorch ref model seq, used for offline torch to tensorflow format weights conversion
    :param imgsz: Input img size, Typ [640,640], required by detection module to find grid size as (imgsz/strides). int
    :param training: For detect layer. Bool. if False (inference & validation), output a processed tensor ready for nms.
    :return:
     model - tf.keras Sequential linear stack of layers.
     savelist: indices list of  layers indices which output is a source to a next but not adjacent (i.e. not -1) layer.
    """
    x = inputs

    # print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    y = []  # outputs
    for i, (f, n, m, args) in enumerate(mlist):  # mlist-list of layers configs. from, number, module, args
        m_str = m
        if f != -1:  # if not from previous layer
            x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]  # from earlier layers

        # if visualize:
        # m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        # n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m_str in ['Conv', 'decoder'
                             'nn.Conv2d', 'Conv', 'DWConv', 'DWConvTranspose2d', 'Bottleneck', 'SPP', 'SPPF', 'Focus',
                     'CrossConv',
                     'BottleneckCSP', 'C3', 'C3x']:
            pass
            # c2 =  args[0] # c1: nof layer's in channels, c2: nof layer's out channels
            # c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # args = [c2, *args[1:]] # [nch_in, nch_o, args]

        # elif m_str is 'nn.BatchNorm2d':
        #     args = [ch[f]]
        elif m_str == 'Concat':
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m_str in ['Detect', 'Segment']:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            # if m_str == 'Segment':
            #     args[3] = make_divisible(args[3] * gw, 8)
            args.append(imgsz)
            args.append(training)
        # else:
        #     print('\n!!!!!', m_str)
        #     c2 = ch[f]

        if m_str == 'Conv':
            bn = True;
            activation = True
            x, layers = _parse_convolutional(x, layers, decay_factor, bn, activation, *args)
        elif m_str == 'ADD':
            x, layers = _parse_shortcut(x, layers)

        elif m_str == 'upsample':
            x, layers = _parse_upsample(x, layers, *args)

        elif m_str == 'concate':
            x, layers = _parse_route(x, layers, *args)
        elif m_str == 'Maxpool':
            x, layers = _parse_maxpool(x, layers, *args)
        elif m_str == 'decoder':
            x, layers = _parse_decoder(x, layers, decay_factor, nc, *args)

        elif m_str == 'detect':
            x, layers = _parse_detect(x, layers, *args)

            # if isinstance(layer_conf['filters'], str):  # eval('3*(2+2+1+nclasses)')
            #     layer_conf['filters'] = eval(layer_conf['filters'])
            # x, layers = self._parse_convolutional(x, layer_conf, layers, decay_factor)
        # if ref_model_seq: # feed weights directly
        #     m_ = keras.Sequential([tf_m(*args, w=ref_model_seq[i][j]) for j in range(n)]) if n > 1 \
        #         else tf_m(*args, w=ref_model_seq[i])  # module
        # else:
        # m_ = keras.Sequential([tf_m(*args) for j in range(n)]) if n > 1 \
        #         else tf_m(*args)  # module
        # layers.append(m_)
        ch.append(c2)
        y.append(x)  # save output
    return layers
    # torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
    #     t = str(m)[8:-2].replace('__main__.', '')  # module type
    #     # np = sum(x.numel() for x in torch_m_.parameters())  # number params
    #     np=0
    #     m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
    #     print(f'{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}')  # print
    #     save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
    #     layers.append(m_)
    #     ch.append(c2)
    # return keras.Sequential(layers), sorted(save)


def build_model(inputs, na, nc, mlist, ch, imgsz, decay_factor):
    # na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    layers = parse_model(inputs, na, nc, mlist, ch, imgsz=imgsz, decay_factor=decay_factor)
    model = Model(inputs, layers[-1], name='model')
    return model


if __name__ == '__main__':
    cfg = '/home/ronen/devel/PycharmProjects/yolo-v3-tf2/config/models/yolov3_tiny/yolov3_tiny.yaml'
    with open(cfg) as f:
        yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

    d = deepcopy(yaml)

    mlist = d['backbone'] + d['head']
    nc = 7

    training = True
    imgsz = [416, 416]
    ch = 3
    inputs = Input(shape=(416, 416, 3))
    decay_factor = 0.01
    na = 3
    # (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors

    # layers=parse_model(inputs, na, nc, mlist, ch=[ch], imgsz=imgsz, decay_factor=decay_factor,
    #                                         )
    # model = Model(inputs, layers[-1], name='model')
    model = build_model(inputs, na, nc, mlist, ch=[ch], imgsz=imgsz, decay_factor=decay_factor)
    #
    #
    xx = tf.zeros([1, 416, 416, 3], dtype=tf.float32)
    dd = model(xx)
    print(model.summary())
