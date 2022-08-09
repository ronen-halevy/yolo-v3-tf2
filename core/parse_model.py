from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras import Input, Model


def parse_model_cfg(nclasses, sub_models, **kwargs):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param inputs: Input tensor
    :param include_yolo_head: Includes the YOLO head
    :return: A list of output layers and the network config
    """

    inputs = Input(shape=(None, None, 3))

    x, layers, yolo_layers = inputs, [], []
    for model in sub_models:
        for layer_config in model['layers_config']:
            layer_type = layer_config['type']

            if layer_type == 'convolutional':
                x, layers = _parse_convolutional(x, layer_config, layers, nclasses, **kwargs)

            elif layer_type == 'shortcut':
                x, layers = _parse_shortcut(x, layer_config, layers)

            elif layer_type == 'yolo':
                x, layers, output_layers = _parse_yolo(x, nclasses, layers, yolo_layers, **kwargs)

            elif layer_type == 'route':
                x, layers = _parse_route(x, layer_config, layers)

            elif layer_type == 'upsample':
                x, layers = _parse_upsample(x, layer_config, layers)

            elif layer_type == 'maxpool':
                x, layers = _parse_maxpool(x, layer_config, layers)

            else:
                raise ValueError('{} not recognized as layer_config type'.format(layer_type))

    return output_layers, layers, inputs


def _parse_convolutional(x, layer_config, layers, nclasses, decay, xy_field, wh_field, obj_field, ngrids, **kwargs):
    if isinstance(layer_config['filters'], str):
        layer_config['filters'] = eval(layer_config['filters'])

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

    return x, layers


def _parse_upsample(x, layer_config, layers):
    stride = int(layer_config['stride'])
    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers


def _parse_maxpool(x, layer_config, layers):
    stride = int(layer_config['stride'])
    size = int(layer_config['size'])

    x = MaxPooling2D(pool_size=(size, size),
                     strides=(stride, stride),
                     padding='same')(x)
    layers.append(x)

    return x, layers


def _parse_route(_x, layer_config, layers):
    selected_layers = [layers[int(l)] for l in layer_config['layers']]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _parse_shortcut(x, layer_config, layers):
    from_layer = layers[int(layer_config['from'])]
    x = Add()([from_layer, x])
    assert layer_config['activation'] == 'linear', 'Invalid activation: {}'.format(layer_config['activation'])
    layers.append(x)

    return x, layers


def _parse_yolo(x, nclasses, layers, outputs, ngrids, xy_field, wh_field, obj_field, **kwargs):
    x = Lambda(lambda xx: tf.reshape(xx, (-1, tf.shape(xx)[1], tf.shape(xx)[2],
                                        ngrids,
                                        nclasses + (xy_field + wh_field + obj_field))))(
        x)
    pred_xy, pred_wh, pred_obj, class_probs = Lambda(lambda xx: tf.split(xx, (2, 2, 1, nclasses), axis=-1))(x)
    pred_xy = Lambda(lambda xx: tf.sigmoid(xx))(pred_xy)
    pred_obj = Lambda(lambda xx: tf.sigmoid(xx))(pred_obj)
    class_probs = Lambda(lambda xx: tf.sigmoid(xx))(class_probs)
    x = Lambda(lambda xx: tf.concat([xx[0], xx[1], xx[2], xx[3]], axis=-1))((pred_xy, pred_wh, pred_obj,
                                                                        class_probs))

    layers.append(x)
    outputs.append(x)
    return x, layers, outputs


if __name__ == '__main__':
    import yaml

    inputs = Input(shape=(None, None, 3))
    model_config_file = '../config/yolov3_model.yaml'

    with open(model_config_file, 'r') as stream:
        model_config = yaml.safe_load(stream)
    classes = 3

    output_layers, layers, inputs = parse_model_cfg(classes, **model_config)

    model = Model(inputs, output_layers)

    model.summary()
    with open("model_summary.txt", "w") as file1:
        model.summary(print_fn=lambda x: file1.write(x + '\n'))
