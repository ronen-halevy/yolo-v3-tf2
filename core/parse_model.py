from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras import Input, Model
import yaml


def parse_model_cfg(nclasses, model_config_file):
    """
    Builds model by parsing model config file

    :param nclasses: Num of dataset classes. Needed by head layers, where field size is according to ncllasses
    :type nclasses: int
    :param model_config_file: yaml conf file which holds the model definitions, according which parser builds model
    :type model_config_file: filename str
    :return: output_layers: edge output layerss (list), layers: list which appends all model's layers while created.
    :rtype:  lists
    """

    with open(model_config_file, 'r') as stream:
        model_config = yaml.safe_load(stream)
    sub_models = model_config['sub_models']
    decay = model_config['decay']
    xy_field = model_config['xy_field']
    wh_field = model_config['wh_field']
    obj_field = model_config['obj_field']
    ngrids = model_config['ngrids']

    inputs = Input(shape=(None, None, 3))

    output_layers = []
    layers = []
    x = inputs
    for model in sub_models:
        for layer_conf in model['layers_config']:
            layer_type = layer_conf['type']

            if layer_type == 'convolutional':
                if isinstance(layer_conf['filters'], str): # eval('ngrids*(xy_field+wh_field+obj_field+nclasses)')
                    layer_conf['filters'] = eval(layer_conf['filters'])
                x, layers = _parse_convolutional(x, layer_conf, layers, decay)


            elif layer_type == 'shortcut':
                x, layers = _parse_shortcut(x, layer_conf, layers)

            elif layer_type == 'yolo':
                x, layers, output_layers = _parse_yolo(x, nclasses, layers, output_layers, ngrids, xy_field, wh_field,
                                                       obj_field)

            elif layer_type == 'route':
                x, layers = _parse_route(x, layer_conf, layers)

            elif layer_type == 'upsample':
                x, layers = _parse_upsample(x, layer_conf, layers)

            elif layer_type == 'maxpool':
                x, layers = _parse_maxpool(x, layer_conf, layers)

            else:
                raise ValueError('{} not recognized as layer_conf type'.format(layer_type))

    return output_layers, layers, inputs

def _parse_convolutional(x, layer_conf, layers, decay):
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
    stride = int(layer_conf['stride'])
    filters = int(layer_conf['filters'])
    kernel_size = int(layer_conf['size'])
    pad = int(layer_conf['pad'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    use_batch_normalization = 'batch_normalize' in layer_conf

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

    assert layer_conf['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(
        layer_conf['activation'])

    if layer_conf['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers


def _parse_upsample(x, layer_conf, layers):
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
    stride = int(layer_conf['stride'])
    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers


def _parse_maxpool(x, layer_conf, layers):
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
    stride = int(layer_conf['stride'])
    size = int(layer_conf['size'])

    x = MaxPooling2D(pool_size=(size, size),
                     strides=(stride, stride),
                     padding='same')(x)
    layers.append(x)

    return x, layers


def _parse_route(_x, layer_conf, layers):
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
    selected_layers = [layers[int(l)] for l in layer_conf['layers']]

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


def _parse_shortcut(x, layer_conf, layers):
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
    from_layer = layers[int(layer_conf['from'])]
    x = Add()([from_layer, x])
    assert layer_conf['activation'] == 'linear', 'Invalid activation: {}'.format(layer_conf['activation'])
    layers.append(x)

    return x, layers


def _parse_yolo(x, nclasses, layers, output_layers, ngrids, xy_field, wh_field, obj_field):
    """

    :param x:
    :type x:
    :param nclasses:
    :type nclasses:
    :param layers:
    :type layers:
    :param output_layers:
    :type output_layers:
    :param ngrids:
    :type ngrids:
    :param xy_field:
    :type xy_field:
    :param wh_field:
    :type wh_field:
    :param obj_field:
    :type obj_field:
    :return:
    :rtype:
    """
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
    output_layers.append(x)
    return x, layers, output_layers


if __name__ == '__main__':
    import yaml

    model_conf_file = '../config/yolov3_model.yaml'

    with open(model_conf_file, 'r') as _stream:
        _model_config = yaml.safe_load(_stream)
    classes = 3

    _output_layers, _layers, _inputs = parse_model_cfg(classes, **_model_config)

    _model = Model(_inputs, _output_layers)

    _model.summary()
    with open("model_summary.txt", "w") as file1:
        _model.summary(print_fn=lambda x: file1.write(x + '\n'))
