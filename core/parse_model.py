from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D, Reshape, Activation
from tensorflow.keras import Input, Model
import yaml


class ParseModel:

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        stride_xy = list(map(int, layer_conf['stride_xy']))
        size_xy = list(map(int, layer_conf['size_xy']))
        padding = layer_conf['padding']

        x = tf.keras.layers.MaxPooling2D(pool_size=size_xy,
                         strides=stride_xy,
                         padding=padding)(x)
        layers.append(x)

        return x, layers

    @staticmethod
    def _parse_route(layer_conf, inputs_entry, layers):
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
        selected_layers = []
        if 'layers' in layer_conf['source']:
            selected_layers = [layers[int(layer)] for layer in layer_conf['source']['layers']]

        selected_inputs = []
        if 'inputs' in layer_conf['source']:
            if isinstance(inputs_entry, list):
                selected_inputs = [inputs_entry[idx] for idx in layer_conf['source']['inputs']]
            else:
                selected_inputs = [inputs_entry]

        selected_layers.extend(selected_inputs)

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

    @staticmethod
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

    @staticmethod
    def _parse_yolo(x, nclasses, layers):
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

        ngrids, xy_field, wh_field, obj_field = 3, 2, 2, 1

        x = Reshape((tf.shape(x)[1], tf.shape(x)[2],
                                            ngrids, nclasses + (xy_field + wh_field + obj_field)))(x)
        pred_xy, pred_wh, pred_obj, class_probs = Lambda(lambda xx: tf.split(xx, (2, 2, 1, nclasses), axis=-1))(x)
        pred_xy = Activation(tf.nn.sigmoid)(pred_xy)
        pred_obj = Activation(tf.nn.sigmoid)(pred_obj)
        class_probs = Activation(tf.nn.sigmoid)(class_probs)
        x = tf.keras.layers.concatenate([pred_xy, pred_wh, pred_obj, class_probs], axis=-1)
        layers.append(x)
        return x, layers

    @staticmethod
    def create_sub_model_inputs(inputs_config, sub_models_outputs_list, models_name):
        if 'shape' in inputs_config:
            inputs = Input(eval(inputs_config['shape']))
            data_inputs = inputs
        else:
            inputs = []
            data_inputs = []
            for idx, source_entry in enumerate(inputs_config['source']):
                selected_sourcing_sub_model = list(
                    filter(lambda x: x['name'] == source_entry['name'], sub_models_outputs_list))
                selected_sourcing_sub_model = selected_sourcing_sub_model[0]  # a single selected model - name is unique
                if not len(selected_sourcing_sub_model):
                    raise Exception(f'Error: sub-model {source_entry["name"]} not found')
                selected_sourcing_output = selected_sourcing_sub_model['outputs']

                source_entry_index = source_entry.get('entry_index', 0)
                if isinstance(selected_sourcing_output, list):
                    inputs.append(Input(selected_sourcing_output[source_entry_index].shape[1:],
                                        name=f'{source_entry["name"]}_to_{models_name}_{source_entry_index}'))
                    data_inputs.append(selected_sourcing_output[source_entry_index])
                else:
                    inputs.append(Input(selected_sourcing_output.shape[1:],
                                        name=f'{source_entry["name"]}_to_{models_name}_{source_entry_index}'))
                    data_inputs.append(selected_sourcing_output)

            if len(inputs) == 1:
                inputs = inputs[0]
                # data_inputs = data_inputs[0]
        return inputs, data_inputs

    def create_sub_model_layers(self, layers_config_file, inputs, nclasses, decay_factor):
        with open(layers_config_file, 'r') as stream:
            model_config = yaml.safe_load(stream)

        x = inputs
        layers = []
        for layer_conf in model_config['layers_config']:
            layer_type = layer_conf['type']

            if layer_type == 'convolutional':
                if isinstance(layer_conf['filters'], str):  # eval('3*(2+2+1+nclasses)')
                    layer_conf['filters'] = eval(layer_conf['filters'])
                x, layers = self._parse_convolutional(x, layer_conf, layers, decay_factor)
            elif layer_type == 'shortcut':
                x, layers = self._parse_shortcut(x, layer_conf, layers)

            elif layer_type == 'yolo':
                x, layers = self._parse_yolo(x, nclasses, layers)

            elif layer_type == 'route':
                x, layers = self._parse_route(layer_conf, inputs, layers)

            elif layer_type == 'upsample':
                x, layers = self._parse_upsample(x, layer_conf, layers)

            elif layer_type == 'maxpool':
                x, layers = self._parse_maxpool(x, layer_conf, layers)

            else:
                raise ValueError('{} not recognized as layer_conf type'.format(layer_type))
        return layers
    def build_model(self, model_inputs, sub_models_configs, output_stage='head', decay_factor=0, nclasses = 0, **kwargs):
        """
        Builds model by parsing model config file

        :param nclasses: Num of dataset classes. Needed by head layers, where field size is according to ncllasses
        :type nclasses: int
        :param model_config_file: yaml conf file which holds the model definitions, according which parser builds model
        :type model_config_file: filename str
        :return: output_layers: edge output layerss (list), layers: list which appends all model's layers while created.
        :rtype:  lists
        """

        sub_models_outputs_list = []

        for sub_model_config in sub_models_configs:
            inputs_config = sub_model_config.get('inputs')
            if inputs_config:
                # locate peers' output according to configuration
                sub_model_inputs, sub_model_data_inputs = self.create_sub_model_inputs(inputs_config, sub_models_outputs_list,
                                                         sub_model_config['name'])
            else:  # peerles bottom model (leftmost) uses model_inputs
                sub_model_inputs = sub_model_data_inputs = model_inputs

            sub_model_layers = self.create_sub_model_layers(sub_model_config['layers_config_file'], sub_model_inputs, nclasses, decay_factor)

            sub_model_outputs = [sub_model_layers[int(layer)] for layer in sub_model_config['outputs_layers']]
            sub_model_outputs = sub_model_outputs[0] if len(sub_model_outputs) == 0 else sub_model_outputs

            model = Model(sub_model_inputs, sub_model_outputs, name=sub_model_config['name'])(sub_model_data_inputs)
            sub_models_outputs_list.append({'outputs': model, 'name': sub_model_config['name']})

        model_outputs = [sub_model_entry['outputs'] for sub_model_entry in sub_models_outputs_list if
                   output_stage in sub_model_entry['name']]

        model = Model(model_inputs, model_outputs, name="yolo")
        return model

    def create_model(self, nclasses, model_config_file):
        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)
        parse_model = ParseModel()
        inputs = Input(shape=(None, None, 3), name='input')
        model = parse_model.build_model(inputs, nclasses, **model_config)
        return model


if __name__ == '__main__':
    model_conf_file = '../config/models/yolov3/model.yaml'

    with open(model_conf_file, 'r') as _stream:
        _model_config = yaml.safe_load(_stream)
    classes = 3
    parse_model = ParseModel()
    _model, _inputs = parse_model.build_model(classes, **_model_config)

    with open("model_summary.txt", "w") as file1:
        _model.summary(print_fn=lambda x: file1.write(x + '\n'))
