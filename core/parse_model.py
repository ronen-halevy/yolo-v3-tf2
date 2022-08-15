from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D
from tensorflow.keras import Input, Model
import yaml


class ParseModel:

    def _find_sub_model_by_name(self, sub_models, name):
        for sub_model in sub_models:
            if sub_model['name'] == name:
                return sub_model
        return None

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
        stride = int(layer_conf['stride'])
        size = int(layer_conf['size'])

        x = MaxPooling2D(pool_size=(size, size),
                         strides=(stride, stride),
                         padding='same')(x)
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
            selected_layers = [layers[int(l)] for l in layer_conf['source']['layers']]

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
        return x, layers

    def create_inputs(self, inputs_config, sub_models):
        # inputs_config = sub_model_config['inputs']
        if 'shape' in inputs_config:
            inputs = Input(eval(inputs_config['shape']))
            data_inputs = inputs
        else:
            inputs = []
            data_inputs = []
            for idx, source_entry in enumerate(inputs_config['source']):
                source_sub_model = self._find_sub_model_by_name(sub_models, source_entry['name'])
                if source_sub_model is None:
                    raise Exception(f'Error: sub-model {source_entry["name"]} not found')
                source_entry_index = source_entry.get('entry_index', 0)
                if isinstance(source_sub_model['outputs'], list):
                    inputs.append(Input(source_sub_model['outputs'][source_entry_index].shape[1:],
                                    name='{entry["name"]}_to_{sub_model_config["name""]}_{source_entry_index}'))
                    data_inputs.append(source_sub_model['outputs'][source_entry_index])
                else:
                    inputs.append(Input(source_sub_model['outputs'].shape[1:],
                                        name='{entry["name"]}_to_{sub_model_config["name""]}'))
                    data_inputs.append(source_sub_model['outputs'])

            if len(inputs) == 1:
                inputs = inputs[0]
                # data_inputs = data_inputs[0]
        return inputs, data_inputs

    def _create_layers(self, layers_config_file, inputs, nclasses, decay_factor):
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

    def build_model(self, nclasses, decay_factor, sub_models_configs):
        """
        Builds model by parsing model config file

        :param nclasses: Num of dataset classes. Needed by head layers, where field size is according to ncllasses
        :type nclasses: int
        :param model_config_file: yaml conf file which holds the model definitions, according which parser builds model
        :type model_config_file: filename str
        :return: output_layers: edge output layerss (list), layers: list which appends all model's layers while created.
        :rtype:  lists
        """

        # sub_models_configs = model_config['sub_models']
        # sub_modules = _find_sub_model_ny_name()
        sub_models_entries = []
        sub_models = []
        first_sub_model_inputs = None

        for sub_model_config in sub_models_configs:

            inputs_config = sub_model_config['inputs']
            inputs, data_inputs = self.create_inputs(inputs_config, sub_models)

            if first_sub_model_inputs is None:
                first_sub_model_inputs = {'inputs': inputs, 'data_inputs': data_inputs}

            layers = self._create_layers(sub_model_config['layers_config_file'], inputs, nclasses, decay_factor)
            outputs = [layers[int(l)] for l in sub_model_config['outputs_layers']]
            outputs = outputs[0] if len(outputs) == 0 else outputs
            model = Model(inputs, outputs, name= sub_model_config['name'])(data_inputs)
            sub_models.append({'outputs': model, 'name': sub_model_config['name']})
        outputs = [sub_model_entry['outputs'] for  sub_model_entry in  sub_models if 'head' in sub_model_entry['name'] ]
        inputs = first_sub_model_inputs['inputs']
        data_inputs = first_sub_model_inputs['data_inputs']

        model = Model(inputs, outputs, name="yolo") # (data_inputs)
        model.summary()
        return model, data_inputs


if __name__ == '__main__':
    model_conf_file = '../config/models/yolov3/model.yaml'

    with open(model_conf_file, 'r') as _stream:
        _model_config = yaml.safe_load(_stream)
    classes = 3
    parse_model = ParseModel()
    _model, inputs = parse_model.build_model(classes, **_model_config)

    # _model = Model(_inputs, _output_layers)

    _model.summary()
    with open("model_summary.txt", "w") as file1:
        _model.summary(print_fn=lambda x: file1.write(x + '\n'))
