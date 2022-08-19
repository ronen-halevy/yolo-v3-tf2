import tensorflow as tf
from tensorflow.keras import Input
import numpy as np
import yaml
import argparse


class Convert:

    @staticmethod
    def find_next_layer(model, layer_name):
        for layer_index, layer in enumerate(model.layers):
            if 'input' in layer.name:  # input is the first layer, so never the next
                continue
            try:
                if hasattr(layer, 'input'):
                    if isinstance(layer.input, list):
                        for input_layer in layer.input:
                            if input_layer.name.startswith(f'{layer_name}'):
                                return layer
                    else:
                        if layer.input.name.startswith(f'{layer_name}'):
                            return layer

            except Exception as e:
                print(e)

        return None

    def load_conv_layer(self, model, layer, wf):
        print(layer.name)
        bn = False
        next_layer = self.find_next_layer(model, layer.name)

        filters = layer.filters
        size = layer.kernel_size[0]
        in_dim = layer.get_input_shape_at(0)[-1]

        if next_layer and next_layer.name.startswith('batch_norm'):
            print(next_layer.name)

            bn = True
            bn_weights = np.fromfile(
                wf, dtype=np.float32, count=4 * filters)
            # tf [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        conv_shape = (filters, in_dim, size, size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        conv_weights = conv_weights.reshape(
            conv_shape).transpose([2, 3, 1, 0])

        if bn:
            layer.set_weights([conv_weights])
            next_layer.set_weights(bn_weights)
        else:
            layer.set_weights([conv_weights, conv_bias])

    def load_sub_model_layers(self, model, wf):
        count = 0
        for idx in range(len(model.layers)):
            if 'conv2d' in model.layers[idx].name:
                self.load_conv_layer(model, model.layers[idx], wf)
                count = count + 1
        return count

    def load_sub_model(self, sub_model, next_conv_layer_name, wf):
        conv_layers_count = 0
        for jdx in range(len(sub_model.layers)):
            if next_conv_layer_name == sub_model.layers[jdx].name:
                conv_layers_count = self.load_sub_model_layers(sub_model, wf)
                print(f'Sub-model name: {sub_model.name}, Conv Layers Loaded: {conv_layers_count}')
                break
        return conv_layers_count

    def load_darknet_weights(self, model, weights_file):
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        search_conv_layers = True
        next_conv_layer_name = 'conv2d'

        next_index = 0
        while (search_conv_layers == True):
            conv_layers_count = 0
            for idx in range(len(model.layers)):
                if hasattr(model.layers[idx], 'layers'):
                    conv_layers_count = self.load_sub_model(model.layers[idx], next_conv_layer_name, wf)
                    if conv_layers_count:
                        next_index += conv_layers_count
                        next_conv_layer_name = f'conv2d_{next_index}'
                        break
                elif model.layers[idx].name == next_conv_layer_name:
                    self.load_conv_layer(model, model.layers[idx], wf)
                    conv_layers_count = 1
                    next_index += conv_layers_count
                    next_conv_layer_name = f'conv2d_{next_index}'
                    print(f'Descrete Layer name: {model.layers[idx].name}, Loaded')
                    break
            if conv_layers_count == 0:
                search_conv_layers = False

        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='utilities/convert_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        convert_config = yaml.safe_load(stream)

    nclasses = convert_config['num_classes']
    weights_file = convert_config['weights_file']
    output_weights_file = convert_config['output_weights_file']
    model_config_file = convert_config['model_config_file']

    from core.parse_model import ParseModel
    parse_model = ParseModel()

    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    inputs = Input(shape=(None, None, 3))
    model = parse_model.build_model(inputs, nclasses, **model_config)

    convert = Convert()
    model = convert.load_darknet_weights(model, weights_file)
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    _ = model(img)
    print('sanity check passed')
    model.save_weights(output_weights_file)
    print('weights saved')


if __name__ == "__main__":
    main()
