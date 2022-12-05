import tensorflow as tf
from tensorflow.keras import Input
import numpy as np
import yaml
import argparse

from core.parse_model import ParseModel


class Convert:

    @staticmethod
    # The next layer is the one which points to current layer in its 'input' configuration entry.
    # So loop on model (or sub_models) layer. Skipping 'input' layer which is definetly not the next, if a layer is
    # sourced by a listof inputs - loop on the list to find a match. If a single input - just check for a match
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


    def set_conv_layer_weights(self, model, layer, weights_file_ref):
        # check if next layer is a batch_normalization.
        # If so, then load 4 words for each filter's bn then filters' wheights. Otherwise load filters' wheights
        # followed by filters' b byand then the  followed by  first.  , no bias-(according to file format..)

        bn = False
        next_layer = self.find_next_layer(model, layer.name)

        filters = layer.filters
        size = layer.kernel_size[0]
        in_dim = layer.get_input_shape_at(0)[-1]
        if next_layer and next_layer.name.startswith('batch_norm'):

            bn = True
            bn_weights = np.fromfile(
                weights_file_ref, dtype=np.float32, count=4 * filters)
            print(f'bn {filters * 4}')

            # tf [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
        # if no bn: read bias and then weights
        else:
            conv_bias = np.fromfile(weights_file_ref, dtype=np.float32, count=filters)
            print(f'bias {filters}')

        conv_shape = (filters, in_dim, size, size)

        conv_weights = np.fromfile(
            weights_file_ref, dtype=np.float32, count=np.product(conv_shape))
        print(f'weights {conv_shape}')

        conv_weights = conv_weights.reshape(
            conv_shape).transpose([2, 3, 1, 0])

        if bn:
            layer.set_weights([conv_weights])
            next_layer.set_weights(bn_weights)
        else:
            layer.set_weights([conv_weights, conv_bias])

    def set_weights_to_sub_model_layers(self, sub_model, weights_file_ref):
        count = 0
        for idx in range(len(sub_model.layers)):
            # Assumed  conv layer hame includes 'conv2d' :
            if 'conv2d' in sub_model.layers[idx].name:
                self.set_conv_layer_weights(sub_model, sub_model.layers[idx], weights_file_ref)
                count = count + 1
        return count

    def load_sub_model(self, sub_model, next_conv_layer_name, weights_file_ref):
        conv_layers_count = 0
        for jdx in range(len(sub_model.layers)):
            if next_conv_layer_name == sub_model.layers[jdx].name:
                conv_layers_count = self.set_weights_to_sub_model_layers(sub_model, weights_file_ref)
                break
        return conv_layers_count

    def load_all_weights(self, model, weights_file_name):
        weights_file_ref = open(weights_file_name, 'rb')
        major, minor, revision, seen, _ = np.fromfile(weights_file_ref, dtype=np.int32, count=5)
        search_conv_layers = True
        # first conv layer assumed, and then  conv2d_{index} where index 1: count
        next_conv_layer_name = 'conv2d'

        next_index = 0
        # Loop on model's layers. Each layer may either be a sub_model with a list of layers or a single layer.
        # Goal is to load wheights to conv layers. The model's first conv layer is named 'conv2d', and rest of conv
        # layers  are numbered i.e. 'conv2d_1', 'conv2d_2' etc. Weights should be loaded consecutively. Sub models list
        # is not necessarily arranged in this order. So the loop looks for a match with next_conv_layer_name, loads
        # weights, breaks the for loop, which will next restart.
        # Example: Submodels arrangement: [[conv2d, conv2d_1], [conv2d_5], [conv2d_2, conv2d_3, conv2d_4]] then:
        # next_conv_layer_name initialized to 'conv2d'.  Loop will match with first entry, set wheights to conv2d,
        # set next_conv_layer_name= 'conv2d_2' and break. While-ing to the for-loop again will hit 3rd entry, load
        # weights to conv2d_2, conv2d_3, conv2d_4, set next_conv_layer_name= 'conv2d_5' and break.  While-ing to
        # for-loop, the second list entry is hitted, set weights to conv2d_5. Whiling again, find no hit, setting
        # next_conv_layer_name=0, while loop let go.
        while (search_conv_layers == True):
            conv_layers_count = 0
             # Loop on sub-models, each holds a list of layers:
            for idx in range(len(model.layers)):
                if hasattr(model.layers[idx], 'layers'):
                    # Load sub model's layer. returns num of conv layers in sub model
                    conv_layers_count = self.load_sub_model(model.layers[idx], next_conv_layer_name, weights_file_ref)
                    if conv_layers_count:
                        print(f'model.layers[idx].name: {model.layers[idx].name} Layers in Sub model: '
                              f'{len(model.layers[idx].layers)} Conv Layers: {conv_layers_count}')

                        next_index += conv_layers_count
                        next_conv_layer_name = f'conv2d_{next_index}'
                        break
                # elif corresponds to potentially layers configured not within a submodel of layer but descrertly
                # (if a conv layer - set weights:)
                elif model.layers[idx].name == next_conv_layer_name:
                    self.set_conv_layer_weights(model, model.layers[idx], weights_file_ref)
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
    weights_file_name = convert_config['weights_file']
    output_weights_file = convert_config['output_weights_file']
    model_config_file = convert_config['model_config_file']

    parse_model = ParseModel()

    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    inputs = Input(shape=(None, None, 3))
    model = parse_model.build_model(inputs, nclasses=nclasses, **model_config)
    model.summary()
    # model = parse_model.build_model(inputs, sub_models_configs, output_stage, decay_factor, nclasses)

    convert = Convert()
    model = convert.load_all_weights(model, weights_file_name)
    img = np.random.random((1, 416, 416, 3)).astype(np.float32)
    _ = model(img)
    print('sanity check passed')
    model.save_weights(output_weights_file)
    print(f'weights saved to {output_weights_file}')


if __name__ == "__main__":
    main()
