import tensorflow as tf
import numpy as np
import yaml
import argparse

from core.parse_model import parse_model_cfg


class Convert:
    @staticmethod
    def find_next_layer(model, layer_name):
        for layer_index, layer in enumerate(model.layers):
            try:
                if hasattr(layer, 'input'):
                    if isinstance(layer.input, list):
                        for input_layer in layer.input:
                            if input_layer.name.startswith(f'{layer_name}/'):
                                return layer
                    else:
                        if layer.input.name.startswith(f'{layer_name}/'):
                            return layer

            except Exception as e:
                print(e)

        return None

    # @staticmethod
    def load_darknet_weights(self, model, weights_file):
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        # layers = layers_lists
        search_conv_layers = True
        first_conv_layer_name = conv_layer_name = 'conv2d'
        index = 0

        while (search_conv_layers == True):
            try:
                layer = model.get_layer(conv_layer_name)
            except Exception as e:
                search_conv_layers = False
                continue
            bn = False
            next_layer = self.find_next_layer(model, conv_layer_name)

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if next_layer and next_layer.name.startswith('batch_norm'):
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
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if bn:
                layer.set_weights([conv_weights])
                next_layer.set_weights(bn_weights)
            else:
                layer.set_weights([conv_weights, conv_bias])

            index += 1
            conv_layer_name = f'{first_conv_layer_name}_{index}'
        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()
        print(f'Convert Completed. {index} conv elements were loaded')
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='convert_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as stream:
        convert_config = yaml.safe_load(stream)

    nclasses = convert_config['num_classes']
    weights_file = convert_config['weights_file']
    output_weights_file = convert_config['output_weights_file']
    model_config_file = convert_config['model_config_file']

    with open(model_config_file, 'r') as stream:
        model_config = yaml.safe_load(stream)

    output_layers, layers, inputs = parse_model_cfg(nclasses, **model_config)
    from tensorflow.keras import Input, Model

    model = Model(inputs, output_layers)

    convert = Convert()
    model = convert.load_darknet_weights(model, weights_file)
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = model(img)
    print('sanity check passed')
    model.save_weights(output_weights_file)
    print('weights saved')


if __name__ == "__main__":
    main()
