import tensorflow as tf
import numpy as np
import yaml
import argparse

from core.models import YoloV3Model


class Convert:
    @staticmethod
    def __process_layer(wf, layer, layers, layer_idx):
        """
        Load weights from a file to conv2d layers. A conv2d weigt entries may either be list of [bias, weight] or,
        in case where a bn module follows, a single [weight] value
        In the latter case, instead of loading bias, the batch_normalization entry is loaded.
        :param wf:  weights input file, darknet format (=caffe)
        :type wf: Buffered Reader
        :param layer: Keras layer
        :type layer: Keras layer
        :param layers: The list of model's layers (maybe a sub-model as well). Needed to extract bn weights, if
         bn follows.
        :type layers: List of Keras layer
        :param layer_idx: Index of current layer within layers list. Needed to locate next layer - for he bn case
        :type layer_idx:
        :return:
        :rtype:
        """
        if layer.name.startswith('conv2d'):
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if len(layer.weights) == 1:  # ebtry contains a weight w/o a bias, which is the case of bn in next layer.
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]] # bn weights
                batch_norm = layers[layer_idx + 1] # targer bn entry
            else:  # a bias + weight layer
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            print(f'{layer.name} {"bn" if len(layer.weights) == 1 else "bias"}')

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])
            if len(layer.weights) == 2:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    def __recurse_model(self, wf, model):
        """
        A recursive method - Loop on model's layers, call __process_layer which loads weights to layer from file
        :param wf: weights input file, darknet format (=caffe)
        :type wf: Buffered Reader
        :param model:  A model can be either a model which contains layers and sub-models or a descrete layer.
        If a model
         call __recurse_model
        :type model: Keras model or Leras layer
        :return:
        :rtype:
        """
        for layer_idx, layer in enumerate(model.layers):
            if not hasattr(layer, 'layers'):
                self.__process_layer(wf, layer, model.layers, layer_idx)
            else:
                print(f'++ start layer of model {layer.name} ++')
                self.__recurse_model(wf, layer)
                print(f'-- done model {layer.name} --')

    def __load_darknet_weights(self, model, weights_file):
        wf = open(weights_file, 'rb')
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
        self.__recurse_model(wf, model)
        assert len(wf.read()) == 0, 'failed to read all data'
        wf.close()

    def __call__(self, nclasses, input_weights_file, output_weights_file):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        yolov3_model = YoloV3Model()
        model = yolov3_model(nclasses=nclasses)
        model.summary()

        self.__load_darknet_weights(model, input_weights_file)
        print('weights loaded')

        img = np.random.random((1, 320, 320, 3)).astype(np.float32)
        output = model(img)
        print('sanity check passed')
        model.save_weights(output_weights_file)
        print('weights saved')


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
    convert = Convert()
    convert(nclasses, weights_file, output_weights_file)

if __name__ == "__main__":
    main()