import tensorflow as tf
import numpy as np
import yaml
import argparse

from core.models import YoloV3Model

def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = [model.layers[idx].name for idx in range(len(model.layers))]
    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        if not hasattr(sub_model, 'layers'):
            continue
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            print("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)
    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def convert(nclasses, input_weights_file, output_weights_file):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolov3_model = YoloV3Model()
    model = yolov3_model(nclasses=nclasses)
    model.summary()

    load_darknet_weights(model, input_weights_file)
    print('weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = model(img)
    print('sanity check passed')
    model.save_weights(output_weights_file)
    print('weights saved')


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
convert(nclasses, weights_file, output_weights_file)
