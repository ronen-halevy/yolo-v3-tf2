#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : train.py
#   Author      : ronen halevy 
#   Created date:  4/27/22
#   Description :
#
# ================================================================
import tensorflow as tf
import numpy as np
# from matplotlib import pyplot as plt
import argparse
from read_dataset import read_dataset


# from ...create-tfrecords.render_dataset import render_dataset_examples

def render(dataset):
    data = dataset.take(1)
    image, y = next(iter(data))
    images = tf.expand_dims(image, axis=0)

    image_size = 416
    boxes = y[..., 0:4].numpy().astype(float) / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    indices = [1, 0, 3, 2]
    boxes = boxes[..., indices]
    boxes = boxes[0:10]
    boxes = tf.expand_dims(boxes, axis=0)
    boxes = tf.cast(boxes, float)
    # image_pil = Image.fromarray(np.uint8(image.numpy() * 255))
    # annotated_bbox_image = draw_bounding_box(image_pil, y[..., 0:4], color=(255, 255, 0),
    #
    #                                        thickness=1)
    colors = [[1, 1, 1]]

    # bb = np.take_along_axis(boxes, range(9), axis=1)

    images = tf.image.draw_bounding_boxes(
        images, boxes, colors, name=None
    )

    from matplotlib import pyplot as plt

    plt.imshow(images.numpy()[0])
    plt.grid()
    plt.show()


def image_preporcessn(image, y, t_w, t_h):
    s_h, s_w, _ = image.shape

    scale = min(t_w / s_w, t_h / s_h)

    n_w, n_h = int(scale * s_w), int(scale * s_h)
    boxes = (y[..., 0:4] * scale) + [(t_w - n_w) // 2, (t_h - n_h) // 2, (t_w - n_w) // 2, (t_h - n_h) // 2]
    classes = y[..., 4]

    classes = classes[tf.newaxis, :, tf.newaxis]

    boxes = tf.expand_dims(boxes, axis=0)

    y_n = tf.concat([boxes, classes], axis=-1)

    scaled_image = tf.image.pad_to_bounding_box(
        image, (t_h - n_h) // 2, (t_w - n_w) // 2, t_h, t_w
    )
    y_n = tf.squeeze(y_n, axis=0)
    return scaled_image, y_n


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecords_dir", type=str,
                        default='/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')
    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')
    parser.add_argument("--max_boxes", type=int, default=100,
                        help='max bounding boxes in an example image')

    parser.add_argument("--t_w", type=int, default=416,
                        help='target image width')

    parser.add_argument("--t_h", type=int, default=516,
                        help='target image height')

    args = parser.parse_args()

    tfrecords_dir = args.tfrecords_dir
    t_w = args.t_w
    t_h = args.t_h

    class_file = args.classes
    max_boxes = args.max_boxes
    dataset = read_dataset(class_file, tfrecords_dir, max_boxes)
    # tf.config.run_functions_eagerly(True)

    dataset = dataset.map(lambda x, y: (image_preporcessn(x, y, t_w, t_h)))

    render(dataset)


if __name__ == '__main__':
    main()
