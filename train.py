#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : scratch.py
#   Author      : ronen halevy
#   Created date:  5/2/22
#   Description :
#
# ================================================================
import numpy as np
import tensorflow as tf
import argparse

from parse_tfrecords import parse_tfrecords

from numpy import loadtxt
import logging


from utils import render_bboxes
from utils import generate_random_dataset, load_fake_dataset, load_sample_dataset
from preprocess_dataset import preprocess_dataset
from loss_func import get_loss_func


def split_dataset(dataset, dataset_size):
    # the "or" condition is for debug small datasets
    train_size = int(0.7 * dataset_size) or dataset_size
    val_size = int(0.20 * dataset_size) or dataset_size
    test_size = int(0.10 * dataset_size) or dataset_size

    # dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=None)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset


def insert_objectiveness_entry(lablels):
    (boxes, classes) = tf.split(lablels, [4, 1], axis=-1)
    objectiveness_entry = tf.cast(tf.fill(tf.shape(classes), 1), dtype=tf.float32)
    lablels = tf.concat([boxes, objectiveness_entry, classes], axis=-1)
    return lablels


def config_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_debug_dataset', default=True, action='store_true')

    parser.add_argument('--render_dataset_example', default=True, action='store_true')

    parser.add_argument("--tfrecords_dir", type=str,
                        default='/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')

    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')

    parser.add_argument("--max_bboxes", type=int, default=100,
                        help='max bounding boxes in an example image')

    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch_size')

    parser.add_argument("--image_size", type=int, default=416,
                        help='Algorithm"s image_size . Assumed a square')

    parser.add_argument("--anchors_file", type=str, default='datasets/shapes/shapes_yolov3_anchors.txt',
                        help='anchors_file')

    parser.add_argument("--label_names_file", type=str, default='datasets/shapes/class.names',
                        help='label_names_file')

    parser.add_argument("--dataset_limit_size", type=str, default=None,
                        help='Train will limit dataset to dataset_size. If None, no limit is set')

    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help='learning_rate')

    parser.add_argument("--epochs", type=int, default=50,
                        help='epochs')

    parser.add_argument("--mode", type=str, default="eager_tf",
                        help="model's execution mode")

    grid_sizes_table = np.array([13, 26, 52])

    args = parser.parse_args()
    tfrecords_dir = args.tfrecords_dir
    image_size = args.image_size
    batch_size = args.batch_size
    class_file = args.classes
    max_bboxes = args.max_bboxes
    use_debug_dataset = args.use_debug_dataset
    dataset_limit_size = args.dataset_limit_size
    anchors_file = args.anchors_file
    label_names_file = args.label_names_file
    learning_rate = args.learning_rate
    epochs = args.epochs
    mode = args.mode
    render_dataset_example = args.render_dataset_example

    debug_annotations_path = 'datasets/shapes/debug_dataset_sample/annotations.json'
    class_names_path = 'datasets/shapes/class.names'

    if use_debug_dataset == 'BB':
        dataset = load_fake_dataset(render_dataset_example)
        dataset = dataset.repeat()

        anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
                                                                   (59, 119)], [(116, 90), (156, 198), (373, 326)]],
                                 np.float32) / image_size
        nclasses = 80  # ronen TODO

    elif use_debug_dataset:

        dataset = load_sample_dataset(debug_annotations_path, class_names_path, max_bboxes, render_dataset_example)
        dataset = dataset.repeat()
        number_of_scale_grids = 3
        anchors_per_scale_grid = 3
        anchor_entry_size = 2
        anchors_table = loadtxt(anchors_file, dtype=np.float, delimiter=',')
        anchors_table = anchors_table.reshape(number_of_scale_grids, anchors_per_scale_grid, anchor_entry_size)
        dataset_names = loadtxt(class_names_path, dtype=str)
        nclasses = dataset_names.shape[0]
    else:
        dataset = parse_tfrecords(tfrecords_dir, image_size, max_bboxes, class_file)
        dataset = dataset.map(lambda x, y: (x, insert_objectiveness_entry(y)))

        anchors_table = loadtxt(anchors_file)

        number_of_scale_grids = 3
        anchors_per_scale_grid = 3
        anchor_entry_size = 2
        anchors_table = anchors_table.reshape(number_of_scale_grids, anchors_per_scale_grid, anchor_entry_size)
        dataset_names = loadtxt(label_names_file, dtype=str)
        nclasses = dataset_names.shape[0]

    dataset_size = dataset_limit_size if dataset_limit_size else dataset.cardinality().numpy()


    # Descending anchors_table Order:
    anchors_table = np.flip(np.sort(anchors_table, axis=- 1))
    return dataset, dataset_size, batch_size, image_size, anchors_table, max_bboxes, grid_sizes_table, nclasses, epochs, mode, learning_rate


def main():
    tf.random.set_seed(seed=42)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s',
                        )

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    train_dataset, dataset_size, batch_size, image_size, anchors_table, max_bboxes, grid_sizes_table, nclasses, epochs, mode, learning_rate = config_train()

    from models import yolov3_model

    model = yolov3_model(anchors_table, image_size, nclasses=nclasses)

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    loss = [get_loss_func(anchors, nclasses=nclasses)
            for anchors in anchors_table]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(mode == 'eager_fit'))

    train_dataset = preprocess_dataset(train_dataset, batch_size, image_size, anchors_table, grid_sizes_table,
                                       max_bboxes)

    # train_dataset = train_dataset.shuffle(buffer_size=512)
    # train_dataset = train_dataset.repeat()
    # train_dataset = train_dataset.batch(batch_size)

    if mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, epochs + 1 + 1000):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)


if __name__ == '__main__':
    # train()
    main()
