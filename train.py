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
from preprocess_dataset import preprocess_dataset, arrange_in_grid
from loss_func import get_loss_func
from models import yolov3_model, yolov3_model_new, YoloV3mm


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


def get_config():
    import os
    if 'COLAB_GPU' not in os.environ:  # colab does not support argparse
        parser = argparse.ArgumentParser()

        parser.add_argument('--use_debug_dataset',
                            default=True, action='store_true')

        parser.add_argument('--render_dataset_example',
                            default=False, action='store_true')

        parser.add_argument("--tfrecords_dir", type=str,
                            default='/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords',
                            help='path to tfrecords files')
        parser.add_argument("--limit", type=int, default=None,
                            help='limit on max input examples')

        parser.add_argument("--classes_name_fle", type=str,
                            default='datasets/shapes/class.names',
                            help='path to classes names file needed to annotate plotted objects')

        parser.add_argument("--max_bboxes", type=int, default=100,
                            help='max bounding boxes in an example image')

        parser.add_argument("--batch_size", type=int, default=4,
                            help='batch_size')

        parser.add_argument("--image_size", type=int, default=416,
                            help='Algorithm"s image_size . Assumed a square')

        parser.add_argument("--anchors_file", type=str, default='datasets/shapes/shapes_yolov3_anchors.txt',
                            help='anchors_file')

        parser.add_argument("--dataset_limit_size", type=str, default=None,
                            help='Train will limit dataset to dataset_size. If None, no limit is set')

        parser.add_argument("--learning_rate", type=float, default=0.001,
                            help='learning_rate')

        parser.add_argument("--epochs", type=int, default=50,
                            help='epochs')

        parser.add_argument("--mode", type=str, default="eager_tf",
                            help="model's execution mode")

        parser.add_argument("--debug_annotations_path", type=str, default=None  # 'datasets/shapes/debug_dataset_sample/annotations.json'
                            , help="model's execution mode")

        args = parser.parse_args()
    else:
        class args:
            use_debug_dataset = True
            render_dataset_example = False
            tfrecords_dir = '/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords'
            limit = None
            classes_name_fle = 'datasets/shapes/class.names'
            max_bboxes = 100
            batch_size = 4
            image_size = 416
            anchors_file = 'datasets/shapes/shapes_yolov3_anchors.txt'
            dataset_limit_size = None
            learning_rate = 0.001
            epochs = 10
            mode = "eager_tf"
            debug_annotations_path = None # 'datasets/shapes/debug_dataset_sample/annotations.json'

    tfrecords_dir = args.tfrecords_dir
    image_size = args.image_size
    batch_size = args.batch_size
    classes_name_fle = args.classes_name_fle
    max_bboxes = args.max_bboxes
    use_debug_dataset = args.use_debug_dataset
    anchors_file = args.anchors_file
    learning_rate = args.learning_rate
    epochs = args.epochs
    mode = args.mode
    render_dataset_example = args.render_dataset_example
    debug_annotations_path = args.debug_annotations_path

    return tfrecords_dir, classes_name_fle, batch_size, image_size, anchors_file, max_bboxes, epochs, mode, learning_rate, render_dataset_example, use_debug_dataset, debug_annotations_path


def get_anchors(anchors_file):
    number_of_scale_grids = 3
    anchors_per_scale_grid = 3
    anchor_entry_size = 2
    anchors_table = loadtxt(anchors_file, dtype=np.float, delimiter=',')
    anchors_table = anchors_table.reshape(
        number_of_scale_grids, anchors_per_scale_grid, anchor_entry_size)
    anchors_table = np.flip(np.sort(anchors_table, axis=- 1))
    return anchors_table


def load_dataset(tfrecords_dir, use_debug_dataset, image_size, max_bboxes, anchors_file, classes_name_fle, debug_annotations_path=None):
    if use_debug_dataset:
        if debug_annotations_path is not None:
            dataset = load_sample_dataset(
                debug_annotations_path, classes_name_fle, max_bboxes)

        else:
            dataset = load_fake_dataset()
    else:
        dataset = parse_tfrecords(
            tfrecords_dir, image_size, max_bboxes, classes_name_fle)

    if not use_debug_dataset or debug_annotations_path:
        dataset_names = loadtxt(classes_name_fle, dtype=str)
        nclasses = dataset_names.shape[0]
        anchors_table = get_anchors(anchors_file)
    else:
        anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
                                 (59, 119)], [(116, 90), (156, 198), (373, 326)]],
                                 np.float32) / 416
        # anchors_table = np.flip(np.sort(anchors_table, axis=- 1))
        nclasses = 80

    dataset = dataset.repeat(100)

    return dataset, nclasses, anchors_table


def main():
    # tf.random.set_seed(seed=42)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s',
                        )

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    tfrecords_dir, classes_name_fle, batch_size, image_size, anchors_file, max_bboxes, epochs, mode, learning_rate, render_dataset_example, use_debug_dataset, debug_annotations_path = get_config()

    lr_init = 1e-3
    lr_end = 1e-6
    steps_per_epoch = int(10000 / batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_epochs = 2

    dataset, nclasses, anchors_table = load_dataset(
        tfrecords_dir, use_debug_dataset, image_size, max_bboxes, classes_name_fle, debug_annotations_path)
    if render_dataset_example:
        x_train, bboxes = next(iter(dataset))
        render_bboxes(x_train, bboxes)

    # model = yolov3_model(anchors_table, image_size, nclasses=nclasses)
    # model = yolov3_model_new(anchors_table, image_size, nclasses=nclasses)

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    model = YoloV3mm(size=None, channels=3, anchors=yolo_anchors,
                 masks=yolo_anchor_masks, classes=80, training=False)

    print(model.summary())
    with open("model_mine.txt", "w") as file1:
        model.summary(print_fn=lambda x: file1.write(x + '\n'))

    optimizer = tf.keras.optimizers.Adam()

    loss = [get_loss_func(anchors, nclasses=nclasses)
            for anchors in anchors_table]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(mode == 'eager_fit'))

    grid_sizes_table = np.array([13, 26, 52])
    # ###
    # image, y = next(iter(dataset.as_numpy_iterator()))
    # y = tf.expand_dims(y, axis=0)
    # for grid_index in [0,1,2]:
    #     arrange_in_grid(y, tf.convert_to_tensor(anchors_table),  grid_index,# ronen TODO was 3,6 check shape
    #                     # +1 is a patch - todo add the obj in dataset already...
    #                     [batch_size, grid_sizes_table[grid_index], grid_sizes_table[grid_index],
    #                      anchors_table.shape[grid_index], tf.shape(y)[-1]], max_bboxes
    #                     )

    ####

    train_dataset = preprocess_dataset(dataset, batch_size, image_size, anchors_table, grid_sizes_table,
                                       max_bboxes)
    if mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        for epoch in range(1, epochs+1):
            for batch_count, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    pred_loss_stack_all = tf.stack(pred_loss, axis=1)
                    pred_loss_per_grid = tf.reduce_sum(
                        pred_loss_stack_all, axis=0)
                    pred_loss_per_source = tf.reduce_sum(
                        pred_loss_stack_all, axis=1)

                    total_loss = tf.reduce_sum(
                        pred_loss_per_source) + regularization_loss

                    # total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                warmup_steps = warmup_epochs * steps_per_epoch
                if global_steps < warmup_steps:
                    lr = global_steps / warmup_steps * lr_init
                else:
                    lr = lr_end + 0.5 * (lr_init - lr_end) * (
                        (1 + tf.cos((global_steps - warmup_steps) /
                         (total_steps - warmup_steps) * np.pi))
                    )
                lr = tf.Variable(.001)
                optimizer.lr.assign(lr.numpy())

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info(
                    f'{epoch}_train_{batch_count}_lr:{optimizer.lr.numpy():.6f}, totLoss:{total_loss.numpy()}, perGrid{list(pred_loss_per_grid.numpy())}, perSource[xy,wh,obj,class]:{pred_loss_per_source.numpy()}, perGridPerSeource:{[list(x.numpy()) for idx, x in enumerate(pred_loss)]}')
                # logging.info(f'Detailed Loss Grid-n[xy,wh,obj,class]: {[list(x.numpy()) for idx, x in enumerate(pred_loss)]}')

                avg_loss.update_state(total_loss)

                global_steps.assign_add(1)
            # if(epoch and epoch % 10 == 0):
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))

if __name__ == '__main__':
    # train()
    main()
