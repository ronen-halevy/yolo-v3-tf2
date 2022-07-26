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
from keras.callbacks import Callback
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from numpy import loadtxt
import logging
import yaml
import argparse
import matplotlib.pyplot as plt


from utils import get_anchors
from render_utils import render_bboxes

from preprocess_dataset import preprocess_dataset, preprocess_dataset_debug
from loss_func import get_loss_func
from models import yolov3_model

from load_tfrecords import parse_tfrecords
from load_dataset import load_dataset, load_debug_dataset



def loss_calc(model, images, labels, loss_fn_list, batch_size):
    outputs = model(images, training=True)
    regularization_loss = tf.reduce_sum(model.losses)
    pred_loss = []
    for output, label, loss_fn in zip(outputs, labels, loss_fn_list):
        pred_loss.append(loss_fn(label, output) / batch_size)
    pred_loss_stack_all = tf.stack(pred_loss, axis=1)
    pred_loss_per_grid = tf.reduce_sum(
        pred_loss_stack_all, axis=0)
    pred_loss_per_source = tf.reduce_sum(
        pred_loss_stack_all, axis=1)

    total_loss = tf.reduce_sum(
        pred_loss_per_source) + regularization_loss

    return total_loss, pred_loss, pred_loss_per_grid, pred_loss_per_source


def train_eager_mode(model, ds_train, ds_val, loss_fn_list, optimizer, batch_size, epochs, checkpoits_path_prefix):
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    for epoch in range(1, epochs + 1):
        for batch, (images, labels) in enumerate(ds_train):
            with tf.GradientTape() as tape:
                total_loss, pred_loss, pred_loss_per_grid, pred_loss_per_source = loss_calc(model, images,
                                                                                            labels,
                                                                                            loss_fn_list,
                                                                                            batch_size)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables))

            logging.info(
                f'{epoch}_train_{batch}_lr:{optimizer.lr.numpy():.6f}, '
                f'totLoss:{total_loss.numpy()}, '
                f'perGrid{list(pred_loss_per_grid.numpy())}, '
                f'perSource[xy,wh,obj,class]:{pred_loss_per_source.numpy()}, '
                f'perGridPerSource:{[list(x.numpy()) for idx, x in enumerate(pred_loss)]}')

            global_steps.assign_add(1)
        model.save_weights(
            f'{checkpoits_path_prefix}{epoch}.tf')

        for batch, (images, labels) in enumerate(ds_val):
            total_loss, pred_loss, pred_loss_per_grid, pred_loss_per_source = loss_calc(model, images,
                                                                                        labels,
                                                                                        loss_fn_list,
                                                                                        batch_size)

            logging.info(
                f'{epoch}_val_{batch}_lr:{optimizer.lr.numpy():.6f}, '
                f'totLoss:{total_loss.numpy()}, '
                f'perGrid{list(pred_loss_per_grid.numpy())}, '
                f'perSource[xy,wh,obj,class]:{pred_loss_per_source.numpy()}, '
                f'perGridPerSource:{[list(x.numpy()) for idx, x in enumerate(pred_loss)]}')


def count_file_lines(filename):
    with open(filename, 'r') as fp:
        nlines = len(fp.readlines())
    return nlines

class WeightsSaver(Callback):
  def __init__(self, n):
    self.n = n
    self.epoch = 0

  def on_epoch_end(self, epoch, logs={}):
    if self.epoch % self.n == 0:
        name = 'checkpoints/yolov3_train.tf'
        self.model.save_weights(name)
        self.epoch += 1

def train(input_data_source,
          image_size,
          batch_size,
          max_bboxes,
          debug_mode,
          anchors_file,
          learning_rate,
          early_stop_patience,
          epochs,
          mode,
          render_dataset_example,
          dataset_cuttof_size,
          dataset_repeats,
          load_weights,
          annotations_path,
          images_dir,
          tfrecords_base_dir,
          classes_name_file,
          save_checkpoits_path_prefix,
          load_checkpoints_path,
          early_stopping
          ):
    grid_sizes_table = np.array([13, 26, 52])

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s',
                        )
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    dataset = []
    if input_data_source == 'tfrecords':
        tfrecords_dir_train = f'{tfrecords_base_dir}/train'
        tfrecords_dir_val = f'{tfrecords_base_dir}/val'
        for ds_dir in [tfrecords_dir_train, tfrecords_dir_val]:
            dset = parse_tfrecords(
                ds_dir, image_size, max_bboxes, classes_name_file)
            dataset.append(dset)
    elif input_data_source == 'images_dir_annot_file':
        train_dataset, val_dataset, test_dataset, train_size, val_size = load_dataset(images_dir, annotations_path,
                                                                                      classes_name_file, image_size,
                                                                                      dataset_cuttof_size=dataset_cuttof_size,
                                                                                      max_bboxes=100, train_split=0.7,
                                                                                      val_split=0.2)

        # Modify batch size, to avoid an empty dataset after batching with drop_remainder=True:
        batch_size = batch_size if (batch_size <= min(train_size, val_size)) else min(train_size, val_size)

    else: #debug_data
        train_dataset = load_debug_dataset()
        val_dataset = load_debug_dataset()
    if not len(dataset):
        dataset = [train_dataset, val_dataset]

    if dataset_repeats:  # repeat train and val
        dataset[0] = dataset[0].repeat(dataset_repeats)
        dataset[1] = dataset[1].repeat(dataset_repeats)

    anchors_table = get_anchors(anchors_file)
    nclasses = count_file_lines(classes_name_file)

    if render_dataset_example:
        x_train, bboxes = next(iter(dataset[0]))
        image = render_bboxes(x_train[tf.newaxis, ...], bboxes[tf.newaxis, ...])
        plt.imshow(image)
        plt.show()

    model = yolov3_model(image_size, nclasses=nclasses)

    with open("model_summary.txt", "w") as file1:
        model.summary(print_fn=lambda x: file1.write(x + '\n'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_fn_list = [get_loss_func(anchors, nclasses, tf.constant(mode == 'eager_tf', dtype=tf.bool))
                    for anchors in anchors_table]

    model.compile(optimizer=optimizer, loss=loss_fn_list,
                  run_eagerly=(mode == 'eager_fit'))

    if debug_mode:
        preprocess_dataset_debug(dataset[0], batch_size, image_size, anchors_table, grid_sizes_table,
                                 max_bboxes)
    ds_preprocessed = []
    for ds_split in dataset:
        data = preprocess_dataset(ds_split, batch_size, image_size, anchors_table, grid_sizes_table,
                                  max_bboxes)
        ds_preprocessed.append(data)

    ds_train, ds_val = ds_preprocessed

    if load_weights:
        model.load_weights(load_checkpoints_path)

    if mode == 'eager_tf':
        train_eager_mode(model, ds_train, ds_val, loss_fn_list, optimizer, batch_size, epochs,
                         save_checkpoits_path_prefix)
    else:

        callbacks = [
            # ReduceLROnPlateau(verbose=1),
            # ModelCheckpoint('checkpoints/yolov3_train.tf',
            #                 verbose=1, save_weights_only=True),
            # TensorBoard(log_dir='logs')
            WeightsSaver(2)
        ]
        from keras.callbacks import Callback

        if early_stopping:
            callbacks.append(
                EarlyStopping(patience=early_stop_patience, restore_best_weights=True, monitor='val_loss', verbose=1))

        history = model.fit(ds_train,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=ds_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config/train_config.yaml',
                        help='yaml config file')

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as stream:
        train_config = yaml.safe_load(stream)
    train(**train_config)
