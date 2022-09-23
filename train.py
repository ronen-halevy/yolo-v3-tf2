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
from tensorflow.keras import Input, Model
from keras.callbacks import Callback
from keras.callbacks import (
    EarlyStopping
)
import logging
import yaml
import argparse
import matplotlib.pyplot as plt

from core.utils import get_anchors, count_file_lines
from core.render_utils import render_bboxes

from core.preprocess_dataset import PreprocessDataset
from core.loss_func import get_loss_func
from core.parse_model import ParseModel

from core.load_tfrecords import parse_tfrecords
from core.create_dataset_from_coco_files import create_dataset_from_coco_files
from core.create_debug_dataset import load_debug_dataset
from core.transfer_learning import do_transfer_learning


class Train:
    @staticmethod
    def get_data_from_tfrecords(train_tfrecords, val_tfrecords, image_size, max_bboxes, classes_name_file):
        dataset = []
        tfrecords_dir_train = f'{train_tfrecords}'
        tfrecords_dir_val = f'{val_tfrecords}'
        for ds_dir in [tfrecords_dir_train, tfrecords_dir_val]:
            dset = parse_tfrecords(
                ds_dir, image_size, max_bboxes, classes_name_file)
            dataset.append(dset)
        return dataset

    @staticmethod
    def _calc_loss(model, images, labels, loss_fn_list, batch_size):
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

    def _train_eager_mode(self, model, ds_train, ds_val, loss_fn_list, optimizer, batch_size, epochs,
                          checkpoits_path_prefix, weights_save_peroid):
        for epoch in range(1, epochs + 1):
            for batch, (images, labels) in enumerate(ds_train):
                with tf.GradientTape() as tape:
                    total_loss, pred_loss, pred_loss_per_grid, pred_loss_per_source = self._calc_loss(model, images,
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
            if epoch % weights_save_peroid == 0:
                model.save_weights(
                    checkpoits_path_prefix)

            for batch, (images, labels) in enumerate(ds_val):
                total_loss, pred_loss, pred_loss_per_grid, pred_loss_per_source = self._calc_loss(model, images,
                                                                                                  labels,
                                                                                                  loss_fn_list,
                                                                                                  batch_size)

                logging.info(
                    f'{epoch}_val_{batch}_lr:{optimizer.lr.numpy():.6f}, '
                    f'totLoss:{total_loss.numpy()}, '
                    f'perGrid{list(pred_loss_per_grid.numpy())}, '
                    f'perSource[xy,wh,obj,class]:{pred_loss_per_source.numpy()}, '
                    f'perGridPerSource:{[list(x.numpy()) for idx, x in enumerate(pred_loss)]}')

    class WeightsSaver(Callback):
        def __init__(self, output_checkpoints_path, weights_save_peroid):
            super(Callback, self).__init__()
            self.weights_save_peroid = weights_save_peroid
            self.output_checkpoints_path = output_checkpoints_path
            self.epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            if self.epoch % self.weights_save_peroid == 0:
                # name = 'checkpoints/yolov3_train.tf'
                self.model.save_weights(self.output_checkpoints_path)
            self.epoch += 1

    def __call__(self,
                 model_config_file,
                 image_size,
                 grid_sizes_table,
                 batch_size,
                 max_bboxes,
                 debug_mode,
                 anchors_file,
                 learning_rate,
                 early_stop_patience,
                 epochs,
                 training_mode,
                 render_dataset_example,
                 max_dataset_examples,
                 dataset_repeats,
                 transfer_learning_config,
                 dataset_config,
                 classes_name_file,
                 output_checkpoints_path,
                 early_stopping,
                 weights_save_peroid,
                 **kwargs
                 ):

        grid_sizes_table = np.array(grid_sizes_table)

        logging.basicConfig(level=logging.INFO,
                            format='%(levelname)s %(message)s',
                            )
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        dataset = []
        if dataset_config['input_data_source'] == 'tfrecords':
            dataset = self.get_data_from_tfrecords(dataset_config['tfrecords']['train'],
                                                   dataset_config['tfrecords']['valid'],
                                                   image_size, max_bboxes,
                                                   classes_name_file)
        elif dataset_config['input_data_source'] == 'coco_format_files':
            dataset_size = []
            for ds_split_config in (dataset_config['coco_format_files']['train'],
                                    dataset_config['coco_format_files']['valid']):
                ds_split, ds_split_size = create_dataset_from_coco_files(ds_split_config['images_dir'],
                                               ds_split_config['annotations'],
                                               image_size,
                                               max_dataset_examples,
                                               max_bboxes=100
                                                            )

                dataset.append(ds_split)
                dataset_size.append(ds_split_size)
            # Modify batch size, to avoid an empty dataset after batching with drop_remainder=True:
            batch_size = min((batch_size, min(dataset_size)))
        else:  # debug_data
            train_dataset = load_debug_dataset()
            val_dataset = load_debug_dataset()
            dataset = [train_dataset, val_dataset]

        if dataset_repeats:  # repeat train and val
            dataset[0] = dataset[0].repeat(dataset_repeats)  # train
            dataset[1] = dataset[1].repeat(dataset_repeats)  # val

        anchors_table = get_anchors(anchors_file)
        nclasses = count_file_lines(classes_name_file)

        if render_dataset_example:
            x_train, bboxes = next(iter(dataset[0]))
            bboxes = bboxes[..., 0:4]
            image = render_bboxes(x_train[tf.newaxis, ...], bboxes[tf.newaxis, ...], colors=[(255, 255, 255)])
            plt.imshow(image[0])
            plt.show()

        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)
        parse_model = ParseModel()
        inputs = Input(shape=(None, None, 3))

        sub_models_configs = model_config['sub_models_configs']
        output_stage = model_config['output_stage']
        decay_factor = model_config['decay_factor']
        model = parse_model.build_model(inputs, sub_models_configs, output_stage, decay_factor, nclasses)

        with open("model_summary.txt", "w") as file1:
            model.summary(print_fn=lambda x: file1.write(x + '\n'))

        if transfer_learning_config and transfer_learning_config.get('transfer_list'):
            if 'all' in transfer_learning_config['transfer_list']:
                input_weights_path = transfer_learning_config['input_weights_path']
                model.load_weights(input_weights_path)
            elif 'none' not in transfer_learning_config['transfer_list']:
                input_weights_path = transfer_learning_config['input_weights_path']
                do_transfer_learning(model, model_config, transfer_learning_config, input_weights_path)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_fn_list = [get_loss_func(anchors, nclasses, tf.constant(training_mode == 'eager_tf', dtype=tf.bool))
                        for anchors in anchors_table]

        model.compile(optimizer=optimizer, loss=loss_fn_list,
                      run_eagerly=(training_mode == 'eager_fit'))
        preprocess_dataset = PreprocessDataset()
        if debug_mode:
            preprocess_dataset.preprocess_dataset_debug(dataset[0], batch_size, image_size, anchors_table,
                                                        grid_sizes_table,
                                                        max_bboxes)
        ds_preprocessed = []
        for ds_split in dataset:
            data = preprocess_dataset(ds_split, batch_size, image_size, anchors_table, grid_sizes_table,
                                      max_bboxes)
            ds_preprocessed.append(data)

        ds_train, ds_val = ds_preprocessed

        if training_mode == 'eager_tf':
            self._train_eager_mode(model, ds_train, ds_val, loss_fn_list, optimizer, batch_size, epochs,
                                   output_checkpoints_path, weights_save_peroid)
        else:

            callbacks = [
                # ReduceLROnPlateau(verbose=1),
                # ModelCheckpoint('checkpoints/yolov3_train.tf',
                #                 verbose=1, save_weights_only=True),
                # TensorBoard(log_dir='logs')
                self.WeightsSaver(output_checkpoints_path, weights_save_peroid)
            ]
            from keras.callbacks import Callback

            if early_stopping:
                callbacks.append(
                    EarlyStopping(patience=early_stop_patience, restore_best_weights=True, monitor='val_loss',
                                  verbose=1))

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

    train = Train()
    train(**train_config)
