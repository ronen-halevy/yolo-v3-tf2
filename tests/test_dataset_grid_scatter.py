#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : dataset_transformation_debug.py
#   Author      : ronen halevy 
#   Created date:  5/9/22
#   Description : Creates dataset. Decodes bboxes from grid and compare with unprocessed bboxes. Prints Pass/Fail
#
# ================================================================
import tensorflow as tf
import train
from core import preprocess_dataset
import numpy as np

def get_preprocessed_dataset_entry(unprocessed_dataset, batch_size, image_size, anchors_table, grid_sizes_table, max_bboxes):
    # Call operatoinal preprocess_dataset method, which arranges boxes in grid
    dataset = preprocess_dataset.preprocess_dataset(unprocessed_dataset, batch_size, image_size, anchors_table,
                                                    grid_sizes_table,
                                                    max_bboxes)
    image, all_grids_scattered_bboxes = next(iter(dataset.as_numpy_iterator()))
    return image, all_grids_scattered_bboxes


def test_dataset_in_scattered_grids():
    tfrecords_dir, classes_name_fle, batch_size, image_size, anchors_file, max_bboxes, epochs, mode, learning_rate, render_dataset_example, use_debug_dataset, debug_annotations_path = train.get_config()

    unprocessed_dataset, nclasses, anchors_table = train.load_dataset(tfrecords_dir, use_debug_dataset, image_size,
                                                                      max_bboxes,
                                                                      classes_name_fle, debug_annotations_path)
    unprocessed_dataset = unprocessed_dataset.repeat(1)

    grid_sizes_table = np.array([13, 26, 52])

    # keep dataset boxes before scatter to grid, as reference
    orig_image, bboxes_orig = next(iter(unprocessed_dataset.as_numpy_iterator()))
    bboxes_orig_mask = bboxes_orig[:, 2] != 0
    bboxes_orig = tf.boolean_mask(bboxes_orig, bboxes_orig_mask)
    #Call operatoinal preprocess_dataset method, which arranges boxes in grid
    dataset = preprocess_dataset.preprocess_dataset(unprocessed_dataset, batch_size, image_size, anchors_table,
                                                    grid_sizes_table,
                                                    max_bboxes)
    image, all_grids_scattered_bboxes = get_preprocessed_dataset_entry(unprocessed_dataset, batch_size, image_size,
                                                                      anchors_table, grid_sizes_table,
                                                                      max_bboxes)


    # extract boxes from 3 grids, and compare to ref:
    # image, all_grids_scattered_bboxes = next(iter(dataset.as_numpy_iterator()))
    for grid_index, grid_scattered_bboxes in enumerate(all_grids_scattered_bboxes):
        grid_scattered_bbox = tf.convert_to_tensor(grid_scattered_bboxes[grid_index])

        mask = grid_scattered_bbox[..., 2] != 0

        bboxes_extracted = tf.boolean_mask(grid_scattered_bbox, mask)

        # Use sort to arrange result and ref tensors in same order - for comparison
        args = tf.argsort(bboxes_extracted[...,0])
        bboxes_extracted_sorted = tf.gather(bboxes_extracted, args)

        args = tf.argsort(bboxes_orig[...,0])
        bboxes_orig_sorted = tf.gather(bboxes_orig, args)

        is_boxes_equal = tf.math.equal(bboxes_orig_sorted, bboxes_extracted_sorted)
        is_identical = tf.reduce_all(is_boxes_equal)

        if is_identical:
            print(f'Grid {grid_index} Test Data PASSED')
        else:
            print(f'Grid {grid_index} Test Data Failed')

test_dataset_in_scattered_grids()


def calc_best_anchor_indices(box, grid_anchors):
    best_anchor_indices_all_grid = []
    best_anchor_indices = preprocess_dataset.calc_best_anchor_per_box(box, grid_anchors)
    return best_anchor_indices


def test_dataset_best_anchor_selection():
    tfrecords_dir, classes_name_fle, batch_size, image_size, anchors_file, max_bboxes, epochs, mode, learning_rate, render_dataset_example, use_debug_dataset, debug_annotations_path = train.get_config()

    unprocessed_dataset, nclasses, anchors_table = train.load_dataset(tfrecords_dir, use_debug_dataset, image_size,
                                                                      max_bboxes,
                                                                      classes_name_fle, debug_annotations_path)
    unprocessed_dataset = unprocessed_dataset.repeat(1)

    grid_sizes_table = np.array([13, 26, 52])

    # keep dataset boxes before scatter to grid, as reference
    orig_image, bboxes_orig = next(iter(unprocessed_dataset.as_numpy_iterator()))

    bboxes_orig_mask = bboxes_orig[:, 2] != 0
    bboxes_orig = tf.boolean_mask(bboxes_orig, bboxes_orig_mask)[tf.newaxis, ...]

    image, all_grids_scattered_bboxes = get_preprocessed_dataset_entry(unprocessed_dataset, batch_size, image_size,
                                                                      anchors_table, grid_sizes_table,
                                                                      max_bboxes)
    for grid_index, anchors_per_grid in enumerate(anchors_table):
        best_anchor_indices = calc_best_anchor_indices(bboxes_orig, anchors_per_grid)
        best_anchor_indices = tf.squeeze(best_anchor_indices)
        scattered_bboxes = all_grids_scattered_bboxes[grid_index]
        scattered_bboxes_best_anchors = tf.gather(scattered_bboxes, indices=[0], axis=-2)
        grid_bboxes_orig = tf.cast(bboxes_orig * 13, tf.int32)

        pass


test_dataset_best_anchor_selection()
