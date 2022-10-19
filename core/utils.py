#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : utils.py
#   Author      : ronen halevy 
#   Created date:  5/3/22
#   Description :
#
# ================================================================
import tensorflow as tf
import os
import numpy as np
from numpy import loadtxt

def resize_image(img, target_height, target_width):
    img = tf.image.resize(
        img,
        (target_height, target_width),
        preserve_aspect_ratio=True,
    )
    scaled_height, scaled_width = img.shape[-3:-1]

    scaled_image = tf.image.pad_to_bounding_box(
        img, (target_height - scaled_height) // 2, (target_width - scaled_width) // 2, target_height, target_width
    )
    return scaled_image


def get_anchors(anchors_file):
    nanchors_per_scale = 3
    anchor_entry_size = 2
    anchors_table = loadtxt(anchors_file, dtype=np.float, delimiter=',')
    anchors_table = anchors_table.reshape(
        -1, nanchors_per_scale, anchor_entry_size)
    return anchors_table


def count_file_lines(filename):
    with open(filename, 'r') as fp:
        nlines = len(fp.readlines())
    return nlines


def dir_filelist(images_dir, ext_list='.*'):
    filenames = []
    for f in os.listdir(images_dir):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in ext_list:
            continue
        filenames.append(f'{images_dir}/{f}')
    return filenames