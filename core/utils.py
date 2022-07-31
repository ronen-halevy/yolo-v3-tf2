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
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import os
import numpy as np
from numpy import loadtxt

def resize_image(img, target_height, target_width):
    img = tf.image.resize(
        img,
        (target_height, target_width),
        preserve_aspect_ratio=True,
    )
    _, scaled_height, scaled_width, _ = img.shape

    scaled_image = tf.image.pad_to_bounding_box(
        img, (target_height - scaled_height) // 2, (target_width - scaled_width) // 2, target_height, target_width
    )
    return scaled_image


def get_anchors(anchors_file):
    number_of_scale_grids = 3
    anchors_per_scale_grid = 3
    anchor_entry_size = 2
    anchors_table = loadtxt(anchors_file, dtype=np.float, delimiter=',')
    anchors_table = anchors_table.reshape(
        number_of_scale_grids, anchors_per_scale_grid, anchor_entry_size)
    return anchors_table




