#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : yolo_nms_layer.py
#   Author      : ronen halevy 
#   Created date:  8/12/22
#   Description :
#
# ================================================================
import tensorflow as tf
from core.yolo_nms import yolo_nms


class YoloNmsLayer(tf.keras.layers.Layer):

    def __init__(self, yolo_max_boxes, nms_iou_threshold, nms_score_threshold, **kwargs):
        self.yolo_max_boxes = yolo_max_boxes
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold
        super(YoloNmsLayer, self).__init__(**kwargs)



    def call(self, decoded_outputs, **kwargs):
        bboxes, class_indices, scores, selected_indices_padded, num_valid_detections = \
            yolo_nms(decoded_outputs, self.yolo_max_boxes, self.nms_iou_threshold, self.nms_score_threshold)
        return bboxes, class_indices, scores, selected_indices_padded, num_valid_detections
