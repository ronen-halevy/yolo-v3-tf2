#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : yolo_nms.py
#   Author      : ronen halevy 
#   Created date:  8/12/22
#   Description :
#
# ================================================================
import tensorflow as tf


# @tf.function
def yolo_nms(outputs, yolo_max_boxes, nms_iou_threshold, nms_score_threshold):
    bboxes, confidence, class_probs = outputs
    class_indices = tf.argmax(class_probs, axis=-1)
    # select class from class probs array
    class_probs = tf.reduce_max(class_probs, axis=-1)
    class_probs1 = tf.expand_dims(class_probs, axis=-1)

    scores = confidence * class_probs1
    scores = tf.squeeze(scores, axis=-1)
    bboxes = tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 4))
    selected_indices_padded, num_valid_detections = tf.image.non_max_suppression_padded(
        boxes=bboxes,
        scores=scores,
        max_output_size=yolo_max_boxes,
        iou_threshold=nms_iou_threshold,
        score_threshold=nms_score_threshold,
        pad_to_max_output_size=True
    )
    return bboxes, class_indices, scores, selected_indices_padded, num_valid_detections
