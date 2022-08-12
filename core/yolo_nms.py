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

@tf.function
def yolo_nms(outputs, yolo_max_boxes, nms_iou_threshold, nms_score_threshold):
    bbox, confidence, class_probs = outputs
    class_probs = tf.squeeze(class_probs, axis=0)

    class_indices = tf.argmax(class_probs, axis=-1)
    class_probs = tf.reduce_max(class_probs, axis=-1)
    scores = confidence * class_probs
    scores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(scores, [1])
    bbox = tf.reshape(bbox, (-1, 4))

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=yolo_max_boxes,
        iou_threshold=nms_iou_threshold,
        score_threshold=nms_score_threshold,
        soft_nms_sigma=0.
    )

    num_of_valid_detections = tf.expand_dims(tf.shape(selected_indices)[0], axis=0)
    selected_boxes = tf.gather(bbox, selected_indices)
    selected_boxes = tf.expand_dims(selected_boxes, axis=0)
    selected_scores = tf.expand_dims(selected_scores, axis=0)
    selected_classes = tf.gather(class_indices, selected_indices)
    selected_classes = tf.expand_dims(selected_classes, axis=0)

    return selected_boxes, selected_scores, selected_classes, num_of_valid_detections