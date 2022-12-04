#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : decode_layer_custom.py
#   Author      : ronen halevy 
#   Created date:  11/3/22
#   Description :
#
# ================================================================
import tensorflow as tf


class YoloDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, nclasses, anchors_table, **kwargs):
        self.nclasses =nclasses
        # self.yolo_max_boxes = yolo_max_boxes
        self.anchors_table = anchors_table
        # self.nms_iou_threshold = nms_iou_threshold
        # self.nms_score_threshold = nms_score_threshold
        super(YoloDecoderLayer, self).__init__(**kwargs)


    @staticmethod
    def arrange_bbox(xy, wh):
        grid_size = tf.shape(xy)[1:3]

        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2
        bbox = tf.concat([xy_min, xy_max], axis=-1)
        return bbox

    def call(self, grids_outputs, **kwargs):
        # pred_xy, pred_wh, pred_obj, class_probs = zip(
        #     *[tf.split(grid_out, (2, 2, 1, self.nclasses), axis=-1) for grid_out in grids_outputs])
        # bboxes_grid0 = self.arrange_bbox(pred_xy[0], tf.exp(pred_wh[0]) * self.anchors_table[0])
        # bboxes_grid1 = self.arrange_bbox(pred_xy[1], tf.exp(pred_wh[1]) * self.anchors_table[1])
        # # bboxes_grid2 = self.arrange_bbox(pred_xy[2], tf.exp(pred_wh[2]) * self.anchors_table[2])
        #
        # all_grids_bboxes = tf.concat(
        #     [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in [bboxes_grid0, bboxes_grid1]],
        #     axis=1)
        #
        # all_grids_confidence = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in pred_obj],
        #                                  axis=1)
        #
        # all_grids_class_probs = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in class_probs],
        #                                   axis=1)

        # (selected_boxes, selected_scores, selected_classes, num_of_valid_detections) = \
        #     self.yolo_nms((all_grids_bboxes, all_grids_confidence, all_grids_class_probs), self.yolo_max_boxes,
        #              self.nms_iou_threshold,
        #              self.nms_score_threshold)
        # return (all_grids_bboxes, all_grids_confidence, all_grids_class_probs)
        return grids_outputs
        return (selected_boxes, selected_scores, selected_classes, num_of_valid_detections)


    #####
    def __arrange_bbox(self, xy, wh):
        grid_size = tf.shape(xy)[1:3]
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2
        bbox = tf.concat([xy_min, xy_max], axis=-1)
        return bbox


    def yolo_decode(self, model_output_grids, anchors_table, nclasses):
        pred_xy, pred_wh, pred_obj, class_probs = zip(
            *[tf.split(output_grid, (2, 2, 1, nclasses), axis=-1) for output_grid in model_output_grids])

        pred_xy = [tf.keras.activations.sigmoid(pred) for pred in pred_xy]
        pred_obj = [tf.keras.activations.sigmoid(pred) for pred in pred_obj]
        class_probs = [tf.keras.activations.sigmoid(probs) for probs in class_probs]

        bboxes_in_grids = [self.__arrange_bbox(xy, tf.exp(wh) * anchors) for xy, wh, anchors in
                           zip(pred_xy, pred_wh, anchors_table)]

        all_grids_bboxes = tf.concat(
            [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in bboxes_in_grids],
            axis=1)

        all_grids_confidence = tf.concat(
            [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in pred_obj],
            axis=1)

        all_grids_class_probs = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in class_probs],
                                          axis=1)
        return all_grids_bboxes, all_grids_confidence, all_grids_class_probs
