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
        (selected_boxes, selected_scores, selected_classes, num_of_valid_detections) = \
            yolo_nms(decoded_outputs, self.yolo_max_boxes, self.nms_iou_threshold, self.nms_score_threshold)
        return (selected_boxes, selected_scores, selected_classes, num_of_valid_detections)




    # def compute_output_shape(self, input_shape):
    #     batch_size = input_shape[0]
    #     num_anchors = len(self.anchors)
    #     stride = self.input_dims[0] // input_shape[1]
    #     grid_size = self.input_dims[0] // stride
    #     num_bboxes = num_anchors * grid_size * grid_size
    #
    #     shape = (batch_size, num_bboxes, self.num_classes + 5)
    #
    #     return tf.TensorShape(shape)




# # from core.yolo_decode_layer import YoloDecoderLayer
# def build_yolo_decoder_layer(x, layer_config, layers, outputs, ptr, config):
#     nclasses = config['nclasses']
#     yolo_max_boxes = config['yolo_max_boxes']
#     anchors_table = config['anchors_table']
#     nms_iou_threshold = config['nms_iou_threshold']
#     nms_score_threshold = config['nms_score_threshold']
#
#     x = YoloDecoderLayer(nclasses)(x)
#     layers.append(None)
#     outputs.append(x)
#     return x, layers, outputs, ptr
#
# if __name__ == '__main__':
#     build_yolo_decoder_layer