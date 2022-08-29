
#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : evaluate.py
#   Author      : ronen halevy
#   Created date:  8/21/22
#   Description :
#
# ================================================================
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Input, Model

import yaml
import argparse
import matplotlib.pyplot as plt

from core.load_tfrecords import parse_tfrecords
from core.parse_model import ParseModel

from core.yolo_decode_layer import YoloDecoderLayer
from core.yolo_nms_layer import YoloNmsLayer
from core.utils import get_anchors, resize_image


class Evaluate:
    def __init__(self, nclasses, iou_thresh):
        self.nclasses = nclasses
        self.iou_thresh = iou_thresh

    @staticmethod
    @tf.function
    def broadcast_iou(box_1, box_2):

        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                           tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                           tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])

        return int_area / (box_1_area + box_2_area - int_area)


    def calc_tp_fp_fn(self, pred_y, true_y, nclasses):
        t_bboxes, t_confidences, t_classes_indices = tf.split(true_y, [4,1,1], axis=-1)
        p_bboxes, p_classes_indices = tf.split(tf.squeeze(pred_y, axis=0), [4,1], axis=-1)

        tp = [0] * nclasses
        fp = [0] * nclasses
        fn = [0] * nclasses

        t_assigned_to_pred = [False] * tf.shape(t_confidences).numpy()[0]

        for p_bbox, p_class_index  in zip(p_bboxes, p_classes_indices):
            if tf.shape(t_bboxes)[0] != 0:
                iou = self.broadcast_iou(p_bbox, t_bboxes)
            else:
                fp += 1
                continue
            best_iou_index = tf.math.argmax(
                    iou, axis=-1, output_type=tf.int32).numpy()[0]
            true_class =  int(t_classes_indices[best_iou_index].numpy()[0])

            if iou[...,best_iou_index] > self.iou_thresh and true_class == p_class_index and not t_assigned_to_pred[best_iou_index]:
                t_assigned_to_pred[best_iou_index]=True

                tp[true_class]+=1
            else:
                pass
                fp[true_class]+=1
        for status in t_assigned_to_pred:
            if not status:
                fn[true_class]+=1

        return tp, fp, fn


    @staticmethod
    def inference_func(true_image, model):
        img = tf.expand_dims(true_image, 0)
        img = resize_image(img, image_size, image_size)
        output = model(img)
        return output


    def create_model(self, model_config_file, nclasses, nms_score_threshold, nms_iou_threshold):
        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)
        parse_model = ParseModel()
        inputs = Input(shape=(None, None, 3))
        model = parse_model.build_model(inputs, nclasses=nclasses, **model_config)

        model.load_weights(input_weights_path)
        print('weights loaded')
        model = model(inputs)

        decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
        nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                                  nms_score_threshold)(decoded_output)

        model = Model(inputs, nms_output, name="yolo_nms")
        return model

    def calc_ap(self, gt_classes, tfrecords_dir, evaluate_iou_threshold, image_size, classes_name_file, model_config_file, input_weights_path, anchors_table,
                nms_iou_threshold, nms_score_threshold, yolo_max_boxes=100):

        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)
        nclasses = 7

        dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes, class_file=classes_name_file)
        # mask dataset's padded boxes:
        dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))
        model = self.create_model(self, model_config_file, nclasses, nms_score_threshold, nms_iou_threshold)

        dataset = dataset.map(lambda x, y: (self.inference_func(x, model), y))


        data = dataset.map(lambda x, y: (tf.concat([x[0], tf.expand_dims(tf.cast(x[2], dtype=tf.float32), axis=-1)], axis=-1), y))
        tp = fp = fn = [0] * nclasses
        for idx, (pred_y, true_y) in enumerate(data):
            tp_update, fp_update, fn_update = self.calc_tp_fp_fn(pred_y, true_y, nclasses)
            tp = [tp_ + tp_update_ for tp_, tp_update_ in zip(tp, tp_update)]
            fp = [fp_ + fp_update_ for fp_, fp_update_ in zip(fp, fp_update)]
            fn = [fn_ + fn_update_ for fn_, fn_update_ in zip(fn, fn_update)]
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)

        hist =  data.map(lambda x, y:  tf.histogram_fixed_width(y[:,5], value_range=[0, nclasses], nbins=nclasses))
        objects_count = hist.reduce(np.int32(0), lambda x, y: tf.cast(x, tf.int32)+y).numpy()
        tn = objects_count - tp -fp - fn

        accuracy = (tp+tn)/(objects_count + 1e-10)
        recal = tp / (tp + fn + 1e-20)
        precision = tp / (tp + fp+1e-20)
        print(accuracy, recal, precision)



if __name__ == '__main__':
    model_conf_file = 'config/models/yolov3/model.yaml'
    parser = argparse.ArgumentParser()


    with open('config/detect_config.yaml', 'r') as stream:
        detect_config = yaml.safe_load(stream)

    tfrecords_dir =  detect_config['tfrecords_dir']
    image_size =  detect_config['image_size']
    classes_name_file =  detect_config['classes_name_file']
    model_config_file =  detect_config['model_config_file']
    input_weights_path =  detect_config['input_weights_path']
    anchors_file =  detect_config['anchors_file']
    nms_iou_threshold =  detect_config['nms_iou_threshold']
    nms_score_threshold =  detect_config['nms_score_threshold']
    yolo_max_boxes =  detect_config['yolo_max_boxes']

    anchors_table = tf.cast(get_anchors(anchors_file), tf.float32)

    evaluate_iou_threshold = 0.5

    gt_classes = ['gt_classes', 'ff']
    evaluate = Evaluate(nclasses=7, iou_thresh=evaluate_iou_threshold)
    evaluate.calc_ap(gt_classes, tfrecords_dir, evaluate_iou_threshold, image_size, classes_name_file, model_config_file, input_weights_path, anchors_table,
                nms_iou_threshold, nms_score_threshold, yolo_max_boxes)


    count_true_positives ={}