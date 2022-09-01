
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
# import numpy as np
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

    def inc_left_arg(self, left_arg, right_arg):
        counter = tf.math.add(left_arg,1)
        return counter, right_arg

    def inc_right_arg(self, left_arg, right_arg):
        counter = tf.math.add(right_arg,1)
        return counter, right_arg

    def inc_arg(self, arg, val=1):
        counter = tf.math.add(arg,val)
        return counter
    def inc_args(self, args, val=1):
        counter = tf.map_fn(n=lambda t: tf.math.add(t,val), elems=args, parallel_iterations=3)
        return counter
    def nop_func(self, arg):
        return arg

    # @tf.function
    #todo split the iou func to iou and stats
    def do_iou(self, pred_y, true_y, tp, fp, fn, all_ref, unmatched_ref_objects_list, errors):
        ref_bboxes, ref_confidences, ref_classes_indices = tf.split(true_y, [4,1,1], axis=-1)
        p_bboxes, p_classes_indices = tf.split(pred_y, [4,1], axis=-1)

        ref_classes_indices =  tf.cast(tf.squeeze(ref_classes_indices, axis=-1), tf.int32)
        p_classes_indices = tf.squeeze(p_classes_indices, axis=-1)

        iou = tf.map_fn(fn=lambda t: self.broadcast_iou(t, ref_bboxes), elems=p_bboxes, parallel_iterations=3)
        iou = tf.squeeze(iou, axis=1)

        if tf.equal(tf.size(iou), 0):
            fn = tf.tensor_scatter_nd_add(fn, # todo arrange dup counts - + like in start
                                     tf.expand_dims(
                                         tf.cast(true_y[..., 4], tf.int32),
                                         axis=-1),
                                     tf.fill(tf.shape(true_y)[0], 1))
            all_ref = tf.tensor_scatter_nd_add(all_ref,
                                      tf.expand_dims(
                                          tf.cast(true_y[..., 4], tf.int32),
                                          axis=-1),
                                      tf.fill(tf.shape(true_y)[0], 1))
            return tp, fp, fn, all_ref, errors
        try:
            selected_ref_indices = tf.math.argmax(iou, axis=-1, output_type=tf.int32)#.numpy()
        except Exception as e:
            print(e)

        best_iou_index_2d = tf.stack([tf.range(tf.shape(iou)[0]), selected_ref_indices], axis=-1)
        sel_iou = tf.gather_nd(iou, best_iou_index_2d)

        thresh_qualified_ious = sel_iou > self.iou_thresh


        selected_ref_class = tf.gather(ref_classes_indices, selected_ref_indices)


        p_classes_indices = tf.cast(p_classes_indices, tf.int32)
        detect_matched_classes = selected_ref_class == p_classes_indices
        detect_decisions = tf.math.logical_and(thresh_qualified_ious, detect_matched_classes)

        is_ref_not_assigned = tf.cast(tf.gather(unmatched_ref_objects_list, selected_ref_indices), bool)
        detect_decisions = tf.math.logical_and(detect_decisions, is_ref_not_assigned)

        indices = tf.expand_dims(selected_ref_indices, axis=-1)
        updates = tf.zeros(tf.size(selected_ref_indices), dtype=tf.int32)
        shape = tf.shape(unmatched_ref_objects_list)
        unmatched_ref_objects_list = tf.scatter_nd(indices, updates, shape)

        # tf.map_fn(fn=lambda t: unmatched_ref_objects_list[t].assign(0), elems=selected_ref_indices, parallel_iterations=3)
        # return detect_decisions, unmatched_ref_objects_list
        indices = tf.expand_dims(p_classes_indices, axis=-1)
        tp_decisions = tf.cast(detect_decisions, dtype=tf.int32)
        tp = tf.tensor_scatter_nd_add(tp, indices, tp_decisions)
        fp_decisions = tf.cast(tf.math.logical_not(detect_decisions), dtype=tf.int32)
        fp = tf.tensor_scatter_nd_add(fp, indices, fp_decisions)

        # unmatched_ref_objects_list, ref_classes_indices)
        fn_decisions = tf.cast(unmatched_ref_objects_list, dtype=tf.int32)
        try:
            fn = tf.tensor_scatter_nd_add(fn, tf.expand_dims(ref_classes_indices, axis=-1), fn_decisions)
        except Exception as e:
            print(f'!!!Exception!!:  {e} probably negative  class id in dataset!! Skipping to next data sample!!') # todo check where -1 class come from
            errors = tf.math.add(errors, 1)
            return tp, fp, fn, all_ref, errors

        updates = tf.ones(tf.size(ref_classes_indices), dtype=tf.int32)
        indices = tf.expand_dims(ref_classes_indices, axis=-1)
        all_ref = tf.tensor_scatter_nd_add(all_ref, indices, updates)
        return tp, fp, fn, all_ref, errors



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
        # dataset = dataset.batch(8)
        dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))
        model = self.create_model(model_config_file, nclasses, nms_score_threshold, nms_iou_threshold)

        dataset = dataset.map(lambda x, y: (self.inference_func(x, model), y))


        data = dataset.map(lambda x, y: (tf.concat([x[0], tf.expand_dims(tf.cast(x[2], dtype=tf.float32), axis=-1)], axis=-1), y))
        # tp = fp = fn = [0] * nclasses



        tp = tf.convert_to_tensor([0] * nclasses)
        fp = tf.convert_to_tensor([0] * nclasses)
        fn = tf.convert_to_tensor([0] * nclasses)
        all_ref = tf.convert_to_tensor([0] * nclasses)
        errors = tf.Variable(0)
        images = tf.Variable(0)

        print(f'Stats Table:\n #image, accum tp per class, accum fp per class, accum fn per calss,  accum true objects per class, accum errors count')

        for idx, (pred_y, true_y) in enumerate(data):
            pred_y =tf.squeeze(pred_y, axis=0)
            unmatched_ref_objects_list = tf.Variable(tf.fill(tf.shape(true_y[..., 1]), 1))

            tp, fp, fn, all_ref, errors = tf.cond(tf.shape(true_y)[0] != 0,
                                                 true_fn=lambda: self.do_iou(pred_y, true_y, tp, fp, fn, all_ref, unmatched_ref_objects_list, errors),
                                                  false_fn=lambda: (tp, fp,
                                                  tf.tensor_scatter_nd_add(fn,
                                                                           tf.expand_dims(tf.cast(true_y[..., 4], tf.int32), axis=-1),
                                                                           tf.fill(tf.shape(true_y)[0], 1)),

                                                  tf.tensor_scatter_nd_add(all_ref,
                                                                           tf.expand_dims(
                                                                               tf.cast(true_y[..., 4], tf.int32),
                                                                               axis=-1),
                                                                           tf.fill(tf.shape(true_y)[0], 1)),errors
                                                                    )
                                                )# rone todo fix inc_arg
            images = tf.math.add(images, 1)

            print(images.numpy(), tp.numpy(), fp.numpy(), fn.numpy(), all_ref.numpy(), errors.numpy())

        tn = all_ref - tp -fp - fn

        accuracy = tf.cast(tp+tn, tf.float32)/(tf.cast(all_ref, tf.float32) + 1e-10)
        recall = tf.cast(tp, tf.float32) / (tf.cast(tp + fn, tf.float32)  + 1e-20)
        precision = tf.cast(tp, tf.float32) / (tf.cast(tp + fp, tf.float32)  + 1e-20)
        print(accuracy, recall, precision)



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