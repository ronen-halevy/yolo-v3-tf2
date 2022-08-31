
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

    def tp_fp_counts(self, result, class_index, tp, fp):
        t = result == class_index
        # self.inc_count(tp[class_index])
        # tf.cond(result, true_fn=lambda: self.inc_count(tp[class_index]), false_fn=lambda: self.inc_count(fp[class_index]))

    def do_iou(self, pred_y, true_y, tp, fp, fn, objects_count):
        t_unassigned = tf.Variable(tf.fill(tf.shape(true_y[...,1]), 1))

        t_bboxes, t_confidences, t_classes_indices = tf.split(true_y, [4,1,1], axis=-1)
        # p_bboxes, p_classes_indices = tf.split(tf.squeeze(pred_y, axis=0), [4,1], axis=-1)
        p_bboxes, p_classes_indices = tf.split(pred_y, [4,1], axis=-1)

        iou = tf.map_fn(fn=lambda t: self.broadcast_iou(t, t_bboxes), elems=p_bboxes, parallel_iterations=3)
        best_iou_index = tf.math.argmax(iou, axis=-1, output_type=tf.int32).numpy()
        best_iou_index = tf.cast(tf.squeeze(best_iou_index, axis=-1), tf.int32)
        t_classes_indices =  tf.cast(tf.squeeze(t_classes_indices, axis=-1), tf.int32)
        true_class = [t_classes_indices[idx] for idx in best_iou_index]
        iou = tf.squeeze(iou, axis=1)
        sel_iou = np.ndarray([tf.shape(iou)[0]])
        ## ronen todo gather?
        for idx, (iou_, sel_index) in enumerate(zip(iou, best_iou_index)):
            sel_iou[idx] = iou_[sel_index]

        valid_detection = sel_iou > self.iou_thresh
        true_class = tf.convert_to_tensor(true_class)
        p_classes_indices = tf.cast(tf.squeeze(p_classes_indices, axis=-1), tf.int32)
        valid_class = true_class == p_classes_indices
        res = tf.math.logical_and(valid_detection, valid_class)
        t_unassigned_sel = tf.gather(t_unassigned, best_iou_index)
        res = tf.math.logical_and(res, tf.cast(t_unassigned_sel, bool))
        tf.map_fn(fn=lambda t: t_unassigned[t].assign(0), elems=best_iou_index, parallel_iterations=3)


        updates = tf.cast(res, dtype=tf.int32)
        # tf.ones(tf.shape(p_classes_indices)[0], dtype=tf.int32)
        indices = tf.expand_dims(p_classes_indices, axis=-1)
        tp = tf.tensor_scatter_nd_add(tp, indices, updates)
        updates = tf.cast(tf.math.logical_not(res), dtype=tf.int32)
        fp = tf.tensor_scatter_nd_add(fp, indices, updates)

        # t_unassigned, t_classes_indices)
        updates = tf.cast(t_unassigned, dtype=tf.int32)
        indices = tf.expand_dims(t_classes_indices, axis=-1)
        fn = tf.tensor_scatter_nd_add(fn, indices, updates)


        updates = tf.ones(tf.shape(t_classes_indices)[0], dtype=tf.int32)
        indices = tf.expand_dims(t_classes_indices, axis=-1)
        objects_count = tf.tensor_scatter_nd_add(objects_count, indices, updates)



        #
        # tp, fp = tf.map_fn(fn=lambda t: tf.cond(t[0],
        #                                true_fn=lambda: tf.tensor_scatter_nd_add(tp, [t[1]], updates)
        #                                self.inc_left_arg(tp[t[1]], fp[t[1]]),
        #                                false_fn=lambda: self.inc_right_arg(fp[t[1]], tp[t[1]])),
        #           elems=(res, p_classes_indices),
        #           dtype=(tf.int32, tf.int32))
        #
        # fn = tf.map_fn(fn=lambda t: tf.cond(t[0],
        #                                         true_fn=lambda: self.inc_arg(fn[t[1]]),
        #                                         false_fn=lambda: self.nop_func(fn[t[1]])),
        #                    elems=(t_unassigned, t_classes_indices),
        #                    dtype=(tf.int32))
        #
        # if tf.shape(tp)[0] != 3:
        #     print(tf.shape(tp))
        return tp, fp, fn, objects_count
        # tp, fp = tf.map_fn(fn=lambda t: tf.cond(t, true_fn=self.inc_count(res) , false_fn=self.inc_count1(res)), elems=res, parallel_iterations=3)

    # def inc_fp(self, fp, classes_indices):
    #     pass
    # def calc_tp_fp_fn(self, pred_y, true_y, nclasses):
    #     # t_bboxes, t_confidences, t_classes_indices = tf.split(true_y, [4,1,1], axis=-1)
    #     # p_bboxes, p_classes_indices = tf.split(tf.squeeze(pred_y, axis=0), [4,1], axis=-1)
    #
    #     tp = tf.convert_to_tensor([0] * nclasses)
    #     fp = tf.convert_to_tensor([0] * nclasses)
    #     fn = tf.convert_to_tensor([0] * nclasses)
    #
    #     # t_unassigned = tf.Variable(tf.fill(tf.shape(true_y[...,1]), 1))
    #     # true_y = tf.concat([true_y, t_assigned], -1)
    #     # t_assigned_to_pred = [False] * tf.shape(t_confidences).numpy()[0]
    #     # usw tf.cond!!!!
    #
    #     tp, fp, fn = tf.cond(tf.shape(true_y)[0] != 0,
    #                          true_fn=  lambda: self.do_iou(pred_y, true_y, tp, fp, fn),
    #                          false_fn= lambda: self.inc_arg(fp, pred_y[...,4]))
    #
    #     print(tp, fp, fn)
        # tp = tf.cond(tf.shape(true_y)[0] != 0,
        #                      true_fn=  lambda: self.do_iou(pred_y, true_y, t_unassigned, tp, fp, fn),
        #                      false_fn= lambda: self.inc_args(fp, pred_y[...,4]))
        # a=tf.convert_to_tensor(4)
        # b = tf.convert_to_tensor(5)
        # result = tf.cond(tf.convert_to_tensor(True), lambda: tf.math.multiply(a, b), lambda: tf.math.add(a, b))
        # result = tf.cond(tf.convert_to_tensor(True), lambda: self.do_iou(pred_y, true_y, t_unassigned, tp, fp, fn), lambda: tf.math.add(a, b))

        # tf.cond(True, true_fn=self.do_iou(pred_y, true_y, t_unassigned, tp, fp, fn), false_fn=self.do_iou(pred_y, true_y, t_unassigned, tp, fp, fn))

        # print(x)
        # pass
        # if tf.shape(true_y)[0] != 0:
        #     iou = tf.map_fn(fn=lambda t: self.broadcast_iou(t, t_bboxes), elems=p_bboxes, parallel_iterations=3)
        #     best_iou_index = tf.math.argmax(iou, axis=-1, output_type=tf.int32).numpy()
        #     best_iou_index = tf.cast(tf.squeeze(best_iou_index, axis=-1), tf.int32)
        #     t_classes_indices = tf.squeeze(t_classes_indices, axis=-1)
        #     true_class = [t_classes_indices[idx] for idx in best_iou_index]
        #     iou = tf.squeeze(iou, axis=1)
        #     sel_iou = np.ndarray([tf.shape(iou)[0]])
        #     for idx, (iou_, sel_index) in enumerate(zip(iou, best_iou_index)):
        #         sel_iou[idx] = iou_[sel_index]
        #
        #
        #     valid_detection = sel_iou > self.iou_thresh
        #     true_class = tf.convert_to_tensor(true_class)
        #     p_classes_indices = tf.squeeze(p_classes_indices, axis=-1)
        #
        #     valid_class = true_class==p_classes_indices
        #     t_not_assigned_to_pred = tf.math.logical_not(tf.convert_to_tensor(t_assigned_to_pred))
        #     res = tf.math.logical_and(valid_detection,valid_class)
        #     res = tf.math.logical_and(res,t_not_assigned_to_pred)
        #     # tp, fp = tf.map_fn(fn=lambda t: tf.cond(t, true_fn=tp[], elems=res, parallel_iterations=3)
        #
        #
        #     print(res)


        #     for p_bbox, p_class_index in zip(p_bboxes, p_classes_indices):
        #         if iou[...,best_iou_index] > self.iou_thresh and true_class == p_class_index and not t_assigned_to_pred[best_iou_index]:
        #         t_assigned_to_pred[best_iou_index]=True
        #
        #
        # else:
        #     fp += tf.shape(p_bboxes).numpy()[0]
        #
        # for p_bbox, p_class_index  in zip(p_bboxes, p_classes_indices):
        #     if tf.shape(t_bboxes)[0] != 0:
        #         iou = self.broadcast_iou(p_bbox, t_bboxes)
        #     else:
        #         fp += 1
        #         continue
        #     best_iou_index = tf.math.argmax(
        #             iou, axis=-1, output_type=tf.int32).numpy()[0]
        #     true_class =  int(t_classes_indices[best_iou_index].numpy()[0])
        #
        #     if iou[...,best_iou_index] > self.iou_thresh and true_class == p_class_index and not t_assigned_to_pred[best_iou_index]:
        #         t_assigned_to_pred[best_iou_index]=True
        #
        #         tp[true_class]+=1
        #     else:
        #         pass
        #         fp[true_class]+=1
        # for status in t_assigned_to_pred:
        #     if not status:
        #         fn[true_class]+=1
        #
        # return tp, fp, fn


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
        model = self.create_model(model_config_file, nclasses, nms_score_threshold, nms_iou_threshold)

        dataset = dataset.map(lambda x, y: (self.inference_func(x, model), y))


        data = dataset.map(lambda x, y: (tf.concat([x[0], tf.expand_dims(tf.cast(x[2], dtype=tf.float32), axis=-1)], axis=-1), y))
        # tp = fp = fn = [0] * nclasses



        tp = tf.convert_to_tensor([0] * nclasses)
        fp = tf.convert_to_tensor([0] * nclasses)
        fn = tf.convert_to_tensor([0] * nclasses)
        objects_count = tf.convert_to_tensor([0] * nclasses)

        for idx, (pred_y, true_y) in enumerate(data):
            pred_y =tf.squeeze(pred_y, axis=0)

            tp, fp, fn, objects_count = tf.cond(tf.shape(true_y)[0] != 0,
                                 true_fn=lambda: self.do_iou(pred_y, true_y, tp, fp, fn, objects_count),
                                 false_fn=lambda: self.inc_arg(fp, pred_y[..., 4]))

            # print(tp, fp, fn)


        # hist =  data.map(lambda x, y:  tf.histogram_fixed_width(y[:,5], value_range=[0, nclasses], nbins=nclasses))
        # objects_count = hist.reduce(np.int32(0), lambda x, y: tf.cast(x, tf.int32)+y).numpy()
        tn = objects_count - tp -fp - fn

        accuracy = tf.cast(tp+tn, tf.float32)/(tf.cast(objects_count, tf.float32) + 1e-10)
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