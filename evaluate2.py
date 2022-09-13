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

import tensorflow as tf
from tensorflow.keras import Input, Model

import yaml
import matplotlib.pyplot as plt

from core.load_tfrecords import parse_tfrecords
from core.parse_model import ParseModel

from core.yolo_decode_layer import YoloDecoderLayer
from core.yolo_nms_layer import YoloNmsLayer
from core.utils import get_anchors, resize_image

from evaluate_detections import EvaluateDetections

def arrange_yolov3_predict_output(batch_bboxes_padded, batch_class_indices_padded,
                                  batch_scores_padded,
                                  batch_selected_indices_padded, \
                                  batch_num_valid_detections, batch_gt_y):

    bboxes_batch = []
    classes_batch = []
    scores_batch = []
    gt_bboxes_batch = []
    gt_classes_batch = []

    for image_index, \
        (bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded, num_valid_detections,
         gt_y) \
            in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                             batch_selected_indices_padded, batch_num_valid_detections,
                             batch_gt_y)):
        bboxes = tf.gather(bboxes_padded, selected_indices_padded[:num_valid_detections], axis=0)
        classes = tf.gather(class_indices_padded, selected_indices_padded[:num_valid_detections], axis=0)
        scores = tf.gather(scores_padded, selected_indices_padded[:num_valid_detections], axis=0)
        bboxes_batch.append(bboxes)
        classes_batch.append(classes)
        scores_batch.append(scores)

        gt_bboxes, _, gt_classes = tf.split(gt_y, [4, 1, 1], axis=-1)
        gt_classes = tf.squeeze(tf.cast(gt_classes, tf.int32), axis=-1)

        gt_bboxes_batch.append(gt_bboxes)
        gt_classes_batch.append(gt_classes)
    bboxes_batch = tf.ragged.stack(bboxes_batch)
    classes_batch = tf.ragged.stack(classes_batch)
    gt_bboxes_batch = tf.stack(gt_bboxes_batch)
    gt_classes_batch = tf.stack(gt_classes_batch)
    return bboxes_batch, classes_batch, gt_bboxes_batch, gt_classes_batch,
    #

def measure_performance(model, dataset, nclasses, iou_thresh):

    eval_dets = EvaluateDetections(nclasses, iou_thresh)

    # main loop on dataset: predict, evaluate, update stats
    for batch_images, batch_gt_y in dataset:
        # prediction outputs padded results - bboxes, class indices, scores indices.
        # batch_selected_indices_padded - a list of indices to prediction nms outputs
        # batch_num_valid_detections - number of valid batch_selected_indices_padded indices. (taken from bottom)
        # shape of batch_bboxes_padded: batch*N*6
        # shape of batch_selected_indices_padded: batch*max_nms_boxes
        # shape of batch_num_valid_detections: batch * valid_boxes

        batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
        batch_num_valid_detections = model.predict(
            batch_images)

        bboxes_batch, classes_batch, gt_bboxes_batch, gt_classes_batch = arrange_yolov3_predict_output(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
        batch_num_valid_detections, batch_gt_y)


        for image_index, \
            (pred_bboxes, pred_classes, gt_bboxes, gt_classes) in enumerate(zip(bboxes_batch, classes_batch, gt_bboxes_batch, gt_classes_batch)):

            counters = eval_dets.evaluate(pred_bboxes, pred_classes, gt_bboxes, gt_classes)
        for counter in counters:
                print(f' {counter}: {counters[counter].numpy()}', end='')
        print('\n')
    #
    # Resultant Stats:
    recall = tf.cast(counters['tp'], tf.float32) / (
                tf.cast(counters['tp'] + counters['fn'], tf.float32) + 1e-20)
    precision = tf.cast(counters['tp'], tf.float32) / (
                tf.cast(counters['tp'] + counters['fp'], tf.float32) + 1e-20)
    print(f'recall: {recall}, precision: {precision}')
    return recall, precision


def prepare_dataset(tfrecords_dir, batch_size, image_size, yolo_max_boxes, classes_name_file):
    # prepare test dataset:
    dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes,
                              class_file=classes_name_file)

    dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda img, y: (resize_image(img, image_size, image_size), y))
    return dataset

def create_model(model_config_file, nclasses, anchors_table, nms_score_threshold, nms_iou_threshold, yolo_max_boxes,
                 input_weights_path):
    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    parse_model = ParseModel()
    inputs = Input(shape=(None, None, 3))
    model = parse_model.build_model(inputs, nclasses=nclasses, **model_config)

    # Note:.expect_partial() prevents warnings at exit time, since save model generates extra keys.
    model.load_weights(input_weights_path).expect_partial()
    print('weights loaded')
    model = model(inputs)

    decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
    nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                              nms_score_threshold)(decoded_output)

    model = Model(inputs, nms_output, name="yolo_nms")
    return model


if __name__ == '__main__':
    def main():
        # parser = argparse.ArgumentParser()
        with open('config/detect_config.yaml', 'r') as stream:
            detect_config = yaml.safe_load(stream)

        tfrecords_dir = detect_config['tfrecords_dir']
        image_size = detect_config['image_size']
        classes_name_file = detect_config['classes_name_file']
        model_config_file = detect_config['model_config_file']
        input_weights_path = detect_config['input_weights_path']
        anchors_file = detect_config['anchors_file']
        nms_iou_threshold = detect_config['nms_iou_threshold']
        nms_score_threshold = detect_config['nms_score_threshold']
        yolo_max_boxes = detect_config['yolo_max_boxes']
        batch_size = detect_config['batch_size']
        evaluate_nms_score_thresholds = detect_config['evaluate_nms_score_thresholds']
        anchors_table = tf.cast(get_anchors(anchors_file), tf.float32)

        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)

        dataset = prepare_dataset(tfrecords_dir, batch_size, image_size, yolo_max_boxes, classes_name_file)

        evaluate_iou_threshold = 0.5

        results = []
        for nms_score_threshold in evaluate_nms_score_thresholds:
            model = create_model(model_config_file, nclasses, anchors_table, nms_score_threshold, nms_iou_threshold,
                                 yolo_max_boxes, input_weights_path)
            recall, precision = measure_performance(model, dataset, nclasses=7, iou_thresh=evaluate_iou_threshold)

            results.append((recall, precision))
        print(results)


    main()
