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


class Evaluate:
    def __init__(self, nclasses, iou_thresh):
        self.nclasses = nclasses
        self.iou_thresh = iou_thresh
        # clear stats counters:
        self.counters = {
            'preds': tf.convert_to_tensor([0] * nclasses),
            'gts': tf.convert_to_tensor([0] * nclasses),
            'tp': tf.convert_to_tensor([0] * nclasses),
            'fp': tf.convert_to_tensor([0] * nclasses),
            'fn': tf.convert_to_tensor([0] * nclasses),
            'errors': tf.Variable(0),
            'examples': tf.Variable(0)
        }

    @staticmethod
    @tf.function
    def iou_alg(box_1, box_2):
        box_2 = tf.expand_dims(box_2, 0)

        overlap_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        overlap_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        overlap_area = overlap_w * overlap_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

        return overlap_area / (box_1_area + box_2_area - overlap_area)

    @tf.function
    def update_counter(self, counter, indices, update):
        indices = tf.expand_dims(indices, axis=-1)
        counter = tf.tensor_scatter_nd_add(counter, indices, update)
        return counter

    @tf.function
    def update_counters(self, p_classes_indices, gt_classes_indices, detect_decisions,
                        gt_boxes_assigned, preds, gts, tp, fp, fn, errors, examples):
        # tp per-class counter is incremented if pred's detect_decisions is set
        tp = self.update_counter(tp, p_classes_indices, tf.cast(detect_decisions, dtype=tf.int32))
        # fp per-class counter is incremented if pred's detect_decision is not set
        fp = self.update_counter(fp, p_classes_indices, tf.cast(tf.math.logical_not(detect_decisions), dtype=tf.int32))
        try:
            # fn per class counter is set if gt entry was not assigned to a pred box
            fn = self.update_counter(fn, gt_classes_indices,
                                     tf.cast(tf.math.logical_not(gt_boxes_assigned), tf.int32))
        except Exception as e:
            print(
                f'!!!Exception!!:  {e} probably negative  class id in dataset!! Skipping to next data sample!!')  # todo check where -1 class come from
            errors = tf.math.add(errors, 1)
            return preds, gts, tp, fp, fn, errors, examples
        # count all gt boxes - per class:
        gts = self.update_counter(gts, gt_classes_indices,
                                  tf.ones(tf.size(gt_classes_indices), dtype=tf.int32))
        # count all preds boxes - per class:
        preds = self.update_counter(preds, p_classes_indices,
                                    tf.ones(tf.size(p_classes_indices), dtype=tf.int32))
        examples = tf.math.add(examples, 1)
        return {'preds': preds, 'gts': gts, 'tp': tp, 'fp': fp, 'fn': fn, 'errors': errors, 'examples': examples}

    @tf.function
    def process_decisions(self, max_iou, preds_classes, max_iou_args_indices, gt_classes_indices, gt_boxes_assigned):

        # Following iou between each pred box and all gt boxes, and selection of iou max gt entry per each pred box,
        # the results will here pass through 3 qualifications:
        # 1. Thresholding: Max iou should exceed a static thresh,
        # 2. Class matching: Pred class matched with selected gt class
        # 3. Availability: Selected gt entry was not assigned already by another pred box

        # A. If all quals passed, then related entry in detect_decisions list is set
        # B. Set gt_boxes_assigned per-gt entry list,  according to recent decisions

        # 1. Thresholding:
        thresh_qualified_ious = max_iou > self.iou_thresh

        # 2. Class matching: check if pred class matches with iou selected gt class:
        selected_classes = tf.gather(gt_classes_indices, max_iou_args_indices)
        detect_matched_classes = selected_classes == tf.cast(preds_classes, tf.int32)


        detect_decisions = tf.math.logical_and(detect_matched_classes, thresh_qualified_ious)

        # 3. Availability: Check if Selected gt entry was not assigned already by another pred box
        is_gt_assigned = tf.gather(gt_boxes_assigned, max_iou_args_indices)

        # A. If all 3 quals passed, then related entry in detect_decisions list is set
        # shape(detect_decisions) equals number of max_iou_args_indices, so each entry relates to a p pred and a gt box
        detect_decisions = tf.math.logical_and(detect_decisions, tf.math.logical_not(is_gt_assigned))

        # B. Set gt_boxes_assigned per-gt entry list, according to recent decisions
        indices = tf.expand_dims(max_iou_args_indices, axis=-1)
        # Use tensor_scatter_nd_add - if at least one decision entry hits an index - set it true:
        # So cast to int, add, and then cast back to bool. If indices = [0, 1, 1] and detect_decisions:
        # [True, True Fasle] result still [True , True]
        gt_boxes_assigned = \
            tf.tensor_scatter_nd_add(tf.cast(gt_boxes_assigned, tf.int32), indices, tf.cast(detect_decisions, tf.int32))
        gt_boxes_assigned = tf.cast(gt_boxes_assigned, tf.bool)
        detect_decisions = tf.cast(detect_decisions, tf.bool)


        return detect_decisions, gt_boxes_assigned


    @tf.function
    def calc_iou(self, p_bboxes, preds_classes, gt_bboxes):
        # Select max iou between each image's pred_box and each of gt_boxes. iou shape = p_boxes x gt_boxes
        iou = tf.map_fn(fn=lambda t: self.iou_alg(t, gt_bboxes), elems=p_bboxes, parallel_iterations=3)
        iou = tf.squeeze(iou, axis=1)
        # Select the indices of the best matching gt entry per each pred box:
        max_iou_args_indices = tf.math.argmax(iou, axis=-1, output_type=tf.int32)
        best_iou_index_2d = tf.stack([tf.range(tf.size(preds_classes)), max_iou_args_indices],
                                     axis=-1)
        max_iou = tf.gather_nd(iou, best_iou_index_2d)

        return max_iou, max_iou_args_indices

    @staticmethod
    def gather_nms_output(bboxes_padded, class_indices_padded, scores_padded,
                          selected_indices_padded, num_valid_detections):

        bboxes = tf.gather(bboxes_padded, selected_indices_padded[:num_valid_detections], axis=0)
        classes = tf.gather(class_indices_padded, selected_indices_padded[:num_valid_detections], axis=0)
        scores = tf.gather(scores_padded, selected_indices_padded[:num_valid_detections], axis=0)
        return bboxes, classes, scores

    #
    # def evaluate(self, tfrecords_dir, image_size, batch_size, classes_name_file, model_config_file, input_weights_path,
    #              anchors_table,
    #              nms_iou_threshold, nms_score_threshold, yolo_max_boxes=100):
    def evaluate(self, model, dataset):

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

            # Loop on prediction batched results: boxes, classes, scores, num_valid_detections, and images and gt_y.
            # Evaluate iou between predictions and gt:
            for image_index, \
                (bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded, num_valid_detections,
                 image, gt_y) \
                    in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                                     batch_selected_indices_padded, batch_num_valid_detections,
                                     batch_images, batch_gt_y)):
                # gather padded nms results using selected_indices_padded. Indices were 'padded',
                # so take num_valid_detections indices from top:
                pred_bboxes, pred_classes, pred_scores = self.gather_nms_output(bboxes_padded,
                                                                                class_indices_padded,
                                                                                scores_padded,
                                                                                selected_indices_padded,
                                                                                num_valid_detections)

                # Init track fp per image: List size is number of gt boxes.
                # Init val 1 is set to 0 when gt box is matched with a pred box
                gt_boxes_assigned = tf.fill(tf.shape(gt_y[..., 1]), False)
                # run iou, if gt_y had any box. Otherwise, increment fp according to pred_classes.
                # Anyway, return the various stats counters

                gt_bboxes, _, gt_classes_indices = tf.split(gt_y, [4, 1, 1], axis=-1)
                gt_classes_indices = tf.squeeze(tf.cast(gt_classes_indices, tf.int32), axis=-1)

                max_iou, max_iou_args_indices = self.calc_iou(pred_bboxes, pred_classes, gt_bboxes)

                detect_decisions, gt_boxes_assigned = self.process_decisions(max_iou,
                                                                             pred_classes,
                                                                             max_iou_args_indices,
                                                                             gt_classes_indices,
                                                                             gt_boxes_assigned)

                self.counters = self.update_counters(pred_classes, gt_classes_indices, detect_decisions,
                                                     gt_boxes_assigned,
                                                     **self.counters)

                for counter in self.counters:
                    print(f' {counter}: {self.counters[counter].numpy()}', end='')
                print('')

        # Resultant Stats:
        recall = tf.cast(self.counters['tp'], tf.float32) / (
                    tf.cast(self.counters['tp'] + self.counters['fn'], tf.float32) + 1e-20)
        precision = tf.cast(self.counters['tp'], tf.float32) / (
                    tf.cast(self.counters['tp'] + self.counters['fp'], tf.float32) + 1e-20)
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

        evaluate = Evaluate(nclasses=7, iou_thresh=evaluate_iou_threshold)
        results = []
        for nms_score_threshold in evaluate_nms_score_thresholds:
            model = create_model(model_config_file, nclasses, anchors_table, nms_score_threshold, nms_iou_threshold,
                                 yolo_max_boxes, input_weights_path)
            recall, precision = evaluate.evaluate(model, dataset)

            results.append((recall, precision))
        print(results)


    main()
