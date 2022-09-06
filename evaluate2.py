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
    def update_counters(self, p_classes_indices, ref_classes_indices, detect_decisions, selected_ref_indices,
                        ref_boxes_assigned, preds, refs, tp, fp, fn, errors, examples):
        # tp per-class counter is incremented if pred's detect_decisions is set
        tp = self.update_counter(tp, p_classes_indices, tf.cast(detect_decisions, dtype=tf.int32))
        # fp per-class counter is incremented if pred's detect_decision is not set
        fp = self.update_counter(fp, p_classes_indices, tf.cast(tf.math.logical_not(detect_decisions), dtype=tf.int32))
        try:
            # fn per class counter is set if ref entry was not assigned to a pred box
            fn = self.update_counter(fn, ref_classes_indices,
                                     tf.cast(tf.math.logical_not(ref_boxes_assigned), tf.int32))
        except Exception as e:
            print(
                f'!!!Exception!!:  {e} probably negative  class id in dataset!! Skipping to next data sample!!')  # todo check where -1 class come from
            errors = tf.math.add(errors, 1)
            return preds, refs, tp, fp, fn, errors, examples
        # count all ref boxes - per class:
        refs = self.update_counter(refs, ref_classes_indices,
                                   tf.ones(tf.size(ref_classes_indices), dtype=tf.int32))
        # count all preds boxes - per class:
        preds = self.update_counter(preds, p_classes_indices,
                                    tf.ones(tf.size(p_classes_indices), dtype=tf.int32))
        examples = tf.math.add(examples, 1)
        return {'preds': preds, 'refs': refs, 'tp': tp, 'fp': fp, 'fn': fn, 'errors': errors, 'examples': examples}

    @tf.function
    def process_decisions(self, max_iou, preds_classes, selected_ref_indices, ref_classes_indices, ref_boxes_assigned):

        # Following iou between each pred box and all ref boxes, and selection of iou max ref entry per each pred box,
        # the results will here pass through 3 qualifications:
        # 1. Thresholding: Max iou should exceed a static thresh,
        # 2. Class matching: Pred class matched with selected ref class
        # 3. Availability: Selected ref entry was not assigned already by another pred box

        # A. If all quals passed, then related entry in detect_decisions list is set
        # B. Set ref_boxes_assigned per-ref entry list,  according to recent decisions

        # 1. Thresholding:
        thresh_qualified_ious = max_iou > self.iou_thresh

        # 2. Class matching: check if pred class matches with iou selected ref class:
        selected_classes = tf.gather(ref_classes_indices, selected_ref_indices)
        detect_matched_classes = selected_classes == tf.cast(preds_classes, tf.int32)
        detect_decisions = tf.math.logical_and(detect_matched_classes, thresh_qualified_ious)

        # 3. Availability: Check if Selected ref entry was not assigned already by another pred box
        is_ref_assigned = tf.gather(ref_boxes_assigned, selected_ref_indices)

        # A. If all 3 quals passed, then related entry in detect_decisions list is set
        # shape(detect_decisions) equals number of selected_ref_indices, so each entry relates to a p pred and a ref box
        detect_decisions = tf.math.logical_and(detect_decisions, tf.math.logical_not(is_ref_assigned))

        # B. Set ref_boxes_assigned per-ref entry list, according to recent decisions
        indices = tf.expand_dims(selected_ref_indices, axis=-1)
        ref_boxes_assigned = tf.tensor_scatter_nd_update(ref_boxes_assigned, indices, detect_decisions)

        return detect_decisions, ref_boxes_assigned

    @tf.function
    def calc_iou(self, p_bboxes, preds_classes, ref_bboxes):
        # Select max iou between each image's pred_box and each of ref_boxes. iou shape = p_boxes x ref_boxes
        iou = tf.map_fn(fn=lambda t: self.iou_alg(t, ref_bboxes), elems=p_bboxes, parallel_iterations=3)
        iou = tf.squeeze(iou, axis=1)
        # Select the indices of the best matching ref entry per each pred box:
        selected_ref_indices = tf.math.argmax(iou, axis=-1, output_type=tf.int32)
        best_iou_index_2d = tf.stack([tf.range(tf.size(preds_classes)), selected_ref_indices],
                                     axis=-1)
        max_iou = tf.gather_nd(iou, best_iou_index_2d)

        return max_iou, selected_ref_indices

    @staticmethod
    def gather_nms_output(bboxes_padded, class_indices_padded, scores_padded,
                          selected_indices_padded, num_valid_detections):

        bboxes = tf.gather(bboxes_padded, selected_indices_padded[:num_valid_detections], axis=0)
        classes = tf.gather(class_indices_padded, selected_indices_padded[:num_valid_detections], axis=0)
        scores = tf.gather(scores_padded, selected_indices_padded[:num_valid_detections], axis=0)
        return bboxes, classes, scores

    @staticmethod
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

    def evaluate(self, tfrecords_dir, image_size, batch_size, classes_name_file, model_config_file, input_weights_path,
                 anchors_table,
                 nms_iou_threshold, nms_score_threshold, yolo_max_boxes=100):

        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)

        # create model:
        model = self.create_model(model_config_file, nclasses, anchors_table, nms_score_threshold, nms_iou_threshold,
                                  yolo_max_boxes, input_weights_path)

        # prepare test dataset:
        dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes,
                                  class_file=classes_name_file)

        dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))  # todo verify this
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda img, y: (resize_image(img, image_size, image_size), y))

        # clear stats counters:
        counters = {
            'preds': tf.convert_to_tensor([0] * nclasses),
            'refs': tf.convert_to_tensor([0] * nclasses),
            'tp': tf.convert_to_tensor([0] * nclasses),
            'fp': tf.convert_to_tensor([0] * nclasses),
            'fn': tf.convert_to_tensor([0] * nclasses),
            'errors': tf.Variable(0),
            'examples': tf.Variable(0)
        }

        # main loop on dataset: predict, evaluate, update stats
        for batch_images, batch_ref_y in dataset:
            # prediction outputs padded results - bboxes, class indices, scores indices.
            # batch_selected_indices_padded - a list of indices to prediction nms outputs
            # batch_num_valid_detections - number of valid batch_selected_indices_padded indices. (taken from bottom)
            # shape of batch_bboxes_padded: batch*N*6
            # shape of batch_selected_indices_padded: batch*max_nms_boxes
            # shape of batch_num_valid_detections: batch * valid_boxes
            batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
            batch_num_valid_detections = model.predict(
                batch_images)

            # evaluate predictions results using iou: batches loop
            for image_index, \
                (bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded, num_valid_detections,
                 image, ref_y) \
                    in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                                     batch_selected_indices_padded, batch_num_valid_detections,
                                     batch_images, batch_ref_y)):
                # gather padded nms results using selected_indices_padded. Indices were 'padded', so take num_valid_detections indices from top:
                pred_bboxes, pred_classes, pred_scores = self.gather_nms_output(bboxes_padded,
                                                                                class_indices_padded,
                                                                                scores_padded,
                                                                                selected_indices_padded,
                                                                                num_valid_detections)

                # Init track fp per image: List size is number of ref boxes. Init val 1 is set to 0 when ref box is matched with a pred box
                ref_boxes_assigned = tf.fill(tf.shape(ref_y[..., 1]), False)

                # run iou, if ref_y had any box. Otherwise, increment fp according to pred_classes.
                # Anyway, return the various stats counters

                ref_bboxes, _, ref_classes_indices = tf.split(ref_y, [4, 1, 1], axis=-1)

                # if tf.shape(ref_bboxes)[0] != 0:
                ref_classes_indices = tf.squeeze(tf.cast(ref_classes_indices, tf.int32), axis=-1)

                max_iou, selected_ref_indices = self.calc_iou(pred_bboxes, pred_classes, ref_bboxes)

                detect_decisions, ref_boxes_assigned = self.process_decisions(max_iou,
                                                                              pred_classes,
                                                                              selected_ref_indices,
                                                                              ref_classes_indices,
                                                                              ref_boxes_assigned)

                counters = self.update_counters(pred_classes, ref_classes_indices, detect_decisions,
                                                selected_ref_indices,
                                                ref_boxes_assigned, **counters)

                # else:
                #     counters.update({'fp': tf.tensor_scatter_nd_add(counters['fp'], tf.expand_dims(tf.cast(pred_classes,
                #                                                                                            tf.int32),
                #                                                                                    axis=-1),
                #                                                     tf.fill(
                #                                                         tf.shape(
                #                                                             pred_classes)[
                #                                                             0],
                #                                                         1))})

                for counter in counters:
                    print(f' {counter}: {counters[counter].numpy()}', end='')
                print('')

        # Resultant Stats:
        recall = tf.cast(counters['tp'], tf.float32) / (tf.cast(counters['tp'] + counters['fn'], tf.float32) + 1e-20)
        precision = tf.cast(counters['tp'], tf.float32) / (tf.cast(counters['tp'] + counters['fp'], tf.float32) + 1e-20)
        print(f'recall: {recall}, precision: {precision}')
        return recall, precision


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

        anchors_table = tf.cast(get_anchors(anchors_file), tf.float32)

        evaluate_iou_threshold = 0.5

        evaluate = Evaluate(nclasses=7, iou_thresh=evaluate_iou_threshold)
        results = []
        for nms_score_threshold in [0.004, 0.1, 0.2, 0.5, 0.9]:
            recal, precision = evaluate.evaluate(tfrecords_dir, image_size, batch_size, classes_name_file,
                                                 model_config_file,
                                                 input_weights_path,
                                                 anchors_table,
                                                 nms_iou_threshold, nms_score_threshold, yolo_max_boxes)
            results.append((recal, precision))


    main()
