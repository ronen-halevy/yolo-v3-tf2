#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : evaluate_detections.py
#   Author      : ronen halevy 
#   Created date:  9/13/22
#   Description :
#
# ================================================================


import tensorflow as tf

class EvaluateDetections:
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
        self.preds_histo = []
        self.gt_histo = []
        self.tp_histo = []
        self.fp_histo = []
        self.fn_histo = []
        self.old_counters = None

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
    def update_counters(self, p_classes_indices, gt_classes, detect_decisions,
                        gt_boxes_assigned, preds, gts, tp, fp, fn, errors, examples):
        # tp per-class counter is incremented if pred's detect_decisions is set

        tp_updated = self.update_counter(tp, p_classes_indices, tf.cast(detect_decisions, dtype=tf.int32))
        # fp per-class counter is incremented if pred's detect_decision is not set
        fp_updated = self.update_counter(fp, p_classes_indices, tf.cast(tf.math.logical_not(detect_decisions), dtype=tf.int32))
        try:
            # fn per class counter is set if gt entry was not assigned to a pred box
            fn_updated = self.update_counter(fn, gt_classes,
                                     tf.cast(tf.math.logical_not(gt_boxes_assigned), tf.int32))
        except Exception as e:
            print(
                f'!!!Exception!!:  {e} probably negative  class id in dataset!! Skipping to next data sample!!')  # todo check where -1 class come from
            errors = tf.math.add(errors, 1)
            return preds, gts, tp, fp, fn, errors, examples
        # count all gt boxes - per class:
        gts_updated = self.update_counter(gts, gt_classes,
                                  tf.ones(tf.size(gt_classes), dtype=tf.int32))
        # count all preds boxes - per class:
        preds_updated = self.update_counter(preds, p_classes_indices,
                                    tf.ones(tf.size(p_classes_indices), dtype=tf.int32))
        examples = tf.math.add(examples, 1)
        return {'preds': preds_updated, 'gts': gts_updated, 'tp': tp_updated, 'fp': fp_updated, 'fn': fn_updated, 'errors': errors, 'examples': examples}

    @tf.function
    def process_decisions(self, max_iou, preds_classes, max_iou_args_indices, gt_classes, gt_boxes_assigned):

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
        selected_classes = tf.gather(gt_classes, max_iou_args_indices)
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
    def evaluate(self, pred_bboxes, pred_classes, gt_bboxes, gt_classes):
        gt_boxes_assigned = tf.fill(tf.shape(gt_classes), False)

        max_iou, max_iou_args_indices = self.calc_iou(pred_bboxes.to_tensor(), pred_classes, gt_bboxes)

        detect_decisions, gt_boxes_assigned = self.process_decisions(max_iou,
                                                                     pred_classes,
                                                                     max_iou_args_indices,
                                                                     gt_classes,
                                                                     gt_boxes_assigned)

        self.counters = self.update_counters(pred_classes, gt_classes, detect_decisions,
                                             gt_boxes_assigned,
                                             **self.counters)

        if self.old_counters:
            self.preds_histo.append(self.counters['preds'] - self.old_counters['preds'])
            self.gt_histo.append(self.counters['gts'] - self.old_counters['gts'])

            self.tp_histo.append(self.counters['tp'] - self.old_counters['tp'])
            self.fp_histo.append(self.counters['fp'] - self.old_counters['fp'])
            self.fn_histo.append(self.counters['fn'] - self.old_counters['fn'])
        else:  # if first time:
            self.fp_histo.append(self.counters['preds'])
            self.tp_histo.append(self.counters['gts'])
            self.fp_histo.append(self.counters['tp'])
            self.tp_histo.append(self.counters['tp'])
            self.fp_histo.append(self.counters['fn'])
        self.old_counters = self.counters

        return self.counters


    @staticmethod
    def gather_nms_output(bboxes_padded, class_indices_padded, scores_padded,
                          selected_indices_padded, num_valid_detections):

        bboxes = tf.gather(bboxes_padded, selected_indices_padded[:num_valid_detections], axis=0)
        classes = tf.gather(class_indices_padded, selected_indices_padded[:num_valid_detections], axis=0)
        scores = tf.gather(scores_padded, selected_indices_padded[:num_valid_detections], axis=0)
        return bboxes, classes, scores
