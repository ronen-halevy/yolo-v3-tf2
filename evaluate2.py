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
    # @tf.function
    def calc_iou(box_1, box_2):

        box_2 = tf.expand_dims(box_2, 0)

        overlap_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        overlap_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        overlap_area = overlap_w * overlap_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])

        return overlap_area / (box_1_area + box_2_area - overlap_area)

    @staticmethod
    def find_missing_entries(self, src_list, target_list):
        value_range = [-1, tf.math.reduce_max(target_list) + 1]
        nbins = value_range[1]
        src_hist = tf.histogram_fixed_width(
            src_list,
            value_range=value_range,
            nbins=nbins,
            dtype=tf.dtypes.int32,
            name=None
        )
        dst_hist = tf.histogram_fixed_width(
            target_list,
            value_range=value_range,
            nbins=nbins,
            dtype=tf.dtypes.int32,
            name=None
        )
        delta = dst_hist - src_hist
        indices = tf.where(delta)
        values = tf.gather(delta, indices)
        return indices, values

    # @tf.function
    # todo split the iou func to iou and stats
    def do_iou(self, p_bboxes, p_classes_indices, ref_y, tp, fp, fn, all_ref, ref_boxes_assigned, errors):
        ref_bboxes, _, ref_classes_indices = tf.split(ref_y, [4, 1, 1], axis=-1)

        ref_classes_indices = tf.squeeze(tf.cast(ref_classes_indices, tf.int32), axis=-1)

        iou = tf.map_fn(fn=lambda t: self.calc_iou(t, ref_bboxes), elems=p_bboxes, parallel_iterations=3)
        iou = tf.squeeze(iou, axis=1)

        if tf.equal(tf.size(iou), 0):
            # singular case: no iou. Count fn for all ref examples and return
            fn = tf.tensor_scatter_nd_add(fn,  # todo arrange dup counts - + like in start
                                          tf.expand_dims(
                                              tf.cast(ref_y[..., 4], tf.int32),
                                              axis=-1),
                                          tf.fill(tf.shape(ref_y)[0], 1))
            all_ref = tf.tensor_scatter_nd_add(all_ref,
                                               tf.expand_dims(
                                                   tf.cast(ref_y[..., 4], tf.int32),
                                                   axis=-1),
                                               tf.fill(tf.shape(ref_y)[0], 1))
            return tp, fp, fn, all_ref, errors

        # Each pred box was  iou'd with all image's ref boxes.Select ref boxes with max iou:
        selected_ref_indices = tf.math.argmax(iou, axis=-1, output_type=tf.int32)

        # Each detection passed through 3 qualifications
        # 1. iou exceeds thresh,
        # 2. class match with ref
        # 3. matched ref is not assigned already by another detection
        # If all quals passed, then inc tp. Otherwise inc fp

        # 1. Per box, gather max iou of the 3 grids values:
        best_iou_index_2d = tf.stack([tf.range(tf.shape(iou)[0]), selected_ref_indices], axis=-1)
        sel_iou = tf.gather_nd(iou, best_iou_index_2d)
        thresh_qualified_ious = sel_iou > self.iou_thresh

        # 2. check if class id match with ref:
        selected_classes = tf.gather(ref_classes_indices, selected_ref_indices)
        # tf.cond(selected_classes == tf.cast(p_classes_indices, tf.int32)), true_fn: (tf.gather(ref_classes_indices, selected_ref_indices), fp)lambda false_fn:
        detect_matched_classes = selected_classes == tf.cast(p_classes_indices, tf.int32)
        detect_decisions = tf.math.logical_and(detect_matched_classes, thresh_qualified_ious)

        # 3. Check if matched ref boxes already assigned to predictions (of current image ofcpourse):
        is_ref_assigned = tf.gather(ref_boxes_assigned, selected_ref_indices)
        # a decision is set True if 3 qualifications passe (detect_decisions size ia (selected_ref_indices)
        detect_decisions = tf.math.logical_and(detect_decisions, tf.math.logical_not(is_ref_assigned))

        # Update tp counter:
        indices = tf.expand_dims(p_classes_indices, axis=-1)
        tp_decisions = tf.cast(detect_decisions, dtype=tf.int32)
        tp = tf.tensor_scatter_nd_add(tp, indices, tp_decisions)

        # Update fp counter:
        fp_decisions = tf.cast(tf.math.logical_not(detect_decisions), dtype=tf.int32)
        indices = tf.expand_dims(p_classes_indices, axis=-1)
        fp = tf.tensor_scatter_nd_add(fp, indices, fp_decisions)

        # Update ref_boxes_assigned - set True uponf a poisitive detect_decision:
        indices = tf.expand_dims(selected_ref_indices, axis=-1)
        ref_boxes_assigned = tf.tensor_scatter_nd_update(ref_boxes_assigned, indices, detect_decisions)
        try:
            # ref_boxes_assigned is True/False if a prediction predicted class matched/unmatched with iou selected ref
            updates = tf.cast(tf.math.logical_not(ref_boxes_assigned), tf.int32)
            indices = tf.expand_dims(ref_classes_indices, axis=-1)
            fn = tf.tensor_scatter_nd_add(fn, indices, updates)
        except Exception as e:
            print(
                f'!!!Exception!!:  {e} probably negative  class id in dataset!! Skipping to next data sample!!')  # todo check where -1 class come from
            errors = tf.math.add(errors, 1)
            return tp, fp, fn, all_ref, errors

        updates = tf.ones(tf.size(ref_classes_indices), dtype=tf.int32)
        indices = tf.expand_dims(ref_classes_indices, axis=-1)
        all_ref = tf.tensor_scatter_nd_add(all_ref, indices, updates)
        return tp, fp, fn, all_ref, errors

    @staticmethod
    def gather_valid_detections_results(bboxes_padded, class_indices_padded, scores_padded,
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

        model.load_weights(input_weights_path)
        print('weights loaded')
        model = model(inputs)

        decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
        nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                                  nms_score_threshold)(decoded_output)

        model = Model(inputs, nms_output, name="yolo_nms")
        return model

    def calc_ap(self, tfrecords_dir, image_size, batch_size, classes_name_file, model_config_file, input_weights_path,
                anchors_table,
                nms_iou_threshold, nms_score_threshold, yolo_max_boxes=100):

        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)  # todo
        nclasses = 7

        # create model:
        model = self.create_model(model_config_file, nclasses, anchors_table, nms_score_threshold, nms_iou_threshold,
                                  yolo_max_boxes, input_weights_path)

        # prepare test dataset:
        dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes,
                                  class_file=classes_name_file)
        dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda img, y: (resize_image(img, image_size, image_size), y))

        # clear stats counters:
        tp = tf.convert_to_tensor([0] * nclasses)
        fp = tf.convert_to_tensor([0] * nclasses)
        fn = tf.convert_to_tensor([0] * nclasses)
        all_ref = tf.convert_to_tensor([0] * nclasses)
        errors = tf.Variable(0)
        images_cnt = tf.Variable(0)

        print(
            f'Stats Table:\n #image, accum tp per class, accum fp per class, accum fn per calss,  accum true objects per class, accum errors count')
        # main loop on dataset: predict, evaluate, update stats
        for batch_images, batch_ref_y in dataset:
            batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
            batch_num_valid_detections = model.predict(
                batch_images)

            # loop on samples: run iou and calculate stats accordingly
            for image_index, \
                (bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded, num_valid_detections,
                 image, ref_y) \
                    in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                                     batch_selected_indices_padded, batch_num_valid_detections,
                                     batch_images, batch_ref_y)):
                # gather pdded nms results using selected_indices_padded. Indices were 'padded', so take num_valid_detections indices from top:
                pred_bboxes, pred_classes, pred_scores = self.gather_valid_detections_results(bboxes_padded,
                                                                                              class_indices_padded,
                                                                                              scores_padded,
                                                                                              selected_indices_padded,
                                                                                              num_valid_detections)

                # Init track fp per image: List size is number of ref boxes. Init val 1 is set to 0 when ref box is matched with a pred box
                ref_boxes_assigned = tf.fill(tf.shape(ref_y[..., 1]), False)

                # run iou, if ref_y had any box. Otherwise, increment fp according to pred_classes.
                # Anyway, return the various stats counters
                tp, fp, fn, all_ref, errors = tf.cond(tf.shape(batch_ref_y)[0] != 0,
                                                      true_fn=lambda: self.do_iou(pred_bboxes, pred_classes, ref_y, tp,
                                                                                  fp, fn, all_ref,
                                                                                  ref_boxes_assigned, errors),
                                                      false_fn=lambda: (tp,
                                                                        tf.tensor_scatter_nd_add(fp,
                                                                                                 tf.expand_dims(tf.cast(
                                                                                                     pred_classes,
                                                                                                     tf.int32),
                                                                                                     axis=-1),
                                                                                                 tf.fill(tf.shape(
                                                                                                     pred_classes)[0],
                                                                                                         1)),
                                                                        fn, all_ref)

                                                      )  # rone todo fix inc_arg
                images_cnt = tf.math.add(images_cnt, 1)

                print(images_cnt.numpy(), tp.numpy(), fp.numpy(), fn.numpy(), all_ref.numpy(), errors.numpy())
        # Resultant Stats:
        recall = tf.cast(tp, tf.float32) / (tf.cast(tp + fn, tf.float32) + 1e-20)
        precision = tf.cast(tp, tf.float32) / (tf.cast(tp + fp, tf.float32) + 1e-20)
        print(f'recall: {recall}, precision: {precision}')


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
        evaluate.calc_ap(tfrecords_dir, image_size, batch_size, classes_name_file, model_config_file,
                         input_weights_path,
                         anchors_table,
                         nms_iou_threshold, nms_score_threshold, yolo_max_boxes)


    main()
