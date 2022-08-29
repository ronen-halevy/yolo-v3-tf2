#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : evaluate2.py
#   Author      : ronen halevy 
#   Created date:  8/24/22
#   Description :
#
# ================================================================
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


from core.exceptions import NoDetectionsFound

# def _iou(box_1, box_2):
#     # box_1: (..., (x1, y1, x2, y2))
#     # box_2: (N, (x1, y1, x2, y2))
#
#     # broadcast boxes
#     # box_1 = tf.expand_dims(box_1, -2)
#     # box_2 = tf.expand_dims(box_2, 0)
#     # # new_shape: (..., N, (x1, y1, x2, y2))
#     # new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
#     # box_1 = tf.broadcast_to(box_1, new_shape)
#     # box_2 = tf.broadcast_to(box_2, new_shape)
#
#     int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
#                        tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
#     int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
#                        tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
#     int_area = int_w * int_h
#     box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
#         (box_1[..., 3] - box_1[..., 1])
#     box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
#         (box_2[..., 3] - box_2[..., 1])
#     return int_area / (box_1_area + box_2_area - int_area)

########################
# # 2. transform all true outputs
# # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
# # true_box, true_obj, true_class_idx = tf.split(
# #     y_true, (4, 1, 1), axis=-1)
# true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
# true_wh = true_box[..., 2:4] - true_box[..., 0:2]
#
# # give higher weights to small boxes
# box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]
#
# # 3. inverting the pred box equations
# grid_size = tf.shape(y_true)[1]
# grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
# grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
# true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
#           tf.cast(grid, tf.float32)
# true_wh = tf.math.log(true_wh / anchors)
# true_wh = tf.where(tf.math.is_inf(true_wh),
#                    tf.zeros_like(true_wh), true_wh)
#
# # 4. calculate all masks
# obj_mask = tf.squeeze(true_obj, -1)
# # ignore false positive when iou is over threshold
# best_iou = tf.map_fn(
#     lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
#         x[1], tf.cast(x[2], tf.bool))), axis=-1),
#     (pred_box, true_box, obj_mask),
#     tf.float32)
# ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
#
#
#
# #####################
class Evaluate:
    def __init__(self, nclasses, iou_thresh):
        self.nclasses = nclasses
        self.recall= [np.array([0]) for _ in range(nclasses)]
        self.precision= [np.array([0]) for _ in range(nclasses)]
        self.true_class_counts = np.zeros(nclasses)
        self.iou_thresh = iou_thresh

    @staticmethod
    def broadcast_iou(box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        # box_1 = tf.expand_dims(box_1, -2)
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

    def mask_padded_entries(self, true_y):
        masked_true_y = true_y[true_y[..., 4] != 0]
        return masked_true_y

    # @tf.function
    def calc_iou(self, pred_y, true_y):


        # bboxes, confidences, classes_indices, valid_count = pred_y

        t_bboxes, t_confidences, t_classes_indices = tf.split(true_y, [4,1,1], axis=-1)
        p_bboxes, p_classes_indices = tf.split(tf.squeeze(pred_y, axis=0), [4,1], axis=-1)

        # return pred_y, true_y

        # t_assigned_to_pred = np.zeros(tf.shape(t_confidences).numpy())
        # tp = np.zeros(confidences[0].shape[0])
        # fp = np.zeros(confidences[0].shape[0])
        # tp = np.array([0])
        tp = 0 #np.zeros([0])
        fp = 0 #np.zeros([0])
        fn = 0


        # tp = np.array([0])
        # fp = np.array([0])
        t_con = tf.fill
        zero_init = tf.fill(tf.shape(t_confidences), tf.constant(0))

        # t_assigned_to_pred = tf.Variable(tf.shape(t_confidences)[0], shape=[2,2])
        # t_assigned_to_pred = np.ndarray(tf.shape(t_confidences)[0], dtype=bool)
        # t_assigned_to_pred.fill(False)
        # False for idx in range(tf.shape(t_bboxes)[0])]
        for idx, (bbox, class_index)  in enumerate(zip(p_bboxes, p_classes_indices)):

            if tf.shape(t_bboxes)[0] != 0:
                iou = self.broadcast_iou(bbox, t_bboxes)
            else:
                fp += 1
                continue

            # iou = tf.squeeze(iou, axis=0)
        #
        #
            best_iou_index = tf.math.argmax(
                    iou, axis=-1, output_type=tf.int32).numpy()[0]
            # a = t_classes_indices[best_iou_index] == class_index
            # b =  t_assigned_to_pred[best_iou_index]
            # c = iou[...,best_iou_index] > self.iou_thresh
            if iou[...,best_iou_index] > self.iou_thresh and t_classes_indices[best_iou_index] == class_index: # and not t_assigned_to_pred[best_iou_index]:
                # t_assigned_to_pred[best_iou_index]=True

                tp+=1
        #             fp[class_index] = np.append(fp[class_index], 0)
            else:
                fp+=1
        # for status in t_assigned_to_pred:
        #     if not status:
        #         fn+=1

        return tp, fp, fn

        #             tp[class_index] = np.append(tp[class_index], 0)
        #             fp[class_index] = np.append(fp[class_index], 1)
        #     pass
        # return tp, fp


    # def calc_recal_precision(self, tp, fp):
    #     tmp1 =  np.pad(tp, [1, 0])
    #     tmp2 = np.pad(tp, [2, 0])
    #     prev = self.recall[...,-1]
    #     tp = tp + tmp1 + tmp2 + prev
    #     self.rec = self.rec + tmp1[:-1] + tmp2[:-2]

    def calc_recal_precision(self, tp, fp):
        for class_idx,  tp_ in enumerate(tp):
            if tp_.shape[0] == 1:
                continue
            tmp1 = tp_
            tmp2 = np.pad(tp_, [1, 0])
            prev = self.recall[class_idx][-1]
            try:
                self.recall[class_idx] = np.append(self.recall[class_idx][1:], tp_[1:] + tmp1[:-1] + tmp2[:-2] + prev)
            except Exception as e:
                print(e)
                exit(1)


        for class_idx, fp_ in enumerate(fp[1:]):
            if fp_.shape[0] == 1:
                continue
            tmp1 = fp_
            tmp2 = np.pad(fp_, [1, 0])
            prev = self.precision[class_idx][-1]
            try:
                self.precision[class_idx] = np.append(self.precision[class_idx], fp_[1:] + tmp1[:-1] + tmp2[:-2] + prev)
            except Exception as e:
                print(e)
                exit(1)


            return self.recall,  self.precision



        tmp1 =  np.pad(tp, [1, 0])
        pass

    # @staticmethod
    # def prepare_true_boxes(true_y):
    #     true_score_mask = true_y[..., 4] != 0
    #     true_y = true_y[true_score_mask]
    #     return true_y

    @staticmethod
    def inference_func(true_image, model):
        img = tf.expand_dims(true_image, 0)
        img = resize_image(img, image_size, image_size)
        output = model(img)
        return output#, true_y





    def calc_ap(self, gt_classes, tfrecords_dir, evaluate_iou_threshold, image_size, classes_name_file, model_config_file, input_weights_path, anchors_table,
                nms_iou_threshold, nms_score_threshold, yolo_max_boxes=100):


        #
        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)
        nclasses = 7

        parse_model = ParseModel()
        inputs = Input(shape=(None, None, 3))


        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)
        model = parse_model.build_model(inputs, nclasses=nclasses, **model_config)



        model.load_weights(input_weights_path)
        print('weights loaded')
        model = model(inputs)

        decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
        nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                                  nms_score_threshold)(decoded_output)

        model = Model(inputs, nms_output, name="yolo_nms")
        #



        dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes, class_file=classes_name_file)
        dataset = dataset.take(30) # ronen todo

        dataset = dataset.map(lambda x, y: (x, y[y[..., 4] == 1]))

        dataset = dataset.map(lambda x, y: (self.inference_func(x, model), y))
        # dataset = dataset.map(lambda x, y: (x, y[y[..., 5] == 4]))
        for class_id in range(nclasses):
            data = dataset.map(lambda x, y: (x, y[y[..., 5] == class_id]))
            # if class_id == 6:
                #     # dd = list(data.as_numpy_iterator())

            data = data.map(lambda x, y: (tf.concat([x[0], tf.expand_dims(tf.cast(x[2], dtype=tf.float32), axis=-1)], axis=-1), y))

                    # data = data.map(lambda x, y:  self.broadcast_iou(x[...,:4], y[...,:4]))
                    # data = data.map(lambda x, y: self.broadcast_iou(x[..., :4], y[..., :4]))
            for dd in data:
                xx = self.calc_iou(dd[0], dd[1])
            # data = data.map(lambda x, y: self.calc_iou(x, y))
                # d=tf.math.reduce_sum(data, axis=0)

            # ss = list(map(lambda x: self.calc_iou(x[0], x[1]), data))
            # ss = map(lambda x: self.calc_iou(x[0], x[1]), data)

            # d=tf.math.reduce_sum(ss, axis=0)

            # dd = list(data.as_numpy_iterator())

            pass

            # dd = list(data.as_numpy_iterator())
            pass

            # tp_fp = list(map(lambda x: self.calc_iou(x[0], x[1]), data))



        # data = dataset.map(lambda x, y: (x, y[y[..., 5] == 4]))
        # data = [dataset.map(lambda x, y: (x, y[y[..., 5] == tf.constant(class_id)])) for class_id in nclasses]

        # dd = list(dataset.as_numpy_iterator())

        # dataset = dataset.map(lambda x, y: broadcast_iou(box_1, box_2))

        # broadcast_iou(box_1, box_2)
        # dataset = dataset.map(lambda x, y: self.calc_iou(x, y))


        # tp_fp =  list(map(lambda x:  self.calc_iou(x[0], x[1]), masked_data))



        # dataset = preprocess_dataset(dataset, batch_size, image_size, anchors_table, grid_sizes_table,
        #                           max_bboxes=yolo_max_boxes)

        # inference = Inference()
        gt_data = []
        # inf_result, true_val = dataset.map(lambda x, y: (inference_func(x, y), y))
        # gt_counter_per_class = {}
        #         image = dataset_entry[0]
        #         img = tf.expand_dims(image, 0)
        #         img = resize_image(img, image_size, image_size)
        #         bboxes, scores, classes, num_of_valid_detections = model.predict(img)
        # bboxes, scores, classes, num_of_valid_detections\

        # tf.boolean_mask(tensor, mask)
        # dataset = dataset.filter(lambda x, y: y[4] == 1)
        # masked_data = list(map(lambda x:  (x[0], x[1][x[1][..., 4] != 0]), dataset))
        #
        #
        # data =  list(map(lambda x:  (self.inference_func(x[0], model), x[1])
        #                  ,masked_data))


        #
        #
        # tp_fp =  list(map(lambda x:  self.calc_iou(x[0], x[1]), masked_data))
        # tp_fp =  list(map(lambda x:  self.calc_recal_precision(x[0], x[1]), tp_fp))


        # tp_fp =  list(map(lambda x:  self.calc_recal_precision(x[0], x[1]), tp_fp))


        # tp = np.take(tp_fp[0][0],  list(range(1,tp_fp[0][0].shape[0]+1)))
        # p = np.take(tp_fp[0][0],  list(range(1,tp_fp[0][0].shape[0]+1)))
        # skip the redundant 1st element of each image result
        # tp = tp_fp[0][0]
        # fp = tp_fp[0][1]
        # for entry in tp_fp:
        #     tp=np.append(tp, entry[0])
        #     fp=np.append(fp, entry[1])
        # tmp = tp
        # for idx in len(tp):
        #     tmp = np.pad(tmp, [1, 0])
        #     tp = tp + tmp[:-1]

        # fp= tpfp[-1][0]
        # tp= tpfp[-1][1]


        # hist =  list(map(lambda x:  tf.histogram_fixed_width(x[1][:,5], value_range=[0, nclasses], nbins=nclasses), masked_data))
        # hist = np.sum(np.array(hist), axis=0)






        # print(hist)
        # pass
        # def count_objects_per_class(self, t_classes_index, nclasses):
        #     value_range = [0.0, nclasses]
        #
        #     hist = tf.histogram_fixed_width(t_classes_index, value_range, nbins=nclasses)
        #     return hist

        # pass
        # print(data)

        # y =  list(map(lambda t:  calc_iou(t), x))

        # for index, (true_image, true_y) in enumerate(dataset):
        #     (bboxes, scores, classes, num_of_valid_detections), (true_bboxes, true_scores, true_classes) = inference_func(true_image, true_y, model)
        #     pass
        #     print(index)
        #     classes = np.expand_dims(classes, axis=-1)
        #     inf_out = np.concatenate([bboxes, classes], axis=-1)
        #     aa = list(map(calc_iou,inf_out[0]))
        #     print('ffff')




            # for idx, (bbox, score, class_idx) in enumerate(zip(bboxes, scores, classes)):


            # rr = tf.map_fn(fn=lambda t: calc_iou(t), elems=true_y)

            # inference_func(true_image, model)
            # inference_func(image, y)

            # pass# todo consider batch inference
    #
    #         inf_result, true_val = dataset.map(lambda x, y: (inference_func(x, y),
    #             y))
    #
    #         image = dataset_entry[0]
    #         img = tf.expand_dims(image, 0)
    #         img = resize_image(img, image_size, image_size)
    #         bboxes, scores, classes, num_of_valid_detections = model.predict(img)
    #         is_used = np.array([np.array([len(bboxes)])])
    #         is_used.fill(False)
    #
    #
    #
    #         true_bboxes, true_scores, true_classes = tf.split(
    #             dataset_entry[0], (4, 1, 1), axis=-1)
    #         count_true_positives = 0
    #
    #         gt_classes = []
    #         tp = np.zeros(np.array([len(bboxes)]))
    #         fp = np.zeros(np.array([len(bboxes)]))
    #
    #         for class_index, class_name in enumerate(gt_classes):
    #
    #             # count_true_positives[class_name] = 0
    #
    #             for idx, (bbox, score, classs) in enumerate(zip(bboxes, scores, classes)):
    #                 for true_idx, (true_bbox, true_score, true_class) in enumerate(zip(true_bboxes, true_scores, true_classes)):
    #                     if true_class == class_name:
    #                         iou = calc_iou(true_bbox, bbox)
    #                         if iou > iou_max:
    #                             iou_max = iou
    #                             true_bbox_sel, true_score_sel, true_class_sel =  true_bbox, true_score, true_class
    #             if iou_max >= iou_thresh:
    #                 if not is_used[true_idx]:
    #                     is_used[true_idx] = False
    #                     tp[idx] = 1
    #                     count_true_positives += 1
    #                 else:
    #                     fp[idx] = 1
    # #####
    #                     # print(tp)
    #                     # compute precision/recall
    #             cumsum = 0
    #             for idx, val in enumerate(fp):
    #                 fp[idx] += cumsum
    #                 cumsum += val
    #             cumsum = 0
    #             for idx, val in enumerate(tp):
    #                 tp[idx] += cumsum
    #                 cumsum += val
    #                 # print(tp)
    #             rec = tp[:]
    #             for idx, val in enumerate(tp):
    #                 rec[idx] = float(tp[idx]) / gt_counter_per_class
    #                 # print(rec)
    #             prec = tp[:]
    #             for idx, val in enumerate(tp):
    #                 prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    #                 # print(prec)
    #
    #                 ap, mrec, mprec = voc_ap(rec[:], prec[:])
    #                 sum_AP += ap
    #                 text = "{0:.2f}%".format(
    #                     ap * 100) + " = " + class_name + " AP "  # class_name + " AP = {0:.2f}%".format(ap*100)
    #                 """
    #                  Write to output.txt
    #                 """
    #                 rounded_prec = ['%.2f' % elem for elem in prec]
    #                 rounded_rec = ['%.2f' % elem for elem in rec]
    #                 output_file.write(
    #                     text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
    #                 if not args.quiet:
    #                     print(text)
    #                 ap_dictionary[class_name] = ap
    #
    #                 n_images = counter_images_per_class[class_name]
    #                 lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
    #                 lamr_dictionary[class_name] = lamr
    #

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