import time
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import cv2
import numpy as np
import tensorflow as tf
# from yolov3_tf2.models2 import (
#     YoloV3, YoloV3Tiny
# )
from models import yolov3_model, yolo_boxes
from preprocess_dataset import resize_image
from utils import render_bboxes
# from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
# from yolov3_tf2.utils import draw_outputs, render_bboxes


# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
# flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
#                     'path to weights file')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# flags.DEFINE_integer('size', 416, 'resize images to')
# flags.DEFINE_string('image', './data/girl.png', 'path to input image')
# flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
# flags.DEFINE_string('output', './output.jpg', 'path to output image')
# flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

class FLAGS:
    classes= './datasets/coco.names'
    classes= './datasets/shapes/class.names'
    weights=  'checkpoints/yolov3_train_13.tf'
    weights=  'checkpoints/yolov3_train_30.tf'
    weights=  'checkpoints/yolov3_train_1000.tf'
    weights=  'checkpoints/yolov3_train_5.tf'
    # weights=  '/home/ronen/Downloads/yolov3_train_2.tf'
    # weights=  'checkpoints/yolov3_train_2.tf'
    # weights=  'checkpoints/yolov3_train_1.tf'
    weights=  'checkpoints/yolov3_train_120.tf'

    # weights = '/home/ronen/PycharmProjects/yolov3-tf2/checkpoints/yolov3_train_2.tf'

    tiny=  False
    size=  416
    image = './datasets/shapes/debug_dataset_sample/000001_triangle.jpg' #'./datasets/girl.png'
    tfrecord =  None
    output = './output.jpg', 'path to output image'
    num_classes = 7 # 80

def main():
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # for physical_device in physical_devices:
    #     tf.config.experimental.set_memory_growth(physical_device, True)
    anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
                                                               (59, 119)], [(116, 90), (156, 198), (373, 326)]],
                             np.float32) / 416
    anchors_table = np.array([[
        (0.08173, 0.04567),
        (0.08173, 0.08173),
        (0.08173, 0.15385)],
        [(0.15385, 0.08173),
         (0.15385, 0.15385),
         (0.25000, 0.12981)],
        [(0.15385, 0.29808),
         (0.25000, 0.25000),
         (0.25000, 0.49038)]])

    anchors_table = np.array([[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45),
                                                                    (59, 119)], [(10, 13), (16, 30), (33, 23)]],
                             np.float32) / 416

    # anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
    #                                                            (59, 119)], [(116, 90), (156, 198), (373, 326)]],
    #                          np.float32) / 416
    yolo_max_boxes = 100
    yolo_iou_threshold = 0.8
    yolo_score_threshold = 0.5

    yolo = yolov3_model(anchors_table, FLAGS.size, nclasses=FLAGS.num_classes, training=False, yolo_max_boxes=yolo_max_boxes, yolo_iou_threshold=yolo_iou_threshold, yolo_score_threshold=yolo_score_threshold)

    yolo.load_weights(FLAGS.weights).expect_partial()


    # yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
    #                          (59, 119), (116, 90), (156, 198), (373, 326)],
    #                         np.float32) / 416
    # yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    # yolo = YoloV3mm(size=None, channels=3, anchors=yolo_anchors,
    #              masks=yolo_anchor_masks, classes=80, training=False)


    # my_load_weights.load_weights(yolo, 'checkpoints/yolov3_train_1000.tf.index')

    # yolo.load_weights(FLAGS.weights).expect_partial()

    print('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    print('classes loaded')

    if FLAGS.tfrecord:
        pass
        # dataset = load_tfrecord_dataset(
        #     FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        # dataset = dataset.shuffle(512)
        # img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    # img = resize_image(img, FLAGS.size, FLAGS.size)
    img = tf.image.resize(img, (FLAGS.size, FLAGS.size))
    t1 = time.time()
    boxes, scores, classes, nums = yolo(img, training=True)

    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')

    render_bboxes( tf.cast(img_raw, tf.float32)/255, boxes)

    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                           np.array(scores[0][i]),
                                           np.array(boxes[0][i])))



main()
