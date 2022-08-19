import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import Input, Model

import yaml
import argparse
import matplotlib.pyplot as plt

from core.utils import get_anchors, resize_image
from core.render_utils import annotate_detections
from core.load_tfrecords import parse_tfrecords
from core.parse_model import ParseModel

from core.yolo_decode_layer import YoloDecoderLayer
from core.yolo_nms_layer import YoloNmsLayer

from core.exceptions import NoDetectionsFound


class Inference:
    @staticmethod
    def _do_inference(model, img_raw, image_size, class_names):
        img = tf.expand_dims(img_raw, 0)
        img = resize_image(img, image_size, image_size)
        boxes, scores, classes, num_of_valid_detections = model.predict(img)

        if tf.equal(num_of_valid_detections, 0):
            raise NoDetectionsFound
        detected_classes = [class_names[idx] for idx in classes[0]]
        return boxes, scores, detected_classes, num_of_valid_detections

    @staticmethod
    def display_detections(img_raw, detected_classes, boxes, scores, yolo_max_boxes, bbox_color, font_size):
        annotated_image, detections = annotate_detections(img_raw, detected_classes, boxes[0], scores[0],
                                                          yolo_max_boxes, bbox_color, font_size)
        plt.imshow(annotated_image)
        plt.show()
        return annotated_image, detections

    @staticmethod
    def _dump_detections_text(image_detections_result, detections_list_outfile):
        detections_list_outfile.write(f'{image_detections_result}\n')
        detections_list_outfile.flush()

    def _inference(self, model, image, image_size, yolo_max_boxes, bbox_color, font_size, class_names, image_id,
                   output_dir, detections_list_outfile):
        try:
            boxes, scores, detected_classes, num_of_valid_detections = self._do_inference(model, image, image_size,
                                                                                          class_names)
            annotated_image, detections_str = self.display_detections(image, detected_classes, boxes, scores,
                                                                      yolo_max_boxes, bbox_color, font_size)
        except NoDetectionsFound:
            detections_str = (f'No detections found. Index = {image_id}')
            import PIL
            annotated_image = PIL.Image.fromarray(np.uint8(image * 255)).convert("RGB")
        self._dump_detections_text(detections_str, detections_list_outfile)
        outfile_path = f'{output_dir}/detect_{image_id}.jpg'
        annotated_image.save(outfile_path)

    @staticmethod
    def _output_annotated_image(annotated_image, output_dir, out_filename, save_result_images, display_result_images):
        outfile_path = f'{output_dir}/{out_filename}'
        if save_result_images:
            annotated_image.save(outfile_path)

    def __call__(self,
                 model_config_file,
                 classes,
                 anchors_file,
                 weights,
                 image_size,
                 input_data_source,
                 images_dir,
                 tfrecords_dir,
                 image_file_path,
                 output_dir,
                 yolo_max_boxes,
                 nms_iou_threshold,
                 nms_score_threshold,
                 display_result_images,
                 save_result_images,
                 bbox_color,
                 font_size
                 ):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        detections_text_list_outfile = f'{output_dir}/detect.txt'
        try:
            os.remove(detections_text_list_outfile)
        except OSError as e:
            pass

        detections_list_outfile = open(detections_text_list_outfile, 'a')

        anchors_table = tf.cast(get_anchors(anchors_file), tf.float32)
        class_names = [c.strip() for c in open(classes).readlines()]
        nclasses = len(class_names)

        inputs = Input(shape=(None, None, 3))

        parse_model = ParseModel()

        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)
        model = parse_model.build_model(inputs, nclasses, **model_config)
        model.load_weights(weights)
        print('weights loaded')
        model = model(inputs)

        decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
        nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                                  nms_score_threshold)(decoded_output)
        model = Model(inputs, nms_output, name="yolo_nms")

        if input_data_source == 'tfrecords':
            dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes, class_file=None)
            for index, dataset_entry in enumerate(dataset):  # todo consider batch inference
                image = dataset_entry[0]
                self._inference(model, image, image_size, yolo_max_boxes, bbox_color, font_size, class_names, index,
                                output_dir, detections_list_outfile)
        else:
            if input_data_source == 'image_file':
                filenames = [image_file_path]
            elif input_data_source == 'images_dir':
                valid_images = ('.jpeg', '.jpg', '.png', '.bmp')
                valid_images = ('.jpeg', '.jpg', '.png', '.bmp')
                filenames = []
                for f in os.listdir(images_dir):
                    ext = os.path.splitext(f)[1]
                    if ext.lower() not in valid_images:
                        continue
                    filenames.append(f'{images_dir}/{f}')

            for index, file in enumerate(filenames):  # todo consider batch inference
                img_raw = tf.image.decode_image(open(file, 'rb').read(), channels=3, dtype=tf.float32)
                self._inference(model, img_raw, image_size, yolo_max_boxes, bbox_color, font_size, class_names, index,
                                output_dir, detections_list_outfile)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config/detect_config_coco.yaml',
                    help='yaml config file')

args = parser.parse_args()
config_file = args.config
with open(config_file, 'r') as stream:
    detect_config = yaml.safe_load(stream)

inference = Inference()
inference(**detect_config)
