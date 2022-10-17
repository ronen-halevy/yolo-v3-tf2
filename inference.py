import os
import tensorflow as tf
from tensorflow.keras import Input, Model

import yaml
import argparse
import matplotlib.pyplot as plt

from core.utils import get_anchors, resize_image, dir_filelist
from core.render_utils import annotate_detections, render_text_annotated_bboxes

from core.load_tfrecords import parse_tfrecords
from core.parse_model import ParseModel

from core.yolo_decode_layer import YoloDecoderLayer
from core.yolo_nms_layer import YoloNmsLayer


class Inference:

    @staticmethod
    def gather_valid_detections_results(bboxes_padded, class_indices_padded, scores_padded,
                                        selected_indices_padded, num_valid_detections):

        bboxes = tf.gather(bboxes_padded, selected_indices_padded[:num_valid_detections], axis=0)
        classes = tf.gather(class_indices_padded, selected_indices_padded[:num_valid_detections], axis=0)
        scores = tf.gather(scores_padded, selected_indices_padded[:num_valid_detections], axis=0)
        return bboxes, classes, scores

    @staticmethod
    def display_detections(img_raw, detected_classes, boxes, scores, bbox_color, font_size):
        annotated_image, detections = annotate_detections(img_raw, detected_classes, boxes[0], scores[0],
                                                          bbox_color, font_size)
        plt.imshow(annotated_image)
        plt.show()
        return annotated_image, detections

    @staticmethod
    def _dump_detections_text(image_detections_result, detections_list_outfile):
        detections_list_outfile.write(f'{image_detections_result}\n')
        detections_list_outfile.flush()

    def results_display_and_save(self, display_result_images, text_annotated_image, detections_string, output_dir,
                                 detections_list_outfile, image_index):
        if display_result_images:
            plt.imshow(text_annotated_image)
            plt.show()
        self._dump_detections_text(detections_string, detections_list_outfile)
        outfile_path = f'{output_dir}/detect_{image_index}.jpg'
        text_annotated_image.save(outfile_path)

    def __call__(self,
                 model_config_file,
                 classes_name_file,
                 anchors_file,
                 input_weights_path,
                 image_size,
                 input_data_source,
                 images_dir,
                 tfrecords_dir,
                 batch_size,
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
        class_names = [c.strip() for c in open(classes_name_file).readlines()]
        nclasses = len(class_names)

        inputs = Input(shape=(None, None, 3))

        parse_model = ParseModel()

        with open(model_config_file, 'r') as _stream:
            model_config = yaml.safe_load(_stream)

        sub_models_configs = model_config['sub_models_configs']
        output_stage = model_config['output_stage']
        model = parse_model.build_model(inputs, sub_models_configs, output_stage, nclasses=nclasses)
        # for debug:
        # with open("model_inference_summary.txt", "w") as file1:
        #     model.summary(print_fn=lambda x: file1.write(x + '\n'))

        # Note:.expect_partial() prevents warnings at exit time, since save model generates extra keys.
        model.load_weights(input_weights_path).expect_partial()
        print('weights loaded')

        model = model(inputs)

        decoded_output = YoloDecoderLayer(nclasses, anchors_table)(model)
        nms_output = YoloNmsLayer(yolo_max_boxes, nms_iou_threshold,
                                  nms_score_threshold)(decoded_output)
        model = Model(inputs, nms_output, name="yolo_nms")

        if input_data_source == 'tfrecords':

            dataset = parse_tfrecords(tfrecords_dir, image_size=image_size, max_bboxes=yolo_max_boxes, class_file=None)
            dataset = dataset.batch(batch_size)
            dataset_images = dataset.map(lambda img, _: resize_image(img, image_size, image_size))

            for batch_dataset_image in dataset_images:
                batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
                batch_num_valid_detections = model.predict(
                    batch_dataset_image)

                for image_index, \
                    (bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded, num_valid_detections,
                     image) \
                        in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                                         batch_selected_indices_padded, batch_num_valid_detections,
                                         batch_dataset_image)):
                    bboxes, classes, scores = self.gather_valid_detections_results(bboxes_padded, class_indices_padded,
                                                                                   scores_padded,
                                                                                   selected_indices_padded,
                                                                                   num_valid_detections)
                    classes_names = [class_names[idx] for idx in classes]
                    text_annotated_image, detections_string = render_text_annotated_bboxes(image, bboxes,
                                                                                           classes_names, scores,
                                                                                           bbox_color, font_size)
                    self.results_display_and_save(display_result_images, text_annotated_image, detections_string,
                                                  output_dir,
                                                  detections_list_outfile, image_index)

        else:
            if input_data_source == 'image_file':
                filenames = [image_file_path]
            elif input_data_source == 'images_dir':
                filenames = dir_filelist(images_dir, ('.jpeg', '.jpg', '.png', '.bmp'))
            else:
                filenames = []

            for image_index, file in enumerate(filenames):
                orig_image = tf.image.decode_image(open(file, 'rb').read(), channels=3, dtype=tf.float32)
                image = tf.image.resize(orig_image, (image_size, image_size))

                batch_dataset_image = tf.expand_dims(image, axis=0)
                batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded, batch_selected_indices_padded, \
                batch_num_valid_detections = model.predict(
                    tf.expand_dims(image, axis=0))
                for index, (
                        bboxes_padded, class_indices_padded, scores_padded, selected_indices_padded,
                        num_valid_detections, image) \
                        in enumerate(zip(batch_bboxes_padded, batch_class_indices_padded, batch_scores_padded,
                                         batch_selected_indices_padded, batch_num_valid_detections,
                                         batch_dataset_image)):
                    bboxes, classes, scores = self.gather_valid_detections_results(bboxes_padded, class_indices_padded,
                                                                                   scores_padded,
                                                                                   selected_indices_padded,
                                                                                   num_valid_detections)
                    classes_names = [class_names[idx] for idx in classes]

                    text_annotated_image, detections_string = render_text_annotated_bboxes(image, bboxes,
                                                                                           classes_names, scores,
                                                                                           bbox_color, font_size)

                    text_annotated_image = text_annotated_image.resize((orig_image.shape[1], orig_image.shape[0]))

                    self.results_display_and_save(display_result_images, text_annotated_image, detections_string,
                                                  output_dir,
                                                  detections_list_outfile, image_index)
        for class_name, bbox, score in zip(classes_names, bboxes.numpy(), scores.numpy()):
            print(f'{class_name} bbox: {bbox} score: {score}')

        return


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config/detect_config.yaml',
                    help='yaml config file')

args = parser.parse_args()
config_file = args.config
with open(config_file, 'r') as stream:
    detect_config = yaml.safe_load(stream)

inference = Inference()
inference(**detect_config)
