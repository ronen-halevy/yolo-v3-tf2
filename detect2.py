import numpy as np
import os
import tensorflow as tf
import yaml
import argparse
import matplotlib.pyplot as plt

from core.models import YoloV3Model
from core.decode_detections import decode_detections
from core.utils import get_anchors, resize_image
from core.render_utils import annotate_detections
from core.load_tfrecords import parse_tfrecords



class Inference:

    @staticmethod
    def _do_detection(yolo, img_raw, size, class_names, bbox_color, font_size, index, yolo_max_boxes, anchors_table, nms_iou_threshold, nms_score_threshold):
        img = tf.expand_dims(img_raw, 0)
        img = resize_image(img, size, size)
        all_grids_output = yolo(img, training=False)
        nclasses=len(class_names)

        boxes, scores, classes, nums = decode_detections(all_grids_output, nclasses,
                   yolo_max_boxes, anchors_table, nms_iou_threshold, nms_score_threshold)

        detected_classes = [class_names[idx] for idx in classes[0]]

        image_pil, num_annotated, num_score_skips = annotate_detections(img_raw, detected_classes, boxes[0], scores[0],
                                                                        yolo_max_boxes, bbox_color, font_size)
        image_detections_result = [f'#{index + 1} detected:{nums[0]}']
        for i in range(nums[0]):
            image_detections_result.append(
                f'{class_names[int(classes[0][i])]}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}')

        return image_pil, image_detections_result

    @staticmethod
    def _dump_detections_text(image_detections_result, print_detections, detections_list_outfile):
        if print_detections:
            print(image_detections_result)

        detections_list_outfile.write(f'{image_detections_result}\n')
        detections_list_outfile.flush()

    @staticmethod
    def _output_annotated_image(annotated_image, output_dir, out_filename, save_result_images, display_result_images):
        outfile_path = f'{output_dir}/{out_filename}'
        if save_result_images:
            annotated_image.save(outfile_path)
        if display_result_images:
            plt.imshow(annotated_image)
            plt.show()

    def __call__(self,
                 classes,
                 anchors_file,
                 weights,
                 size,
                 input_data_source,
                 images_dir,
                 tfrecords_dir,
                 image_file_path,
                 output_dir,
                 yolo_max_boxes,
                 nms_iou_threshold,
                 nms_score_threshold,
                 display_result_images,
                 print_detections,
                 save_result_images,
                 bbox_color,
                 font_size
                 ):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        detections_outfile = f'{output_dir}/detect.txt'
        try:
            os.remove(detections_outfile)
        except OSError as e:
            pass

        detections_list_outfile = open(detections_outfile, 'a')

        anchors_table = tf.cast(get_anchors(anchors_file), tf.float32)
        class_names = [c.strip() for c in open(classes).readlines()]
        nclasses = len(class_names)

        yolov3_model = YoloV3Model()

        model = yolov3_model(size, nclasses)

        model.load_weights(weights).expect_partial()
        print('weights loaded')

        if input_data_source == 'tfrecords':
            dataset = parse_tfrecords(tfrecords_dir, image_size=size, max_bboxes=yolo_max_boxes, class_file=None)
            for index, entry in enumerate(dataset):
                annotated_image, image_detections_result = self._do_detection(model, entry[0], size, class_names,
                                                                              yolo_max_boxes, bbox_color, font_size,
                                                                              index)
                self._dump_detections_text(image_detections_result, print_detections, detections_list_outfile)
                out_filename = f'detect_{index}.jpg'
                self._output_annotated_image(annotated_image, output_dir, out_filename, save_result_images,
                                             display_result_images)
        else:
            if input_data_source == 'image_file':
                filenames = [image_file_path]
            elif input_data_source == 'images_dir':
                # dirname = images_dir

                import glob
                types = ('*.jpeg', '*.jpg', '*.png', '*.bmp')  # the tuple of file types
                filenames = []
                for files in types:
                    filenames.extend(glob.glob(files))

                from os.path import join
                from glob import glob

                for ext in ('*.jpeg', '*.jpg', '*.png', '*.bmp'
                            ):
                    filenames.extend(glob(join(images_dir, ext)))
            else:
                raise Exception(f'input_data_source {input_data_source} not valid')

            for index, file in enumerate(filenames):
                img_raw = tf.image.decode_image(open(file, 'rb').read(), channels=3)


                annotated_image, image_detections_result = self._do_detection(model, img_raw / 255, size, class_names,
                                                                              bbox_color, font_size,
                                                                              index, yolo_max_boxes, anchors_table, nms_iou_threshold, nms_score_threshold)
                self._dump_detections_text(image_detections_result, print_detections, detections_list_outfile)
                out_filename = f'detect_{index}.jpg'
                self._output_annotated_image(annotated_image, output_dir, out_filename, save_result_images,
                                             display_result_images)


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config/detect_config_coco.yaml',
                    help='yaml config file')

args = parser.parse_args()
config_file = args.config
with open(config_file, 'r') as stream:
    detect_config = yaml.safe_load(stream)

inference = Inference()
inference(**detect_config)
