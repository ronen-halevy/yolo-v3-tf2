import numpy as np
import os
import tensorflow as tf
import yaml
import argparse
import matplotlib.pyplot as plt

from models import yolov3_model
from utils import get_anchors, resize_image
from render_utils import annotate_detections
from load_tfrecords import parse_tfrecords


class Inference:
    def __init__(self, output_dir, print_detections, save_result_images, display_result_images):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        try:
            self.detections_outfile = f'{output_dir}/detect.txt'
            os.remove(self.detections_outfile)
        except OSError:
            pass
        self.detections_list_outfile = open(self.detections_outfile, 'a')
        self.output_dir = output_dir
        self.print_detections = print_detections
        self.save_result_images = save_result_images
        self.display_result_images = display_result_images

    @staticmethod
    def do_inference(yolo, img_raw, size, class_names, yolo_max_boxes, bbox_color, font_size, index):
        img = tf.expand_dims(img_raw, 0)
        img = resize_image(img, size, size)
        boxes, scores, classes, nums = yolo(img, training=False)
        detected_classes = [class_names[idx] for idx in classes[0]]

        image_pil, num_annotated, num_score_skips = annotate_detections(img_raw, detected_classes, boxes[0], scores[0],
                                                                        yolo_max_boxes, bbox_color, font_size)
        image_detections_result = [f'#{index + 1} detected:{nums[0]}']
        for i in range(nums[0]):
            image_detections_result.append(
                f'{class_names[int(classes[0][i])]}, {np.array(scores[0][i])}, {np.array(boxes[0][i])}')

        return image_pil, image_detections_result

    def output_detections_text(self, image_detections_result):
        if self.print_detections:
            print(image_detections_result)

        self.detections_list_outfile.write(f'{image_detections_result}\n')
        self.detections_list_outfile.flush()

    def output_annotated_image(self, annotated_image, out_filename):
        outfile_path = f'{self.output_dir}/{out_filename}'
        if self.save_result_images:
            annotated_image.save(outfile_path)
        if self.display_result_images:
            plt.imshow(annotated_image)
            plt.show()


def detect(
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
    anchors_table = get_anchors(anchors_file)
    class_names = [c.strip() for c in open(classes).readlines()]
    nclasses = len(class_names)

    yolo = yolov3_model(anchors_table, size, nclasses=nclasses, training=False,
                        yolo_max_boxes=yolo_max_boxes, nms_iou_threshold=nms_iou_threshold,
                        nms_score_threshold=nms_score_threshold)

    yolo.load_weights(weights).expect_partial()
    print('weights loaded')

    imagefile_ext = ('.jpg', '.png')
    inferecnce = Inference(output_dir, print_detections, save_result_images, display_result_images)

    if input_data_source == 'tfrecords':
        dataset = parse_tfrecords(tfrecords_dir, image_size=size, max_bboxes=yolo_max_boxes, class_file=None)
        for index, entry in enumerate(dataset):
            annotated_image, image_detections_result = inferecnce.do_inference(yolo, entry[0], size, class_names,
                                                                               yolo_max_boxes, font_size,
                                                                               index)
            inferecnce.output_detections_text(image_detections_result)
            inferecnce.output_annotated_image(annotated_image, out_filename=f'detect_{index}.jpg')

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

            files = []
            for ext in ('*.jpeg', '*.jpg', '*.png', '*.bmp'
                        ):
                filenames.extend(glob(join(images_dir, ext)))


        else:
            raise Exception(f'input_data_source {input_data_source} not valid')

        for index, file in enumerate(filenames):
            img_raw = tf.image.decode_image(open(file, 'rb').read(), channels=3)
            annotated_image, image_detections_result = inferecnce.do_inference(yolo, img_raw / 255, size, class_names,
                                                                               yolo_max_boxes, bbox_color, font_size,
                                                                               index)
            inferecnce.output_detections_text(image_detections_result)
            inferecnce.output_annotated_image(annotated_image, out_filename=f'detect_{index}.jpg')


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config/detect_config_coco.yaml',
                    help='yaml config file')

args = parser.parse_args()
config_file = args.config
with open(config_file, 'r') as stream:
    detect_config = yaml.safe_load(stream)

detect(**detect_config)
