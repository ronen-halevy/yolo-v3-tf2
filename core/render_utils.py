#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : render_utils.py
#   Author      : ronen halevy 
#   Created date:  7/5/22
#   Description :
#
# ================================================================
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

import tensorflow as tf


def render_bboxes(image, bboxes, colors):
    """
    :param image: target image
    :type image:  tf.float32, rank-4
    :param bboxes: bboxes tf boxes. Assumed xmin,ymin,xmax,ymax
    :type bboxes: tf.float32, rank-3
    :return:
    :rtype:
    """

    indices = [1, 0, 3, 2]  # [xmin,ymin, xmax,ymax]->[ymin, xmin,ymax, xmax]
    bboxes = tf.gather(bboxes, indices, axis=-1)
    image = tf.image.draw_bounding_boxes(
        image, bboxes, colors, name=None
    )
    return image


def draw_text_on_bounding_box(image, ymin, xmin, color, display_str_list=(), font_size=30):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", font_size)
    except IOError:
        print("Font not found, using default font.")
        font = ImageFont.load_default()

    text_heights = [font.getsize(string)[1] for string in display_str_list]
    text_margin_factor = 0.05
    total_text_height = (1 + 2 * text_margin_factor) * sum(text_heights)

    if ymin > total_text_height:
        text_bottom = ymin
    else:
        text_bottom = ymin + total_text_height

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        text_margin = np.ceil(text_margin_factor * text_height)
        draw.rectangle([(xmin, text_bottom - text_height - 2 * text_margin),
                        (xmin + text_width, text_bottom)],
                       fill=color)
        draw.text((xmin + text_margin, text_bottom - text_height - text_margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * text_margin
        return image


def annotate_text(image_pil, bbox, class_name, score, font_size=30):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    im_width, im_height = image_pil.size
    xmin, ymin, xmax, ymax = tuple(bbox * [im_width, im_height, im_width, im_height])
    colors = list(ImageColor.colormap.values())
    color = colors[hash(class_name) % len(colors)]

    detections_str = "{}: {}%".format(class_name,
                                      int(100 * score))
    ymin_text = tf.where(tf.greater(ymin, 0), ymin, font_size)
    xmin_text = tf.where(tf.greater(xmin, 0), xmin, 0)


    image_pil = draw_text_on_bounding_box(
        image_pil,
        ymin_text,
        xmin_text,
        color,
        display_str_list=[detections_str], font_size=font_size)

    return (detections_str, xmin, ymin, xmax, ymax)


def annotate_detections(image, class_names, bboxes, scores, bbox_color, font_size):
    annotated_image = Image.fromarray(np.uint8(image * 255)).convert("RGB")
    detections = []
    for idx, (bbox, class_name, score) in enumerate(zip(bboxes, class_names, scores)):
        detection = annotate_text(annotated_image, bbox, class_name, score, font_size)
        detections.append(detection)
    return annotated_image, detections


def render_text_annotated_bboxes(image, bboxes, classes_names, scores, bbox_color, font_size):  # , image_index,
    # image_output_dir, detections_list_outfile, display_result_images=False):
    rendered_image = render_bboxes(tf.expand_dims(image, axis=0), tf.expand_dims(bboxes, axis=0),
                                   colors=[bbox_color])
    rendered_image = tf.squeeze(rendered_image, axis=0)
    text_annotated_image, detections_string = annotate_detections(rendered_image, classes_names, bboxes,
                                                                  scores, bbox_color, font_size)

    return text_annotated_image, detections_string
