#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : transorm-train-dataset.py
#   Author      : ronen halevy 
#   Created date:  4/27/22
#   Description :
#
# ================================================================

import tensorflow as tf
import numpy as np

def resize_image(image, t_w, t_h):
    _, s_h, s_w, _ = image.shape

    scale = min(t_w / s_w, t_h / s_h)

    n_w, n_h = int(scale * s_w), int(scale * s_h)

    image = tf.image.resize(
        image,
        [n_h, n_w],
    )

    scaled_image = tf.image.pad_to_bounding_box(
        image, (t_h - n_h) // 2, (t_w - n_w) // 2, t_h, t_w
    )
    # y_n = tf.squeeze(y_n, axis=0)
    return scaled_image

@tf.function
def arrange_in_gridold(y_train, anchors, image_size, ):
    """

    :param y_train: shape [batch, boxes, 5]
    :type y_train:
    :param anchors: shape [3, 3]
    :type anchors:
    :param image_size: Image assumed square
    :type image_size:
    :return:
    :rtype:
    """
    # out = tf.TensorArray(tf.float32, size=3, dynamic_size=False, infer_shape=False, clear_after_read=False)
    # batch_size = 32
    grid_size = image_size // batch_size
    index = 0
    # for grid_anchors in anchors:
    grid_anchors = tf.cast(anchors, tf.float32)
    anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(grid_anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], grid_anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], grid_anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    iou_max = tf.math.argmax(iou, axis=-1)
    iou_max = tf.expand_dims(iou_max, axis=-1)
    iou_max = tf.cast(iou_max, tf.int32)
    box_xy = (y_train[...,0:2] + y_train[...,2:4]) / 2

    # image_size = tf.cast(image_size, tf.float32)
    # grid_size = tf.cast(grid_size, tf.float32)
    box_xy = tf.cast(box_xy, tf.float32)

    grid_xy = tf.cast(box_xy // (image_size /  tf.cast(grid_size, tf.float32)), tf.int32)
    #
    indices_tmp = tf.concat([grid_xy, iou_max], axis=-1)
    # boxes_indices = np.indices([batch_size, y_train.shape[1]])
    tf.print(y_train.shape[1])
    boxes_indices = np.indices([indices_tmp.shape[0], indices_tmp.shape[1]])

    batch_box_indices = boxes_indices[0]
    batch_box_indices = np.expand_dims(batch_box_indices, axis=-1)
    indices = tf.concat([batch_box_indices, indices_tmp], axis=-1)
    # tf.print(indices)
    scale_dataset = tf.scatter_nd(
        indices, y_train, (y_train.shape[0], grid_size, grid_size, anchors.shape[0], 5)
    )
    # out = out.write(index, scale_dataset)
    tf.print(scale_dataset.shape)
    index = index + 1
    # grid_size *= 2
    # tf.print(out)
    return scale_dataset


# @tf.function
def arrange_in_grid(y_train, anchors, downsize_stride, output_shape, max_boxes):
    """

    :param y_train:
    :type y_train:
    :param anchors:
    :type anchors:
    :param downsize_stride:
    :type downsize_stride:
    :param output_shape:
    :type output_shape:
    :return:
    :rtype:
    """

    grid_anchors = tf.cast(anchors, tf.float32)
    anchor_area = grid_anchors[..., 0] * grid_anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(grid_anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], grid_anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], grid_anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)

    iou_max = tf.math.argmax(iou, axis=-1)
    iou_max = tf.expand_dims(iou_max, axis=-1)
    iou_max = tf.cast(iou_max, tf.int32)
    box_xy = (y_train[..., 0:2] + y_train[..., 2:4]) / 2
    box_xy = tf.cast(box_xy, tf.float32)

    grid_xy = tf.cast(box_xy // downsize_stride, tf.int32)

    indices_tmp = tf.concat([grid_xy, iou_max], axis=-1)

    # boxes_indices = np.indices(
    #     [
    #     batch_size, padded_num_of_boxes]
    # )


    boxes_indices = np.indices([32, 50])

    batch_box_indices = boxes_indices[0]





    batches =  y_train.shape[0]
    ##!!!! ronen should be number of boxes 950)
    # boxes = y_train.shape[-1]
    boxes = max_boxes
    batch_box_indices = tf.tile(tf.expand_dims(tf.range(batches), -1), [1, boxes])


    batch_box_indices = tf.expand_dims(batch_box_indices, axis=-1)
    batch_box_indices = tf.cast(batch_box_indices, tf.int32)
    indices = tf.concat([batch_box_indices, indices_tmp], axis=-1)
    scale_dataset = tf.scatter_nd(
        indices, y_train, output_shape
    )
    tf.print(scale_dataset.shape)

    return scale_dataset


def preprocess_dataset1(dataset, batch_size,image_size, anchors):

    y_train = list(dataset.as_numpy_iterator())[0][1]#.take(1)[1]
    y_train = tf.expand_dims(y_train, axis=0)

    y_train = tf.tile(y_train, [32, 1, 1])
    yy = arrange_in_grid(y_train, anchors, image_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (
        resize_image(x, image_size, image_size),
        arrange_in_grid(y, anchors, image_size)))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


    # downsize_strides = [32, 16, 8]
    # grid_sizes = [13, 26, 52]
    # anchors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    #
    # batch_size = 32
    # image_size = 416
def preprocess_dataset(dataset, batch_size,image_size, anchors, grid_sizes, max_boxes):
    dataset = dataset.batch(batch_size, drop_remainder=True)
    downsize_strides = image_size / grid_sizes
    dataset = dataset.map(lambda x, y: (
        resize_image(x, image_size, image_size),
        tuple([arrange_in_grid(y, tf.convert_to_tensor(anchor), grid_stride,
                               [batch_size, grid_size, grid_size, 3, 5], max_boxes) for anchor, grid_stride, grid_size in
               zip(anchors, downsize_strides, grid_sizes)])
    ))
    return dataset

if __name__ == '__main__':

    from utils import generate_random_dataset
    dataset = generate_random_dataset()
    # y_train =list(dataset.as_numpy_iterator())[0][1]
    # paddings = tf.constant( [[0, 50],[0,0]])
    # y_train = tf.pad(y_train, paddings)
    #
    # y_train = tf.expand_dims(y_train, axis=0)
    # y_train = tf.tile(y_train, [32,1,1])
    #
    # # tt = tf.Variable(4, dtype=tf.float32)
    # # out = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # # idx = 2
    # # out = out.write(idx, tt)
    # # ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    # # ta = ta.unstack([0., 1.])
    #
    # # for i in range(2, n):
    # #     # ta = ta.write(i, ta.read(i - 1) + ta.read(i - 2))
    # #     tt = tf.Variable(4, dtype=tf.float32)
    # #     ta = ta.write(i, tt)
    #
    # anchors = tf.convert_to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    #
    # image_size = 416
    # # out = tf.TensorArray(tf.float32, size=3, dynamic_size=False, infer_shape=False)
    #
    # scale_dataset = arrange_in_grid(y_train, anchors, image_size)
    # dd = scale_dataset.as_numpy_iterator()
    # pass
    # print('hhh')
    # tensors = [scale_dataset.read(i) for i in range(3)]
    # # print(tensors)
    # # # print(scale_dataset)
    # pass
    downsize_strides = [32, 16, 8]
    grid_sizes = [13, 26, 52]
    anchors = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    batch_size = 32
    image_size = 416

    dataset = dataset.batch(batch_size, drop_remainder=True)


    dataset = dataset.map(lambda x, y: (
        resize_image(x, image_size, image_size),
        tuple([arrange_in_grid(y, tf.convert_to_tensor(anchor), grid_stride,
                               [batch_size, grid_size, grid_size, 3, 5]) for anchor, grid_stride, grid_size in zip(anchors, downsize_strides, grid_sizes)])
    ))

    dd = list(dataset.as_numpy_iterator())

    pass

    image = dd[0][0][0]
    pass
    from matplotlib import pyplot as plt
    plt.imshow(image)
    plt.show()
    pass