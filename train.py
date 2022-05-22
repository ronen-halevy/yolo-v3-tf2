#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2022 . All rights reserved.
#
#   File name   : scratch.py
#   Author      : ronen halevy
#   Created date:  5/2/22
#   Description :
#
# ================================================================
import numpy as np
import tensorflow as tf
import argparse

from parse_tfrecords import parse_tfrecords

from preprocess_dataset import preprocess_dataset
from numpy import loadtxt

from utils import render_bboxes
from utils import generate_random_dataset, load_fake_dataset
from preprocess_dataset import preprocess_dataset
from loss_func import get_loss_func

def split_dataset(dataset, dataset_size):
    # the "or" condition is for debug small datasets
    train_size = int(0.7 * dataset_size) or dataset_size
    val_size = int(0.20 * dataset_size) or dataset_size
    test_size = int(0.10 * dataset_size) or dataset_size

    # dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=None)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    val_dataset = test_dataset.skip(test_size)
    test_dataset = test_dataset.take(test_size)

    return train_dataset, test_dataset, val_dataset


def insert_objectiveness_entry(lablels):
    (boxes, classes) = tf.split(lablels, [4, 1], axis=-1)
    objectiveness_entry = tf.cast(tf.fill(tf.shape(classes), 1), dtype=tf.float32)
    lablels = tf.concat([boxes, objectiveness_entry, classes], axis=-1)
    return lablels

def config_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_debug_dataset', default=True, action='store_true')

    parser.add_argument('--render_dataset_example', default=False, action='store_true')

    parser.add_argument("--tfrecords_dir", type=str,
                        default='/home/ronen/PycharmProjects/create-tfrecords/dataset/tfrecords',
                        help='path to tfrecords files')
    parser.add_argument("--limit", type=int, default=None,
                        help='limit on max input examples')

    parser.add_argument("--classes", type=str,
                        default='/home/ronen/PycharmProjects/shapes-dataset/dataset/class.names',
                        help='path to classes names file needed to annotate plotted objects')

    parser.add_argument("--max_boxes", type=int, default=100,
                        help='max bounding boxes in an example image')

    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch_size')

    parser.add_argument("--image_size", type=int, default=416,
                        help='Algorithm"s image_size . Assumed a square')

    parser.add_argument("--anchors_file", type=str, default='anchors/shapes_yolov3_anchors.txt',
                        help='anchors_file')

    parser.add_argument("--dataset_limit_size", type=str, default=None,
                        help='Train will limit dataset to dataset_size. If None, no limit is set')

    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help='learning_rate')

    grid_sizes_table = np.array([13, 26, 52])

    args = parser.parse_args()
    tfrecords_dir = args.tfrecords_dir
    image_size = args.image_size
    batch_size = args.batch_size
    class_file = args.classes
    max_boxes = args.max_boxes
    use_debug_dataset = args.use_debug_dataset
    anchors_file = args.anchors_file
    dataset_limit_size = args.dataset_limit_size
    anchors_file = args.anchors_file

    if use_debug_dataset:
        DATASET_REPEAT_COUNT = 1000
        dataset = load_fake_dataset()
        dataset = dataset.repeat()

        # dataset = dataset.repeat(DATASET_REPEAT_COUNT)
        # anchors_table = np.array(
        #     [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]],
        #     np.float32) / image_size

        anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
                                 (59, 119)], [(116, 90), (156, 198), (373, 326)]],
                                np.float32) / image_size
        nclasses = 80 # ronen TODO
    else:
        dataset = parse_tfrecords(tfrecords_dir, image_size, max_boxes, class_file)
        dataset = dataset.map(lambda x, y: (x, insert_objectiveness_entry(y)))

        # cc = next(dataset.as_numpy_iterator())
        anchors_table = loadtxt(anchors_file, comments="#", delimiter=",", unpack=False)
        nanchors_per_grid = 3
        anchors_table = anchors_table.reshape(-1, nanchors_per_grid, 2)
        nclasses = 80 # ronen TODO

    dataset_size = dataset_limit_size if dataset_limit_size else dataset.cardinality().numpy()

    if args.render_dataset_example:
        image, y = list(dataset.as_numpy_iterator())[0]
        images = tf.expand_dims(image, axis=0)  # * 255
        render_bboxes(images, y)

    # Descending anchors_table Order:
    anchors_table = np.flip(anchors_table, axis=[0])
    return dataset, dataset_size, batch_size, image_size, anchors_table, max_boxes, grid_sizes_table, nclasses


def train(learning_rate=0.001, mode='eager_fit'):
    train_dataset, dataset_size, batch_size, image_size, anchors_table, max_boxes, grid_sizes_table, nclasses = config_train()
    # train_dataset, test_dataset, val_dataset = split_dataset(dataset, dataset_size)



    # train_dataset = preprocess_dataset(train_dataset, batch_size, image_size, anchors_table, grid_sizes_table, max_boxes)
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / image_size

    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    import dataset
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, image_size),
        dataset.transform_targets(y, anchors, anchor_masks, image_size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)


    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    from models import yolov3_model

    from models import YoloV3




    yolo_model = yolov3_model(anchors_table, image_size, nclasses=nclasses)




    # yolo_model= YoloV3(image_size, classes=80, training=True)


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    loss_f = [get_loss_func(anchors, nclasses=nclasses)
            for anchors in anchors_table]
    #
    # loss_f = [YoloLoss(anchors, classes=80, ignore_thresh=0.5)
    #           for anchors in anchors_table]

    loss = [YoloLoss(anchors[mask], classes=nclasses)
            for mask in anchor_masks]


    # model.fit(dataset)
    # model.predict(dataset)


    # model = model.compile(optimizer=optimizer, loss=loss,
    #               run_eagerly=(FLAGS.mode == 'eager_fit'))
    # loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
    #         for mask in anchor_masks]

    epochs = 10
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    for epoch in range(1, epochs + 1):
        for batch, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                outputs = yolo_model(images)
                print(epoch, batch)
                regularization_loss = tf.reduce_sum(yolo_model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    #
            grads = tape.gradient(total_loss, yolo_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, yolo_model.trainable_variables))
            # from absl import logging
            import logging
            logging.basicConfig(level=logging.INFO,
                                format='%(levelname)s %(message)s',
                                )

            logging.info(f'epoch: {epoch} batch: {batch} total_loss: {total_loss.numpy()} ')
            for index, grid_loss in enumerate(pred_loss):
                logging.info(f' pred_loss grid {index} xy,wh,obj,class: {list(map(lambda x: np.sum(x.numpy()), grid_loss))}')

            avg_loss.update_state(total_loss)
    #
    #     for batch, (images, labels) in enumerate(val_dataset):
    #         outputs = model(images)
    #         regularization_loss = tf.reduce_sum(model.losses)
    #         pred_loss = []
    #         for output, label, loss_fn in zip(outputs, labels, loss):
    #             pred_loss.append(loss_fn(label, output))
    #         total_loss = tf.reduce_sum(pred_loss) + regularization_loss
    #
    #         logging.info("{}_val_{}, {}, {}".format(
    #             epoch, batch, total_loss.numpy(),
    #             list(map(lambda x: np.sum(x.numpy()), pred_loss))))
    #         avg_val_loss.update_state(total_loss)
    #
    #     logging.info("{}, train: {}, val: {}".format(
    #         epoch,
    #         avg_loss.result().numpy(),
    #         avg_val_loss.result().numpy()))
    #
    #     avg_loss.reset_states()
    #     avg_val_loss.reset_states()
        yolo_model.save_weights(
            'checkpoints/yolov3_train_{}.tf'.format(epoch))


def setup_model():

    from models import YoloV3
    image_size = 416
    num_classes = 80
    learning_rate = 0.001
    mode = 'eager_tf'
    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / image_size

    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    # anchors = yolo_anchors
    # anchor_masks = yolo_anchor_masks

    # model = YoloV3(416, training=True, classes=80)
    from models import yolov3_model

    anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
                                                               (59, 119)], [(116, 90), (156, 198), (373, 326)]],
                             np.float32) / image_size

    grid_sizes_table = np.array([13, 26, 52])
    nclasses = 80

    model = yolov3_model(anchors_table, image_size, nclasses=nclasses)


    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks

# from absl import app, flags, logging
import logging
import numpy as np

def main():
    tf.random.set_seed(seed=42)
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(message)s',
                        )

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    size = 416
    # Setup
    # if False:
    #     pass
        # for physical_device in physical_devices:
        #     tf.config.experimental.set_memo_meshgridry_growth(physical_device, True)
        #
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        # FLAGS.batch_size = BATCH_SIZE
        #
        # with strategy.scope():
        #     model, optimizer, loss, anchors, anchor_masks = setup_model()
    # else:
    #     model, optimizer, loss, anchors, anchor_masks = setup_model()



    train_dataset, dataset_size, batch_size, image_size, anchors_table, max_boxes, grid_sizes_table, nclasses = config_train()

    ###
    epochs = 100
    import dataset
    mode = 'eager_tf'
    num_classes = 80
    learning_rate = 0.001

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                             (59, 119), (116, 90), (156, 198), (373, 326)],
                            np.float32) / image_size

    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    # anchors = yolo_anchors
    # anchor_masks = yolo_anchor_masks
    from models import yolov3_model

    model = yolov3_model(anchors_table, image_size, nclasses=nclasses)

    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    # loss = [YoloLoss(anchors[mask], classes=num_classes)
    #         for mask in anchor_masks]
    # # # #
    loss = [get_loss_func(anchors, nclasses=nclasses)
            for anchors in anchors_table]


    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(mode == 'eager_fit'))


    ###


    # if FLAGS.dataset:
    #     train_dataset = dataset.load_tfrecord_dataset(
    #         FLAGS.dataset, FLAGS.classes, FLAGS.size)
    # else:
    # batch_size = 4

    # image_size = 416
    # max_boxes = 100

    # anchors_table = np.array([[(10, 13), (16, 30), (33, 23)], [(30, 61), (62, 45),
    #                                                            (59, 119)], [(116, 90), (156, 198), (373, 326)]],
    #                          np.float32) / image_size

    # grid_sizes_table = np.array([13, 26, 52])
    # train_dataset = load_fake_dataset()
    # train_dataset = train_dataset.repeat()
    # train_dataset = dataset.load_fake_dataset()
    train_dataset = preprocess_dataset(train_dataset, batch_size, image_size, anchors_table, grid_sizes_table,
                                       max_boxes)

    # ds1 = next(train_dataset1.as_numpy_iterator())
    # y1 = ds1[1]
    # x1 = ds1[0]

    # ds = next(train_dataset.as_numpy_iterator())
    # y = ds[1]
    # y = tf.tile(y[tf.newaxis, :, :], [batch_size, 1, 1])
    # from preprocess_dataset import arrange_in_grid
    # xx = arrange_in_grid(y, tf.convert_to_tensor(anchors_table[0]),  # ronen TODO was 3,6 check shape
    #                      [batch_size, 13, 13, anchors_table[0].shape[0], tf.shape(y)[-1]+1], max_boxes)


    # train_dataset = train_dataset.shuffle(buffer_size=512)
    # train_dataset = train_dataset.repeat()
    # train_dataset = train_dataset.batch(batch_size)

    # ds = next(train_dataset.as_numpy_iterator())
    # y = ds[1]
    # x = ds[0]
    # cc = dataset.transform_targets(y, anchors, anchor_masks, size)



    # train_dataset = train_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, size),
    #     dataset.transform_targets(y, anchors, anchor_masks, size)))
    # train_dataset = train_dataset.prefetch(
    #     buffer_size=tf.data.experimental.AUTOTUNE)
    #
    # ds = next(train_dataset.as_numpy_iterator())
    # y = ds[1]
    # x = ds[0]
    # cc = dataset.transform_targets(y, anchors, anchor_masks, size)


    # if FLAGS.val_dataset:
    #     val_dataset = dataset.load_tfrecord_dataset(
    #         FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    # else:
    # val_dataset = dataset.load_fake_dataset()
    # val_dataset = val_dataset.batch(batch_size)
    # val_dataset = val_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, size),
    #     dataset.transform_targets(y, anchors, anchor_masks, size)))

    if mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, epochs + 1+1000):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            # for batch, (images, labels) in enumerate(val_dataset):
            #     outputs = model(images)
            #     regularization_loss = tf.reduce_sum(model.losses)
            #     pred_loss = []
            #     for output, label, loss_fn in zip(outputs, labels, loss):
            #         pred_loss.append(loss_fn(label, output))
            #     total_loss = tf.reduce_sum(pred_loss) + regularization_loss
            #     import numpy as np
            #     logging.info("{}_val_{}, {}, {}".format(
            #         epoch, batch, total_loss.numpy(),
            #         list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            #     avg_val_loss.update_state(total_loss)
            #
            # logging.info("{}, train: {}, val: {}".format(
            #     epoch,
            #     avg_loss.result().numpy(),
            #     avg_val_loss.result().numpy()))
            #
            # avg_loss.reset_states()
            # avg_val_loss.reset_states()
            # model.save_weights(
            #     'checkpoints/yolov3_train_{}.tf'.format(epoch))
    # else:
    #
    #     callbacks = [
    #         ReduceLROnPlateau(verbose=1),
    #         EarlyStopping(patience=3, verbose=1),
    #         ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
    #                         verbose=1, save_weights_only=True),
    #         TensorBoard(log_dir='logs')
    #     ]
    #
    #     start_time = time.time()
    #     history = model.fit(train_dataset,
    #                         epochs=epochs,
    #                         callbacks=callbacks,
    #                         validation_data=val_dataset)
    #     end_time = time.time() - start_time
    #     print(f'Total Training Time: {end_time}')




if __name__ == '__main__':
    # train()
    main()
