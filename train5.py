from absl import app, flags, logging
from absl.flags import FLAGS
import yaml
import tensorflow as tf
import numpy as np
# import cv2
import time
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3_train.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2000, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_boolean('multi_gpu', False, 'Use if wishing to train with more than 1 GPU.')

from tensorflow.keras import Input, Model
from config.core.parse_model import ParseModel

def setup_model_new(nclasses=80, model_config_file='config/models/yolov3/model.yaml'):
    with open(model_config_file, 'r') as _stream:
        model_config = yaml.safe_load(_stream)
    parse_model = ParseModel()
    inputs = Input(shape=(None, None, 3))
    inputs = Input(shape=(416, 416, 3))

    sub_models_configs = model_config['sub_models_configs']
    output_stage = model_config['output_stage']
    decay_factor = model_config['decay_factor']
    model = parse_model.build_model(inputs, sub_models_configs, output_stage, decay_factor, nclasses)

    with open("model_summary_new.txt", "w") as file1:
        model.summary(print_fn=lambda x: file1.write(x + '\n'))
    return model


def setup_model():
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        # model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        # with open("model_summary_old.txt", "w") as file1:
            # model.summary(print_fn=lambda x: file1.write(x + '\n'))
        model = setup_model_new()

        # for l1 in model_old.layers:
        #     print(l1.get_config())
        #
        # for l1 in model.layers:
        #     print(l1.get_config())

        # for l1, l2 in zip(mdl.layers, mdl2.layers):
        #     print(l1.get_config() == l2.get_config())


        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes
        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(FLAGS.weights)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)
    else:
        # All other transfer require matching classes
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]
    # # # ######### ronen
    # from core.utils import get_anchors, count_file_lines
    # from core.loss_func import get_loss_func
    #
    # anchors_file = 'core/anchors.txt'
    # anchors_table = get_anchors(anchors_file)
    #
    # loss_new = [get_loss_func(anchors, nclasses=FLAGS.num_classes)
    #         for anchors in anchors_table]
    # # ########

    model.compile(optimizer=optimizer, loss=loss,
                  run_eagerly=(FLAGS.mode == 'eager_fit'))

    return model, optimizer, loss, anchors, anchor_masks


from config.core.utils import get_anchors, count_file_lines
from config.core.preprocess_dataset import PreprocessDataset


def preprocess(train_dataset, val_dataset):
    #######
    #
    ds = [train_dataset, val_dataset]

    #
    batch_size = 1
    image_size = 416
    anchors_file = 'core/anchors.txt'
    anchors_table = get_anchors(anchors_file)

    grid_sizes_table = [13, 26, 52]
    grid_sizes_table = np.array(grid_sizes_table)

    max_bboxes = 100
    preprocess_dataset = PreprocessDataset()

    # if True:  # debug_mode:
    #     preprocess_dataset.preprocess_dataset_debug(ds[0], batch_size, image_size, anchors_table,
    #                                                 grid_sizes_table,
    #                                                 max_bboxes)
    ds_preprocessed = []
    for ds_split in ds:
        data = preprocess_dataset(ds_split, batch_size, image_size, anchors_table, grid_sizes_table,
                                  max_bboxes)
        ds_preprocessed.append(data)

    train_dataset, val_dataset = ds_preprocessed

    return train_dataset, val_dataset


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # Setup
    if FLAGS.multi_gpu:
        for physical_device in physical_devices:
            tf.config.experimental.set_memo_meshgridry_growth(physical_device, True)

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
        FLAGS.batch_size = BATCH_SIZE

        with strategy.scope():
            model, optimizer, loss, anchors, anchor_masks = setup_model()
    else:
        model, optimizer, loss, anchors, anchor_masks = setup_model()

    # model= setup_model_new()
    # # ######### ronen
    from config.core.utils import get_anchors, count_file_lines
    from config.core.loss_func import get_loss_func


    anchors_file = 'core/anchors.txt'
    anchors_table = get_anchors(anchors_file)

    loss_new = [get_loss_func(anchors, nclasses=FLAGS.num_classes)
            for anchors in anchors_table]
    # ########




    if FLAGS.dataset:
        train_dataset_c = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset_c = dataset.load_fake_dataset()
    # train_dataset = train_dataset_c.shuffle(buffer_size=512)
    # train_dataset = train_dataset.repeat()
    # train_dataset = train_dataset_c.batch(FLAGS.batch_size)
    # train_dataset = train_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, FLAGS.size),
    #     dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # train_dataset = train_dataset.prefetch(
    #     buffer_size=tf.data.experimental.AUTOTUNE)
    # ds = train_dataset.take(1)
    # for x,yy in ds:
    #     for idx, y in enumerate(yy):
    #         iy = tf.where(y)
    #         print(f'iy {idx}', iy)
    #         giy = tf.gather_nd(y, iy)
    #         print(f'giy {idx}', giy)


    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
    # val_dataset = val_dataset.batch(FLAGS.batch_size)
    # val_dataset = val_dataset.map(lambda x, y: (
    #     dataset.transform_images(x, FLAGS.size),
    #     dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    train_dataset, val_dataset = preprocess(train_dataset_c, train_dataset_c)

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    pred_loss_new = []
                    for output, label, loss_fn, loss_fn_new in zip(outputs, labels, loss, loss_new):
                        #pred_loss.append(loss_fn(label, output))
                        pred_loss.append(loss_fn_new(label, output))

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
            #
            #     logging.info("{}_val_{}, {}, {}".format(
            #         epoch, batch, total_loss.numpy(),
            #         list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            #     avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train.tf')
    else:

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        start_time = time.time()
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)
        end_time = time.time() - start_time
        print(f'Total Training Time: {end_time}')


if __name__ == '__main__':
    tf.random.set_seed(42) # todo temp

    try:
        app.run(main)
    except SystemExit:
        pass
