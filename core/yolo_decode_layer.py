import tensorflow as tf


class YoloDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, nclasses, yolo_max_boxes, anchors_table, nms_iou_threshold, nms_score_threshold, **kwargs):
        self.nclasses =nclasses
        self.yolo_max_boxes = yolo_max_boxes
        self.anchors_table = anchors_table
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_score_threshold = nms_score_threshold
        super(YoloDecoderLayer, self).__init__(**kwargs)

    @staticmethod
    def yolo_nms(outputs, yolo_max_boxes, nms_iou_threshold, nms_score_threshold):
        bbox, confidence, class_probs = outputs
        class_probs = tf.squeeze(class_probs, axis=0)

        class_indices = tf.argmax(class_probs, axis=-1)
        class_probs = tf.reduce_max(class_probs, axis=-1)
        scores = confidence * class_probs
        scores = tf.squeeze(scores, axis=0)
        scores = tf.reduce_max(scores, [1])
        bbox = tf.reshape(bbox, (-1, 4))

        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=scores,
            max_output_size=yolo_max_boxes,
            iou_threshold=nms_iou_threshold,
            score_threshold=nms_score_threshold,
            soft_nms_sigma=0.
        )

        num_of_valid_detections = tf.expand_dims(tf.shape(selected_indices)[0], axis=0)
        selected_boxes = tf.gather(bbox, selected_indices)
        selected_boxes = tf.expand_dims(selected_boxes, axis=0)
        selected_scores = tf.expand_dims(selected_scores, axis=0)
        selected_classes = tf.gather(class_indices, selected_indices)
        selected_classes = tf.expand_dims(selected_classes, axis=0)

        return selected_boxes, selected_scores, selected_classes, num_of_valid_detections

    @staticmethod
    def arrange_bbox(xy, wh):
        grid_size = tf.shape(xy)[1:3]

        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        xy_min = xy - wh / 2
        xy_max = xy + wh / 2
        bbox = tf.concat([xy_min, xy_max], axis=-1)
        return bbox

    def call(self, grids_outputs, **kwargs):
        pred_xy, pred_wh, pred_obj, class_probs = zip(
            *[tf.split(grid_out, (2, 2, 1, self.nclasses), axis=-1) for grid_out in grids_outputs])
        bboxes_grid0 = self.arrange_bbox(pred_xy[0], tf.exp(pred_wh[0]) * self.anchors_table[2])
        bboxes_grid1 = self.arrange_bbox(pred_xy[1], tf.exp(pred_wh[1]) * self.anchors_table[1])
        bboxes_grid2 = self.arrange_bbox(pred_xy[2], tf.exp(pred_wh[2]) * self.anchors_table[0])

        all_grids_bboxes = tf.concat(
            [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in [bboxes_grid0, bboxes_grid1, bboxes_grid2]],
            axis=1)

        all_grids_confidence = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in pred_obj],
                                         axis=1)

        all_grids_class_probs = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in class_probs],
                                          axis=1)

        (selected_boxes, selected_scores, selected_classes, num_of_valid_detections) = \
            self.yolo_nms((all_grids_bboxes, all_grids_confidence, all_grids_class_probs), self.yolo_max_boxes,
                     self.nms_iou_threshold,
                     self.nms_score_threshold)
        return (selected_boxes, selected_scores, selected_classes, num_of_valid_detections)




    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        num_anchors = len(self.anchors)
        stride = self.input_dims[0] // input_shape[1]
        grid_size = self.input_dims[0] // stride
        num_bboxes = num_anchors * grid_size * grid_size

        shape = (batch_size, num_bboxes, self.num_classes + 5)

        return tf.TensorShape(shape)

from core.yolo_decode_layer import YoloDecoderLayer
def build_yolo_decoder_layer(x, layer_config, layers, outputs, ptr, config):
    nclasses = config['nclasses']
    yolo_max_boxes = config['yolo_max_boxes']
    anchors_table = config['anchors_table']
    nms_iou_threshold = config['nms_iou_threshold']
    nms_score_threshold = config['nms_score_threshold']

    x = YoloDecoderLayer(nclasses, yolo_max_boxes, anchors_table, nms_iou_threshold, nms_score_threshold)(x)
    layers.append(None)
    outputs.append(x)
    return x, layers, outputs, ptr