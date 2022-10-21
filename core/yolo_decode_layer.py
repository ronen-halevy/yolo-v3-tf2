import tensorflow as tf


def __arrange_bbox(xy, wh):
    grid_size = tf.shape(xy)[1:3]
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    xy = (xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    bbox = tf.concat([xy_min, xy_max], axis=-1)
    return bbox


def yolo_decode(model_output_grids, anchors_table, nclasses):
    pred_xy, pred_wh, pred_obj, class_probs = zip(
        *[tf.split(output_grid, (2, 2, 1, nclasses), axis=-1) for output_grid in model_output_grids])
    bboxes_in_grids = [__arrange_bbox(xy, tf.exp(wh) * anchors) for xy, wh, anchors in
                       zip(pred_xy, pred_wh, anchors_table)]

    all_grids_bboxes = tf.concat(
        [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in bboxes_in_grids],
        axis=1)

    all_grids_confidence = tf.concat(
        [tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in pred_obj],
        axis=1)

    all_grids_class_probs = tf.concat([tf.reshape(y, [tf.shape(y)[0], -1, tf.shape(y)[-1]]) for y in class_probs],
                                      axis=1)
    return all_grids_bboxes, all_grids_confidence, all_grids_class_probs
