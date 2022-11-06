async function yolo_nms(bboxes, confidence, class_probs, yolo_max_boxes, nms_iou_threshold, nms_score_threshold) {

    axis=-1
    class_indices = class_probs.argMax(axis)
    // select class from class probs array
    class_probs = class_probs.max(axis)
    class_probs = class_probs.expandDims(axis)


    scores = confidence.mul(class_probs)

    scores = scores.squeeze(axis)
    axis = 0
    bboxes = bboxes.squeeze(axis)
    scores = scores.squeeze(axis)

    // non_max_suppression_padded vs non_max_suppression supports batched input, returns results per batch
    pad_to_max_output_size=true;
    // nms_result = await tf.image.nonMaxSuppressionPaddedAsync(bboxes, scores, yolo_max_boxes, nms_iou_threshold, nms_score_threshold, pad_to_max_output_size)
    let nms_result = await tf.image.nonMaxSuppressionAsync(bboxes, scores, yolo_max_boxes, nms_iou_threshold, nms_score_threshold)
    nms_result.print()


    return nms_result;
}
