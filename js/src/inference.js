


  async function runInference(img) {
    tensor = imagePreprocess(img);
    // const MODEL_URL = "http://127.0.0.1:8887/model/model.json";
    // const model = await tf.loadLayersModel(MODEL_URL);
    // console.log(model.summary());
    model = await load_model()
    var model_output_grids = await model.predict(tensor);
    console.log("model_output_grids", model_output_grids);
    const nclasses = 7;

    let outputs = await yolo_decode(model_output_grids, nclasses);
    
    zz = await outputs[0];

    let yolo_max_boxes = 100;
    let nms_iou_threshold = 0.5;
    let nms_score_threshold = 0.5;
    let detection_indices = await yolo_nms(
      outputs[0],
      outputs[1],
      outputs[1],
      yolo_max_boxes,
      nms_iou_threshold,
      nms_score_threshold
    );

    
    let bbbox = outputs[0].squeeze();
    let classes_ids = outputs[1].squeeze();

    box = bbbox.gather(detection_indices).print();
    classes = bbbox.gather(detection_indices).print();
    box = bbbox.gather(detection_indices).print();

  }


  function imagePreprocess(img) {
    const imgTensor = tf.browser.fromPixels(img);
  
    var resized = tf.image.resizeBilinear(imgTensor, [416, 416]);
    var tensor = resized.expandDims(0);
    tensor = tensor.div(255);
    return tensor

  }

