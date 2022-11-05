
// const {readFileSync, promises: fsPromises} = require('fs');

// function fn1(){};


async function arrange_bbox(xy, wh) {
  console.log("arrange_bbox xy", xy);
  console.log("arrange_bbox wh", wh);

  let grid_size = [xy.shape[1], xy.shape[1]];

  console.log("pre grid_size", grid_size);

  console.log("xx1", grid_size);
  console.log("xx1", grid_size);
  vv = tf.range(0, 13, 1).print();
  let grid = tf.meshgrid(
    tf.range(0, xy.shape[1], 1),
    tf.range(0, xy.shape[1], 1)
  );
  console.log("000!!!grid", grid);
  var axis = -1;
  grid = tf.stack(grid, axis);
  console.log("--11111!!!grid", grid);

  axis = 2;
  grid = grid.expandDims(axis);
  console.log("!!!grid", tf.cast(grid, "float32"));
  //gg.print();
  //xy.print();

  //xy.print();
  //     xx = tf.cast(grid_size, "float32").print();
  //     zz = tf.cast(grid, "float32").print();
  xy = xy.add(tf.cast(grid, "float32"));
  xy = xy.div(tf.cast(grid_size, "float32"));

  console.log("xy", xy);

  // console.log("wh", wh);
  xy.print()
  let  value1 = tf.scalar(2.);
  wh = wh.div(value1);
  wh.print()
  var xy_min = xy.sub(wh);
  var xy_max = xy.add(wh);

//   var xy_min = xy.sub(wh / 2);
//   var xy_max = xy.add(wh / 2);
  xy_min.print()
  xy_max.print()

  //  console.log("xy_min", xy_min);
  //  console.log("xy_max", xy_max);

  var bbox = tf.concat([xy_min, xy_max], -1);
  bbox.print();
  return bbox;
}

async function yolo_decode(grids_outputs, nclasses) {
    let anchors_table = [
        0.16827, 0.16827,  
        0.16827,  0.16827,
        0.16827,   0.16827,
        0.16827,   0.16827,
        0.16827,   0.16827,
    0.16827,   0.16827];

    nanchors_per_scale = 3
    anchor_entry_size = 2
    anchors_table = tf.reshape(anchors_table, [-1, nanchors_per_scale, anchor_entry_size])

    let preds = output_grid.map(x => tf.split(x, [2, 2, 1, nclasses], axis=-1));
    console.log('11',xx);
    let pred_xy = []
    let pred_wh = []
    let pred_obj = []
    let class_probs = []
    for (let grid_output in grids_outputs) {
        // let preds = grid_output.map(x => tf.split(x, [2, 2, 1, nclasses], axis=-1));
        fruits.slice(1, 3);
        pred_xy.pushp(grid_output.slice(0, 2))
        pred_wh.pushp(grid_output.slice(0, 2))
        pred_obj.pushp(grid_output.slice(0, 2))
        class_probs.pushp(grid_output.slice(0, 2))
    }

    console.log('1', pred_xy.shape);
    console.log('2', pred_wh.shape);
    console.log('3', pred_obj.shape);
    console.log('4', class_probs.shape);



};

function get_anchorsf(anchors_file) {


    const nanchors_per_scale = 3
    const anchor_entry_size = 2
    anchors_table = loadtxt(anchors_file, dtype=np.float, delimiter=',')
    anchors_table = anchors_table.reshape(
        -1, nanchors_per_scale, anchor_entry_size)
    return anchors_table
}


async function yolo_decode(grids_outputs, nclasses) {
    const anchors = [
      0.16827, 0.16827, 0.16827, 0.16827, 0.16827, 0.16827, 0.16827, 0.16827,
      0.16827, 0.16827, 0.16827, 0.16827,
    ];

 

    const nanchors_per_scale = 3;
    const anchor_entry_size = 2;
    let anchors_table = tf.reshape(anchors, [
       -1,
      nanchors_per_scale,
      anchor_entry_size,
    ]);
    let pred_xy = [];
    let pred_wh = [];
    let pred_obj = [];
    let class_probs = [];
    // grids_outputs[0].print()
    grids_outputs[0].min().print()
    grids_outputs[0].max().print()

    console.log("??grids_outputs", grids_outputs);
    console.log("??grids_outputs[0]", grids_outputs[0]);
    console.log("??grids_outputs[1]", grids_outputs[1]);

    let grids_bboxes = [];
    let grids_confidence = [];
    let grids_class_probs = [];
    console.log("000000grids_outputs!!!!!!!!!!!!!!!!!", grids_outputs);
    for (let idx = 0; idx < grids_outputs.length; idx++) {
      console.log("grids_outputs!!!!!!!!!!!!!!!!!", grids_outputs);
      let [xy, wh, obj, class_prob] = tf.split(
        grids_outputs[idx],
        [2, 2, 1, nclasses],
        (axis = -1)
      );
      wh.print();
      console.log("after split wh", wh);
      console.log("after split tf.exp(wh)", tf.exp(wh));
      var whh = wh.exp();
      const indices = tf.tensor1d([0], 'int32');
      console.log(anchors_table);
      let anchors =  tf.slice(anchors_table, [idx], 1);
      var wha =whh.mul(anchors).print();

      const bboxes_in_grid = await arrange_bbox(
        tf.sigmoid(xy),
        wh.exp().mul(anchors)
      );
      console.log("after boxes obj", obj);

      bb = [bboxes_in_grid.shape[0], -1, bboxes_in_grid.shape[-1]];
      console.log("bb", bboxes_in_grid.shape);
      console.log("bb", bboxes_in_grid.shape[0]);
      console.log("bb", bboxes_in_grid.shape[4]);

      grids_bboxes.push(
        tf.reshape(bboxes_in_grid, [
          bboxes_in_grid.shape[0],
          -1,
          bboxes_in_grid.shape[4],
        ])
      );

      grids_confidence.push(
        tf.reshape(tf.sigmoid(obj), [obj.shape[0], -1, obj.shape[4]])
      );
      grids_class_probs.push(
        tf.reshape(tf.sigmoid(class_prob), [
          class_prob.shape[0],
          -1,
          class_prob.shape[4],
        ])
      );

      //  class_probs.push(tf.sigmoid(class_prob));
    }

    // console.log("1", pred_xy);
    // console.log("2", pred_wh);
    // console.log("3", pred_obj);
    // console.log("4", class_probs);
    grids_bboxes = tf.concat(grids_bboxes, (axis = 1));
    grids_confidence = tf.concat(grids_confidence, (axis = 1));

    grids_class_probs = tf.concat(grids_class_probs, (axis = 1));

    console.log("grids_bboxes", grids_bboxes);
    console.log("grids_confidence", grids_confidence);
    console.log("grids_class_probs", grids_class_probs);

    return [grids_bboxes, grids_confidence, grids_class_probs];
  }


