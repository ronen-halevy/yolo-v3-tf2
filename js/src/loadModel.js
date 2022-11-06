async function load_model() {
    // It's possible to load the model locally or from a repo
    // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json" just set it before in your https server
    //const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
    // const model = await loadGraphModel("https://raw.githubusercontent.com/hugozanini/TFJS-object-detection/master/models/kangaroo-detector/model.json");

    const MODEL_URL = "http://127.0.0.1:8887/model/model.json";
    const model = await tf.loadLayersModel(MODEL_URL);
    console.log(model.summary());


    return model;
  }