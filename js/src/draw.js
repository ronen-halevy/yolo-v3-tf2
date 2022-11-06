
      function drawOnImage(image, rect) {
        const imageWidth = image.width;
        const imageHeight = image.height;
        canvas.width = imageWidth;
        canvas.height = imageHeight;
        context.drawImage(image, 0, 0, imageWidth, imageHeight);
        let a = tf.ones([2,2])
        context.beginPath();
        
        context.rect(188, 50, 200, 100);
        context.fillStyle = "yellow";
        //context.fill();
        context.lineWidth = 7;
        context.strokeStyle = "white";
        context.stroke();
      }

      function fileToDataUri(field) {
        return new Promise((resolve) => {
          const reader = new FileReader();

          reader.addEventListener("load", () => {
            resolve(reader.result);
          });

          reader.readAsDataURL(field);
        });
      }
    function addFileEventListener(fileInput){



      fileInput.addEventListener("change", async (e) => {
        const [file] = fileInput.files;

        // displaying the uploaded image
        const image = document.createElement("img");
        image.src = await fileToDataUri(file);

        // enabling the brush after after the image
        // has been uploaded
        image.addEventListener("load", () => {
          drawOnImage(image);
          runInference(image)
        });

        return false;
      });
    }


