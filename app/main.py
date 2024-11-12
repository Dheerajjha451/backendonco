from flask import request, jsonify
from PIL import Image
import numpy as np
from app import app, model, class_names

@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image file is in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file)
        image = image.resize((256, 256))
        image = np.array(image) / 255.0  # Normalize image

        # Predict using the loaded model
        prediction = model.predict(image[np.newaxis, ...])
        predicted_class = class_names[np.argmax(prediction)]

        return jsonify({"prediction": predicted_class})

    except Exception as e:
        return jsonify({"error": str(e)}), 400
