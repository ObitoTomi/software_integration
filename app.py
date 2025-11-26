import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

# Suppress TensorFlow logging
"""
2025-11-19 15:09:03.254768: I tensorflow/core/util/port.cc:153] oneDNN 
custom operations are on. You may see slightly different numerical results 
due to floating-point round-off errors from different computation orders. 
To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

model_path = os.getenv("MODEL_PATH")

if not model_path:
    raise ValueError("MODEL_PATH is not set in the .env file")

model = load_model(model_path)


app = Flask(__name__)


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}


def allowed_file(filename):
    """Check file extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image, target_size=(128, 128)):
    """Convert to RGB, resize and scale."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, 0)  # batch dimension
    return img_array



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file in request"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400


        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Unsupported file type"}), 400


        try:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes))
        except:
            return jsonify({"success": False, "error": "Failed to decode image"}), 400

        try:
            target_h = model.input_shape[1]
            target_w = model.input_shape[2]
            image = image.convert("RGB")
            image = image.resize((target_w, target_h))

        except Exception as e:
            return jsonify({"success": False, "error": f"Resize error: {str(e)}"}), 500

        try:
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, 0)
        except Exception as e:
            return jsonify({"success": False, "error": f"Preprocessing error: {str(e)}"}), 500


        try:
                # label_map = {0: "cat", 1: "dog"}
                prediction = model.predict(img_array)[0][0] # returns first image in the batch and signle output
                label = "dog" if prediction >= 0.5 else "cat"
                confidence = prediction if prediction >= 0.5 else 1 - prediction


        except Exception as e:
            # Prevent freeze by catching TensorFlow errors
            return jsonify({
                "success": True,
                "prediction": "unknown",
                "confidence": 0.0,
                "warning": f"Model error: {str(e)}"
            }), 200


        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": round(float(confidence), 4)
        }), 200

    except Exception as e:
        # SAFETY NET — Avoid full API freeze
        return jsonify({"success": False, "error": f"Server crash prevented: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=True)
