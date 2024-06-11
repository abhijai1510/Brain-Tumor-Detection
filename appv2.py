import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained Keras model with TensorFlow Hub layer
MODEL_PATH = 'brain_tumor.h5'
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})

def prepare_image(img_path):
    """Load and preprocess the image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(img_array):
    """Predict the presence of a brain tumor from the image array."""
    prediction = model.predict(img_array)
    return "The Person has Brain Tumor" if prediction[0] > 0.5 else "The Person has no Brain Tumor"

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    """Handle image upload and return prediction results."""
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected for uploading", 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        img_array = prepare_image(file_path)
        result = predict_image(img_array)
        return result  # Directly return the result as plain text

    return "Invalid file or file upload failed", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    app.run(debug=True, port=5000)