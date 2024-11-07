from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'my_model_digit.keras')
model = load_model(model_path)

def preprocess_image(image_data):
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

def predict_digit(image_data):
    image = preprocess_image(image_data)
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data.get('image')
    if image_data:
        predicted_digit = predict_digit(image_data)
        return jsonify({'prediction': int(predicted_digit)})
    return jsonify({'error': 'No image data provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
