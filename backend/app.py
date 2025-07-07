# app.py

import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import request, jsonify, Flask
from flask_cors import CORS

# Initialize our Flask application
app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# Load the pre-trained Keras model
# This is loaded only once when the server starts
def get_model():
    global model
    model = load_model('mnist_cnn.h5')
    print(" * Model loaded!")

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the image data from the POST request
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded.split(',')[1])
    
    # Open the image and process it
    image = Image.open(io.BytesIO(decoded))
    
    # 1. Convert to grayscale
    image = image.convert('L')
    
    # 2. Resize to 28x28 pixels
    image = image.resize((28, 28))
    
    # 3. Convert image to a numpy array
    image_array = np.array(image)
    
    # 4. Reshape for the model (1 sample, 28x28 pixels, 1 channel)
    image_array = image_array.reshape(1, 28, 28, 1)
    
    # 5. Normalize the image data (from 0-255 to 0-1)
    image_array = image_array.astype('float32') / 255.0
    
    # 6. Make a prediction
    prediction = model.predict(image_array)
    
    # 7. Get the highest probability index (the predicted digit)
    predicted_digit = int(np.argmax(prediction))
    
    # Return the prediction as a JSON response
    response = {'prediction': predicted_digit}
    
    return jsonify(response)

# --- Main execution block ---
if __name__ == "__main__":
    print(" * Loading Keras model...")
    get_model()
    # Run the app on localhost at port 5000
    app.run(port=5000, debug=True)