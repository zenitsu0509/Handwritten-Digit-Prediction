from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import base64
from io import BytesIO
import torch
from torchvision import transforms
from VisionTransformer import VisionTransformer 
import os

app = Flask(__name__)
CORS(app)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device)
model.load_state_dict(torch.load("vit_mnist_model.pth", map_location=device)) 
model.eval()

def preprocess_image(image_data):
    
    image_data = image_data.split(",")[1]
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L') 
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),  
        transforms.ToTensor(),        
        transforms.Normalize((0.5,), (0.5,))  
    ])
    
    image = preprocess(image).unsqueeze(0).to(device)
    return image

def predict_digit(image_data):
    image = preprocess_image(image_data)
    with torch.no_grad():  
        prediction = model(image)
        predicted_digit = torch.argmax(prediction, dim=1).item() 
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
