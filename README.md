# Vision Transformer for Digit Recognition

## Overview

This project implements a Vision Transformer (ViT) model for recognizing handwritten digits from the MNIST dataset. The model processes images drawn on a web-based canvas and predicts the corresponding digit. The application consists of a Flask backend serving the model and a simple front-end for user interaction.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Training](#model-training)
- [Contributing](#contributing)

## Features

- Draw digits on a web canvas.
- Predict handwritten digits using a Vision Transformer model.
- Clear the canvas to start anew.
- Responsive and user-friendly interface.

## Technologies Used

- Python
- Flask (for the backend API)
- PyTorch (for building and running the Vision Transformer model)
- OpenCV (for image processing)
- HTML/CSS/JavaScript (for the frontend)
- CORS (for cross-origin requests)

## Setup Instructions

1. Clone the repository:

```bash
   git clone https://github.com/zenitsu0509/DeepVisionModels/handwritten_digtit.git
   cd your-repo
```

-----

1. Run the Flask application:
    `python app.py`

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Draw a digit on the canvas and click the "Predict" button to see the predicted digit.

4. Use the "Clear" button to reset the canvas.

API Endpoints
-------------

- **POST /predict**

  - **Description**: Predict the digit from the drawn image.
  - **Request Body**:
        `{
          "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
        }`

  - **Response**:
        `{
          "prediction": 5
        }`

Model Training
--------------

If you need to retrain the Vision Transformer model, ensure you have the MNIST dataset and follow these steps:

1. Modify the training script to load the MNIST dataset.
2. Adjust hyperparameters such as batch size, learning rate, and number of epochs as needed.
3. Save the model as `vit_mnist_model.pth` after training.

# Contributing

---

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.
