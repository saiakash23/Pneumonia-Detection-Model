
# 🩺 Pneumonia Detection Model

Welcome to the **Pneumonia Detection Model** project! This project leverages a deep learning model to detect pneumonia from chest X-ray images. The backend is powered by Flask, providing a simple web interface for users to upload images and get predictions.

## ✨ Features

- 📷 **Image Upload**: Users can upload chest X-ray images through a web interface.
- 🤖 **Deep Learning Model**: Utilizes a pre-trained neural network model for pneumonia detection.
- 📈 **Real-time Predictions**: Get instant predictions on whether the uploaded image shows signs of pneumonia or not.

## 🚀 Getting Started

### 📋 Prerequisites

Ensure you have the following installed:

- [Python 3.6+](https://www.python.org/downloads/)
- Required Python libraries: `Flask`, `tensorflow`, `numpy`

Install the required libraries using pip:
```bash
pip install Flask tensorflow numpy
```

### ▶️ Running the Application

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. **Set Up Model**:
   - Place your pre-trained model (`Pneumonia_model.h5`) in the root directory or specify its path in the code.

3. **Run the Flask Application**:
   ```bash
   python app.py
   ```

4. **Access the Web Interface**:
   - Open your browser and go to `http://127.0.0.1:5000/`.

## 🖥️ User Interface Overview

### 🎛️ Main Interface

- **Upload Image**: Select and upload a chest X-ray image for pneumonia detection.
- **View Results**: After uploading, the model's prediction will be displayed.

### 📜 Instructions

1. **Upload an Image**:
   - Click the upload button and select a chest X-ray image from your device.
2. **Get Prediction**:
   - Once uploaded, the model will analyze the image and provide a prediction: "Normal" or "Pneumonia".

## 🧩 Code Overview

The main components of the project include:

- **Flask Application**: Manages the web interface and handles image uploads.
- **Model Loading and Prediction**: Loads the pre-trained model and processes images to make predictions.

### Key Functions and Routes

```python
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
loaded_model = tf.keras.models.load_model('Pneumonia_model.h5')

# Function to process uploaded image and make prediction
def predict_pneumonia(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = loaded_model.predict(img_array)
    predicted_class = "Normal" if np.argmax(prediction)==0 else "Pneumonia"
    return predicted_class

# Route to handle file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class = predict_pneumonia(file_path)
            return render_template('result.html', filename=filename, predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)
```

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements and bug fixes.
