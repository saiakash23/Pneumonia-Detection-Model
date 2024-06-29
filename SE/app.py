# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import io

# app = Flask(__name__)

# # Define paths for uploading and saving files
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# # Load the TensorFlow model
# loaded_model = tf.keras.models.load_model('C:/Users/saiak/OneDrive/Desktop/SE Project/SE/Pneumonia_model.h5')

# def preprocess_image(image_bytes):
#     # Preprocess the image
#     img = Image.open(io.BytesIO(image_bytes))
#     img = img.resize((256, 256))  # Specify the target size used during training
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#     img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1
#     return img_array

# def predict_pneumonia(image_bytes):
#     # Make prediction
#     prediction = loaded_model.predict(image_bytes)
#     # If your model predicts probabilities, you might want to get the class label with highest probability
#     predicted_class = "Normal" if np.argmax(prediction)==0 else "Pneumonia"
#     return predicted_class

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(file_path)

#         with open(file_path, 'rb') as f:
#             file_content = f.read()
#             print("File content:", file_content)

#         image_bytes = preprocess_image(file_content)  # Use file_content instead of file.read()
#         prediction = predict_pneumonia(image_bytes)
#         return render_template('result.html', prediction=prediction)

#     else:
#         return redirect(request.url)

# if __name__ == '__main__':
#     # Create upload folder if it doesn't exist
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
loaded_model = tf.keras.models.load_model('C:/Users/saiak/OneDrive/Desktop/SE Project/SE/Pneumonia_model.h5')

# Function to process uploaded image and make prediction
def predict_pneumonia(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))  # Specify the target size used during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1
    # Make prediction
    prediction = loaded_model.predict(img_array)
    # If your model predicts probabilities, you might want to get the class label with highest probability
    predicted_class = "Normal" if np.argmax(prediction)==0 else "Pneumonia"
    return predicted_class

# Route to handle file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_class = predict_pneumonia(file_path)
            return render_template('result.html', filename=filename, predicted_class=predicted_class)
    return render_template('se.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=True)

