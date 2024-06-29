import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
loaded_model = tf.keras.models.load_model('C:/Users/saiak/OneDrive/Desktop/SE Project/Pneumonia_model.h5')
image_path = r"C:/Users/saiak/OneDrive/Desktop/SE Project/person8_virus_27.jpeg"
# Load and preprocess the image
img = image.load_img(image_path, target_size=(256, 256))  # Specify the target size used during training
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize the pixel values to be between 0 and 1
# Make prediction
prediction = loaded_model.predict(img_array)
print(prediction)
# If your model predicts probabilities, you might want to get the class label with highest probability
predicted_class = "Normal" if np.argmax(prediction)==0 else "Pneumonia"
# Print the predicted class
print("Predicted class:", predicted_class)