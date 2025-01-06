from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import uuid
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)

# Load the trained model
model = load_model('D:/Lehan/mmmmmm/eye_disease_classifier_model.h5')

# Define image dimensions for model input
img_width, img_height = 180, 180

# Define a directory to store images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define a route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load the image
        img = Image.open(file)
        img = img.resize((img_width, img_height))  # Resize the image to fit the model input
        
        # Preprocess the image for the model
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Rescale the image
        
        # Predict the class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)  # Get the class with the highest probability
        confidence = prediction[0][predicted_class]  # Get the confidence level
        
        # Get the class label
        class_labels = ['Cataract', 'Glaucoma', 'Retina Disease', 'Normal']  # Change according to your classes
        predicted_label = class_labels[predicted_class]
        
        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")  # Save image in a buffer
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Convert to base64 string
        
        # Return the response with base64-encoded image and prediction
        return jsonify({
            'predicted_class': predicted_label,
            'confidence': float(confidence),
            'img': img_base64  # Send the image as base64 string
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
