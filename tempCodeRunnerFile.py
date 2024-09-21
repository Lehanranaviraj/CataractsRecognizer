from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'  # Needed for flash messages
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load the model
model = load_model('model/cataracts_recognizer_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(filepath):
    img = cv2.imread(filepath)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        return img
    else:
        raise ValueError("Image file could not be read")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                img = prepare_image(filepath)
                prediction = model.predict(img)
                result = 'Cataract' if prediction[0][0] > 0.5 else 'Normal'
                return render_template('index.html', result=result, img_path=filepath)
            except Exception as e:
                flash(f"An error occurred while processing the image: {e}")
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif')
            return redirect(request.url)
    return render_template('index.html', result=None, img_path=None)

if __name__ == '__main__':
    app.run(port=5002, debug=True)  # Changed to port 5001 to avoid conflicts
