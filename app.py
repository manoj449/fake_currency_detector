import numpy as np
from flask import Flask, request, render_template
import cv2
import tensorflow as tf
from keras.models import load_model
from werkzeug.utils import secure_filename
import os

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model('vgg.h5')

# Function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function
def process_jpg_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")

    img = cv2.resize(img, (224, 224))  # ✅ Required
    img = tf.convert_to_tensor(img[:, :, :3])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_text='No file selected')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process image and predict
            test_image = process_jpg_image(file_path)
            prediction = model.predict(test_image)
            class_names = [('fake', 0), ('real', 1)]
            predicted_index = int(np.argmax(prediction))
            predicted_label = class_names[predicted_index][0]

            return render_template(
                'index.html',
                prediction_text=f'The currency is {predicted_label}'
            )
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {e}')
    else:
        return render_template('index.html', prediction_text='Invalid file type. Only JPG, JPEG, and PNG are allowed.')

# Run app
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
