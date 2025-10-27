import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from datetime import datetime

# -------------------- Configuration --------------------
app = Flask(__name__)

# Folders for uploads and models
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_PATH'] = r"C:\Users\V K MAOJ\Desktop\brain_tumor_app\models\resnet50_brain_tumor_classifier.h5"


# Labels used in training
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# -------------------- Load Model --------------------
if not os.path.exists(app.config['MODEL_PATH']):
    raise FileNotFoundError(
        f"Model file not found at '{app.config['MODEL_PATH']}'. "
        f"Please train the model first and save it at this path."
    )

model = load_model(app.config['MODEL_PATH'])
WIDTH, HEIGHT = 200, 200

# -------------------- Utility Functions --------------------
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=1)
    img = tf.image.grayscale_to_rgb(img)  # Convert to 3 channels
    img = tf.image.resize(img, [WIDTH, HEIGHT])
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Preprocess and predict
    try:
        img = preprocess_image(file_path)
        pred = model.predict(img)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100
        result = f"{LABELS[pred_class]} ({confidence:.2f}% confidence)"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

    # Create result text file for download
    result_filename = f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    with open(result_path, 'w') as f:
        f.write(f"Predicted Tumor Type: {LABELS[pred_class]}\n")
        f.write(f"Confidence: {confidence:.2f}%\n")
        f.write(f"File: {filename}\n")

    return render_template(
        'result.html',
        image_path=file_path,
        result=result,
        download_link=url_for('download', filename=result_filename)
    )

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found!"

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run(debug=True)
