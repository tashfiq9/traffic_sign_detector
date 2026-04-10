from flask import Flask, render_template, request, jsonify
import os
import time
import threading
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import requests as _requests

import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten,
    Dropout, BatchNormalization, Input
)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

NUM_CLASSES = 43
MODEL_IMG_SIZES = {'cnn': 48, 'eff': 96, 'mob': 96}

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_IDS = {
    'cnn': '1aL5YZVqdY7yOBhTCHBWccyV1zBBzzlZh',
    'eff': '1EmJIAoRSsyeM5btNgEviDv5TapNehJJm',
    'mob': '1SFwKU9pXo6rLtRb4UXWDolOph8AVg61f',
}

MODEL_FILENAMES = {
    'cnn': 'TSR_best.keras',
    'eff': 'EfficientNetB0_fixed.keras',
    'mob': 'MobileNetV2_fixed.keras',
}

AVAILABLE_MODELS = ['cnn', 'eff', 'mob']

_model_cache = {}
_load_locks  = {key: threading.Lock() for key in AVAILABLE_MODELS}


def log(msg):
    print(msg, flush=True)


# ---------------- CNN ARCHITECTURE ----------------
def build_cnn():
    model = Sequential([
        Input(shape=(48, 48, 3)),

        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Dropout(0.2),

        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Dropout(0.2),

        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Dropout(0.2),

        Flatten(),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


# ---------------- DOWNLOAD ----------------
def download_model(key, dest):
    file_id = DRIVE_IDS[key]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    log(f"Downloading {key}...")
    r = _requests.get(url, stream=True)

    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)


# ---------------- LOAD MODEL ----------------
def get_model(key):
    if key in _model_cache:
        return _model_cache[key]

    with _load_locks[key]:
        if key in _model_cache:
            return _model_cache[key]

        path = os.path.join(MODEL_DIR, MODEL_FILENAMES[key])

        if not os.path.exists(path):
            download_model(key, path)

        log(f"Loading {key}...")

        if key == 'cnn':
            model = build_cnn()
            model.load_weights(path)
        else:
            model = load_model(path, compile=False)

        _model_cache[key] = model
        return model


# ---------------- CLASS LABELS ----------------
CLASSES = {
    0:'Speed limit (20km/h)',        1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',        3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',        5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',       9:'No passing',
    10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road',              13:'Yield',
    14:'Stop',                       15:'No vehicles',
    16:'Vehicle > 3.5 tons prohibited', 17:'No entry',
    18:'General caution',            19:'Dangerous curve left',
    20:'Dangerous curve right',      21:'Double curve',
    22:'Bumpy road',                 23:'Slippery road',
    24:'Road narrows on the right',  25:'Road work',
    26:'Traffic signals',            27:'Pedestrians',
    28:'Children crossing',          29:'Bicycles crossing',
    30:'Beware of ice/snow',         31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead',
    34:'Turn left ahead',            35:'Ahead only',
    36:'Go straight or right',       37:'Go straight or left',
    38:'Keep right',                 39:'Keep left',
    40:'Roundabout mandatory',       41:'End of no passing',
    42:'End no passing vehicle > 3.5 tons'
}


# ---------------- OPTIMIZED PREPROCESSING ----------------
def preprocess_image(img_path, model_key):
    size = MODEL_IMG_SIZES[model_key]

    img = Image.open(img_path).convert('RGB').resize((size, size))

    arr = np.array(img, dtype=np.float32)

    # Normalize
    arr = arr / 255.0

    # Standardization (boosts confidence)
    mean = np.mean(arr)
    std = np.std(arr) + 1e-7
    arr = (arr - mean) / std

    # Stability clipping
    arr = np.clip(arr, -3, 3)

    return np.expand_dims(arr, axis=0)


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)


@app.route('/ready')
def ready():
    return jsonify({'all_ready': True})


@app.route('/predict', methods=['POST'])
def predict():
    file_path = None

    try:
        f = request.files['file']
        model_key = request.form.get('model', 'cnn')

        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        model = get_model(model_key)

        X = preprocess_image(file_path, model_key)

        start = time.time()
        pred = model.predict(X, verbose=0)
        print(f"{model_key} prediction took {time.time() - start:.2f}s")

        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)

        return jsonify({
            'result': CLASSES[pred_class],
            'confidence': f"{confidence:.1f}",
            'model': model_key.upper(),
            'class_id': pred_class
        })

    except Exception as e:
        return jsonify({'error': str(e)})

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


if __name__ == '__main__':
    app.run()