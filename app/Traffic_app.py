from flask import Flask, render_template, request, jsonify
import os
import sys
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
from keras.applications.efficientnet import preprocess_input as eff_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess

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

# ---------------- CNN ----------------
def build_cnn():
    model = Sequential([
        Input(shape=(48, 48, 3)),
        Conv2D(32, (3,3), activation='relu'), MaxPool2D(),
        Conv2D(64, (3,3), activation='relu'), MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# ---------------- Download ----------------
def download_model(key, dest):
    file_id = DRIVE_IDS[key]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    log(f"Downloading {key}...")
    r = _requests.get(url, stream=True)

    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk)

# ---------------- Load Model (LAZY) ----------------
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

# ---------------- Classes ----------------
CLASSES = {i: f"Class {i}" for i in range(43)}

# ---------------- Preprocess ----------------
def preprocess(img_path, model_key):
    size = MODEL_IMG_SIZES[model_key]
    img = Image.open(img_path).convert('RGB').resize((size, size))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)

@app.route('/ready')
def ready():
    # Always ready (no blocking warmup anymore)
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

        X = preprocess(file_path, model_key)

        start = time.time()
        pred = model.predict(X)
        log(f"Prediction time: {time.time() - start:.2f}s")

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