from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import gc

import tensorflow as tf

# 🔥 Prevent TensorFlow from grabbing too much memory (VERY IMPORTANT)
tf.config.set_visible_devices([], 'GPU')

from keras.models import load_model, Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten,
    Dropout, BatchNormalization, Input
)
from keras.applications.efficientnet import preprocess_input as eff_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
import gdown

app = Flask(__name__)

# ── Config ─────────────────────────────────────────
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

NUM_CLASSES = 43

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_IMG_SIZES = {
    'cnn': 48,
    'eff': 96,
    'mob': 96
}

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

# ── Custom preprocessing layers ───────────────────
class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return eff_preprocess(x * 255.0)

class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return mob_preprocess(x * 255.0)

CUSTOM_OBJECTS = {
    'EfficientNetPreprocess': EfficientNetPreprocess,
    'MobileNetPreprocess': MobileNetPreprocess,
}

# ── CNN model ─────────────────────────────────────
def build_cnn():
    model = Sequential([
        Input(shape=(48, 48, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(),
        Dropout(0.2),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# ── Load model ────────────────────────────────────
def get_model(key):
    if key in _model_cache:
        return _model_cache[key]

    dest = os.path.join(MODEL_DIR, MODEL_FILENAMES[key])

    if not os.path.exists(dest):
        print(f"Downloading {key} model...")
        gdown.download(id=DRIVE_IDS[key], output=dest, quiet=False)

    print(f"Loading {key} model...")

    if key == 'cnn':
        model = build_cnn()
        model.load_weights(dest)
    else:
        model = load_model(
            dest,
            compile=False,
            custom_objects=CUSTOM_OBJECTS,
            safe_mode=False
        )

    _model_cache[key] = model
    print(f"{key} model ready")

    return model

# ── Classes ───────────────────────────────────────
CLASSES = {i: f"Class {i}" for i in range(43)}  # you can replace with real names

# ── Preprocess image ──────────────────────────────
def preprocess_image(img_path, model_key):
    size = MODEL_IMG_SIZES.get(model_key, 48)

    image = Image.open(img_path).convert('RGB').resize((size, size))
    arr = np.array(image, dtype=np.float32) / 255.0

    return np.expand_dims(arr, axis=0)

# ── Routes ───────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        model_key = request.form.get('model', 'cnn')

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(file_path)

        # 🔥 keep only ONE model in memory
        _model_cache.clear()

        model = get_model(model_key)

        X = preprocess_image(file_path, model_key)
        print("MODEL:", model_key, "| SHAPE:", X.shape)

        try:
            # 🔥 MEMORY SAFE prediction
            pred = model(X, training=False).numpy()
        except Exception as pred_err:
            print("Prediction error:", pred_err)
            return jsonify({'error': 'Prediction failed'}), 500

        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)

        os.remove(file_path)

        gc.collect()  # 🔥 free memory

        return jsonify({
            'result': CLASSES.get(pred_class, "Unknown"),
            'confidence': f"{confidence:.2f}",
            'model': model_key.upper()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("ERROR:", str(e))
        return jsonify({'error': 'An error occurred'}), 500

# ── Run ───────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)