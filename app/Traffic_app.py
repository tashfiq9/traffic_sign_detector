from flask import Flask, render_template, request, jsonify
import os
import threading
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

import tensorflow as tf
from keras.models import load_model, Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Flatten,
    Dropout, BatchNormalization, Input
)
from keras.applications.efficientnet import preprocess_input as eff_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess
import gdown

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

NUM_CLASSES = 43

# ── Per-model image sizes ─────────────────────────────────────────────────
MODEL_IMG_SIZES = {
    'cnn': 48,
    'eff': 96,
    'mob': 96,
}

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Google Drive file IDs ─────────────────────────────────────────────────
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

# ── Thread-safe model cache ───────────────────────────────────────────────
# FIX 1: One lock per model key to prevent concurrent load/download races.
# FIX 2: One predict lock per model to prevent concurrent TF graph execution.
_model_cache   = {}
_load_locks    = {key: threading.Lock() for key in AVAILABLE_MODELS}
_predict_locks = {key: threading.Lock() for key in AVAILABLE_MODELS}

# ── Custom preprocessing layers ───────────────────────────────────────────
class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return eff_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()

class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return mob_preprocess(x * 255.0)
    def get_config(self):
        return super().get_config()

CUSTOM_OBJECTS = {
    'EfficientNetPreprocess': EfficientNetPreprocess,
    'MobileNetPreprocess':    MobileNetPreprocess,
}

# ── CNN architecture ──────────────────────────────────────────────────────
def build_cnn():
    model = Sequential([
        Input(shape=(MODEL_IMG_SIZES['cnn'], MODEL_IMG_SIZES['cnn'], 3)),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.2),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.2),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2,2)),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def _download_model(key, dest):
    """Download model from Google Drive with retry logic.
    FIX 3: Removes corrupt/incomplete files before retrying so a bad
    download never gets cached on disk and reused on the next restart.
    """
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Remove any previously broken download
            if os.path.exists(dest):
                os.remove(dest)
            print(f"Downloading '{key}' from Google Drive (attempt {attempt}/{MAX_RETRIES}) ...")
            gdown.download(id=DRIVE_IDS[key], output=dest, quiet=False)
            # Verify the file is non-empty
            if os.path.exists(dest) and os.path.getsize(dest) > 1024:
                print(f"  ✓ Download of '{key}' complete ({os.path.getsize(dest) // 1024} KB)")
                return
            raise RuntimeError(f"Downloaded file for '{key}' is missing or too small.")
        except Exception as e:
            print(f"  ✗ Download attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download model '{key}' after {MAX_RETRIES} attempts: {e}"
                )


def get_model(key):
    """Download (if needed) and load a model on first use — thread-safe."""
    # Fast path: already loaded
    if key in _model_cache:
        return _model_cache[key]

    # FIX 1: Acquire per-model lock so only one thread loads/downloads at a time.
    with _load_locks[key]:
        # Double-checked locking: another thread may have loaded while we waited.
        if key in _model_cache:
            return _model_cache[key]

        dest = os.path.join(MODEL_DIR, MODEL_FILENAMES[key])

        # Download if not on disk (or if the file looks corrupt / too small)
        if not os.path.exists(dest) or os.path.getsize(dest) < 1024:
            _download_model(key, dest)

        print(f"Loading '{key}' ...")
        try:
            if key == 'cnn':
                model = build_cnn()
                model.load_weights(dest)
            else:
                model = load_model(dest, compile=False, custom_objects=CUSTOM_OBJECTS)
        except Exception as e:
            # If loading fails the file on disk may be corrupt — delete it so
            # the next request triggers a fresh download instead of re-failing.
            if os.path.exists(dest):
                os.remove(dest)
            raise RuntimeError(f"Failed to load model '{key}': {e}")

        _model_cache[key] = model
        print(f"  ✓ '{key}' ready (input size: {MODEL_IMG_SIZES[key]}x{MODEL_IMG_SIZES[key]})")
        return model


# ── Class names ───────────────────────────────────────────────────────────
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

# ── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_image(img_path, model_key):
    img_size = MODEL_IMG_SIZES[model_key]
    image = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)


@app.route('/predict', methods=['POST'])
def predict():
    file_path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        model_key = request.form.get('model', 'cnn')
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f"Invalid model. Choose from: {AVAILABLE_MODELS}"}), 400

        # Save upload
        filename  = secure_filename(f.filename)
        # Use thread id in filename to avoid collisions between concurrent requests
        unique_name = f"{threading.get_ident()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        f.save(file_path)

        # Load model (thread-safe lazy load)
        model = get_model(model_key)

        # Preprocess
        X = preprocess_image(file_path, model_key)

        # FIX 4: Serialise TF predictions per model to avoid TF threading bugs.
        with _predict_locks[model_key]:
            pred = model.predict(X, verbose=0)

        pred_class = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred) * 100)
        label      = CLASSES.get(pred_class, 'Unknown')

        print(f"[{model_key.upper()}] class={pred_class} | conf={confidence:.1f}% | label={label}")
        return jsonify({
            'result':     label,
            'confidence': f"{confidence:.1f}",
            'model':      model_key.upper(),
            'class_id':   pred_class,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # FIX 2: Always clean up the temp file, even if an exception occurred.
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)