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

MODEL_IMG_SIZES = {
    'cnn': 48,
    'eff': 96,
    'mob': 96,
}

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

# ── Thread-safe model cache ───────────────────────────────────────────────
_model_cache   = {}
_load_locks    = {key: threading.Lock() for key in AVAILABLE_MODELS}
_predict_locks = {key: threading.Lock() for key in AVAILABLE_MODELS}

# ── Warmup status — lets /ready report progress to the browser ────────────
_warmup_status = {key: 'pending' for key in AVAILABLE_MODELS}

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
    """Download model with retry. Deletes corrupt files before retrying."""
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if os.path.exists(dest):
                os.remove(dest)
            print(f"Downloading '{key}' from Google Drive (attempt {attempt}/{MAX_RETRIES}) ...")
            gdown.download(id=DRIVE_IDS[key], output=dest, quiet=False)
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
    """Load model on first use — thread-safe with double-checked locking."""
    if key in _model_cache:
        return _model_cache[key]

    with _load_locks[key]:
        if key in _model_cache:
            return _model_cache[key]

        _warmup_status[key] = 'loading'
        dest = os.path.join(MODEL_DIR, MODEL_FILENAMES[key])

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
            _warmup_status[key] = 'failed'
            if os.path.exists(dest):
                os.remove(dest)
            raise RuntimeError(f"Failed to load model '{key}': {e}")

        _model_cache[key] = model
        _warmup_status[key] = 'ready'
        print(f"  ✓ '{key}' ready (input size: {MODEL_IMG_SIZES[key]}x{MODEL_IMG_SIZES[key]})")
        return model


def _warmup_all_models():
    """
    Runs in a background thread at startup.
    Loads all 3 models ONE AT A TIME (sequential) to avoid OOM on
    Render's 512 MB free tier. After this finishes, every predict
    request is instant — no cold download ever happens mid-request.
    """
    print("[Warmup] Starting background model warmup ...")
    for key in AVAILABLE_MODELS:   # cnn first (smallest), then eff, then mob
        try:
            get_model(key)
            print(f"[Warmup] '{key}' ✓")
        except Exception as e:
            print(f"[Warmup] '{key}' FAILED: {e}")
    print("[Warmup] All models ready.")


# Start warmup immediately when gunicorn imports this module.
# daemon=True so the thread doesn't block a clean shutdown.
threading.Thread(target=_warmup_all_models, daemon=True).start()


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


def preprocess_image(img_path, model_key):
    img_size = MODEL_IMG_SIZES[model_key]
    image = Image.open(img_path).convert('RGB').resize((img_size, img_size))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', available_models=AVAILABLE_MODELS)


@app.route('/ready')
def ready():
    """Health-check: visit /ready in your browser to see warmup progress."""
    all_ready = all(s == 'ready' for s in _warmup_status.values())
    return jsonify({
        'status': 'ready' if all_ready else 'warming_up',
        'models': _warmup_status,
    }), 200 if all_ready else 503


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

        # Block the request with a clear message if warmup hasn't finished yet.
        # This prevents gunicorn from timing out mid-download.
        status = _warmup_status.get(model_key, 'pending')
        if status in ('pending', 'loading'):
            return jsonify({
                'error': f"Model '{model_key.upper()}' is still loading, please wait a moment and try again."
            }), 503
        if status == 'failed':
            return jsonify({
                'error': f"Model '{model_key.upper()}' failed to load on startup. Please redeploy."
            }), 503

        filename    = secure_filename(f.filename)
        unique_name = f"{threading.get_ident()}_{filename}"
        file_path   = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        f.save(file_path)

        model = get_model(model_key)
        X     = preprocess_image(file_path, model_key)

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
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)